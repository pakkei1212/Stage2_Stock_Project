"""Telegram notification (Stage H) — downstream and non-critical.

Formats the post-run reports (HTML parse mode), splits them into chunks below
Telegram's 4096-character limit without splitting a stock entry, and sends
text + optional Stage-G VCP verification charts with bounded retries. Two
separate text deliveries: the decision-making dashboard (:func:`build_report`,
which reads the persisted Stage-I setup plans) and the optional operational
debug summary (:func:`build_debug_report`).

Credentials come only from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` in the
environment. The token is part of every Bot API URL, so nothing here ever logs
a URL, and every error string is passed through ``redact`` before it is logged
or persisted (``requests`` exceptions embed the full URL).
"""
import html
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from . import trend_analysis as ta

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 3800        # safety target below Telegram's 4096 limit
MAX_CAPTION_CHARS = 1000        # below the 1024 caption limit
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class TelegramError(Exception):
    def __init__(self, message, attempts=1):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class SendResult:
    ok: bool
    attempts: int
    message_ids: list
    error: Optional[str] = None


def esc(value):
    return html.escape("" if value is None else str(value), quote=False)


class TelegramClient:
    """Thin Bot API client. ``session`` (requests-like) and ``sleep`` are injectable."""

    def __init__(self, token, chat_id, *, session=None, sleep=time.sleep, max_attempts=3,
                 timeout=20, backoff_seconds=2.0):
        if not token or not chat_id:
            raise TelegramError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set")
        self._token = token
        self.chat_id = chat_id
        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._sleep = sleep
        self.max_attempts = max(1, int(max_attempts))
        self.timeout = timeout
        self.backoff_seconds = backoff_seconds

    @classmethod
    def from_env(cls, config, **kwargs):
        return cls(os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"),
                   max_attempts=config.get("telegram_max_attempts", 3), **kwargs)

    def redact(self, text):
        text = "" if text is None else str(text)
        return text.replace(self._token, "***") if self._token else text

    def _url(self, method):
        return f"{API_BASE}/bot{self._token}/{method}"

    def _call(self, method, data, files=None):
        """POST with bounded retries. Returns (message_id, attempts). Raises TelegramError (redacted)."""
        result, attempts = self._request(method, data, files)
        message_id = result.get("message_id") if isinstance(result, dict) else None
        return message_id, attempts

    def _request(self, method, data, files=None, timeout=None):
        """POST with bounded retries. Returns (result, attempts). Raises TelegramError (redacted)."""
        last_error = "unknown error"
        timeout = self.timeout if timeout is None else timeout
        for attempt in range(1, self.max_attempts + 1):
            retry_after = None
            try:
                if files:
                    for f in files.values():
                        f[1].seek(0)
                resp = self._session.post(self._url(method), data=data, files=files, timeout=timeout)
                status = getattr(resp, "status_code", None)
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if status == 200 and body.get("ok"):
                    return body.get("result"), attempt
                description = body.get("description") or f"HTTP {status}"
                last_error = f"Telegram {method} failed: HTTP {status}: {description}"
                retry_after = (body.get("parameters") or {}).get("retry_after")
                if status not in _RETRYABLE_STATUS:
                    raise TelegramError(self.redact(last_error), attempts=attempt)
            except TelegramError:
                raise
            except Exception as e:  # network error — message may embed the token-bearing URL
                last_error = f"Telegram {method} failed: {type(e).__name__}: {e}"
            last_error = self.redact(last_error)
            if attempt < self.max_attempts:
                delay = float(retry_after) if retry_after else self.backoff_seconds * attempt
                logger.warning("%s — retry %d/%d in %.0fs", last_error, attempt, self.max_attempts - 1, delay)
                self._sleep(delay)
        raise TelegramError(last_error, attempts=self.max_attempts)

    def send_message(self, text, reply_markup=None, chat_id=None):
        data = {"chat_id": chat_id or self.chat_id, "text": text, "parse_mode": "HTML",
                "disable_web_page_preview": "true"}
        if reply_markup is not None:
            data["reply_markup"] = json.dumps(reply_markup)
        return self._call("sendMessage", data)

    def send_photo(self, path, caption=None, reply_markup=None, chat_id=None):
        with open(path, "rb") as fh:
            data = {"chat_id": chat_id or self.chat_id}
            if caption:
                data["caption"] = caption[:MAX_CAPTION_CHARS]
                data["parse_mode"] = "HTML"
            if reply_markup is not None:
                data["reply_markup"] = json.dumps(reply_markup)
            return self._call("sendPhoto", data, files={"photo": (os.path.basename(path), fh)})

    # ── inbound (trade journal bot, pipeline/trade_bot.py) ────────────────
    # Reading updates never mutates anything by itself: pipeline.telegram_trade
    # decides what a message means, and every mutation still needs a Confirm.

    def get_updates(self, offset=None, timeout=0):
        data = {"timeout": int(timeout), "allowed_updates": json.dumps(["message", "callback_query"])}
        if offset is not None:
            data["offset"] = int(offset)
        result, attempts = self._request("getUpdates", data, timeout=self.timeout + int(timeout))
        return (result or []), attempts

    def answer_callback_query(self, callback_query_id, text=None):
        data = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text[:200]
        return self._request("answerCallbackQuery", data)

    def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
        """Drop (or replace) the inline keyboard of a message — used to retire the
        Confirm/Cancel buttons once an action has been resolved."""
        data = {"chat_id": chat_id, "message_id": message_id,
                "reply_markup": json.dumps(reply_markup or {"inline_keyboard": []})}
        return self._request("editMessageReplyMarkup", data)


def send_text_chunks(client, chunks, reply_markup=None):
    """Send chunks in order; ``reply_markup`` (if any) goes on the last one, so the
    buttons sit under the end of the report."""
    ids, total_attempts = [], 0
    try:
        for i, chunk in enumerate(chunks):
            markup = reply_markup if reply_markup and i == len(chunks) - 1 else None
            mid, attempts = (client.send_message(chunk, reply_markup=markup) if markup
                             else client.send_message(chunk))
            ids.append(mid)
            total_attempts += attempts
        return SendResult(True, total_attempts, ids)
    except TelegramError as e:
        return SendResult(False, total_attempts + e.attempts, ids, str(e))


# ── formatting ────────────────────────────────────────────────────────────

def chunk_blocks(header, blocks, limit=MAX_MESSAGE_CHARS):
    """Greedy packing of whole blocks into messages <= ``limit`` chars.

    Every message starts with ``header`` (continuations get a "(cont.)" marker).
    A single block longer than the limit is hard-split as a last resort.
    """
    cont_header = f"{header} <i>(cont.)</i>"
    messages, current = [], header
    for block in blocks:
        if not block:
            continue
        candidate = f"{current}\n\n{block}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current not in (header, cont_header):
            messages.append(current)
            current = cont_header
            candidate = f"{current}\n\n{block}"
            if len(candidate) <= limit:
                current = candidate
                continue
        # oversize single block: split on lines, then hard-cut
        for piece in _split_oversize(block, limit - len(cont_header) - 2):
            if current not in (header, cont_header):
                messages.append(current)
                current = cont_header
            current = f"{current}\n\n{piece}"
    if current not in (header, cont_header) or not messages:
        messages.append(current)
    return messages


def _split_oversize(block, size):
    pieces, buf = [], ""
    for line in block.split("\n"):
        while len(line) > size:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(line[:size])
            line = line[size:]
        if len(buf) + len(line) + 1 > size and buf:
            pieces.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces


def _fmt_int(v):
    return "–" if v is None else f"{int(v):,}"


def _fmt_score(v):
    if v is None:
        return "–"
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


# Numeric detector and Claude verdict are shown side by side, never merged into one score.
# Two separate questions: "which stocks have the best VCP setups?" (the Action Board,
# independent of rank) and "which stocks rank highest overall?" (Overall Leaders).
from .stage_vcp_analysis import NUMERIC_POSITIVE, vcp_agreement  # noqa: E402  (re-exported)

AGREEMENT_LABELS = ("agree_positive", "agree_negative",
                    "numeric_positive_claude_negative", "numeric_negative_claude_positive")


def _json_field(row, key):
    raw = row.get(key)
    if not raw:
        return {}
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def row_vcp_metrics(row):
    """The stored numeric metrics for a stock_results row ({} if absent/unreadable)."""
    return _json_field(row, "vcp_metrics_json")


def claude_state(row):
    """reviewed | not_requested | review_off | token_free | error | None (not a Stage-G row)."""
    status = row.get("vcp_status")
    if status == "analyzed":
        return "reviewed"
    if status == "error":
        return "error"
    if status == "skipped":
        reason = _json_field(row, "vcp_verdict_json").get("skipped")
        return {"not_requested": "not_requested", "llm_review_off": "review_off"}.get(reason, "token_free")
    return None


def _row_agreement(row):
    """Only for symbols Claude actually reviewed; skipped/not-requested never counts."""
    if claude_state(row) != "reviewed":
        return None
    return vcp_agreement(row_vcp_metrics(row).get("vcp_numeric_quality"), row.get("vcp_is_pattern"))


def _disagrees(row):
    agreement = _row_agreement(row)
    return bool(agreement) and not agreement.startswith("agree")


def vcp_priority(row):
    """VCP relevance (not Composite rank): 0 numeric strong, 1 acceptable, 2 numeric/Claude
    disagreement, 3 other Claude-positive; None = not in the VCP review set."""
    quality = row_vcp_metrics(row).get("vcp_numeric_quality")
    if quality == "strong":
        return 0
    if quality == "acceptable":
        return 1
    if _disagrees(row):
        return 2
    if claude_state(row) == "reviewed" and row.get("vcp_is_pattern"):
        return 3
    return None


def vcp_review_set(results):
    """Union of numeric positive, Claude positive and disagreements, deduplicated and
    ordered by VCP priority, then Composite rank."""
    seen, picked = set(), []
    for row in results:
        priority = vcp_priority(row)
        if priority is None or row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        picked.append((priority, row.get("final_rank") or 10 ** 9, row))
    return [row for _, _, row in sorted(picked, key=lambda p: (p[0], p[1]))]


def select_vcp_chart_rows(results, max_charts):
    """VCP review set rows (priority order) that have a stored verification chart."""
    rows = [r for r in vcp_review_set(results) if r.get("vcp_debug_chart_path")]
    return rows[:max(0, int(max_charts))]


def _sequence(m):
    pcts = m.get("contraction_pcts") or []
    if not pcts:
        return "no contractions"
    seq = " → ".join(f"{float(p):.1f}" for p in pcts) + "%"
    return seq + (" (forming)" if m.get("final_contraction_confirmed") is False else "")


def _pivot_text(m, with_state=True):
    if m.get("pct_from_pivot") is None:
        return None
    text = f"{float(m['pct_from_pivot']):+.1f}%"
    if with_state and m.get("pivot_state"):
        text += f" · {esc(str(m['pivot_state']).replace('_pivot', '').replace('_', ' '))}"
    return text


def _claude_line(row):
    state = claude_state(row)
    if state == "reviewed":
        parts = ["VCP" if row.get("vcp_is_pattern") else "no VCP", row.get("vcp_pattern_stage"),
                 row.get("vcp_entry_recommendation"), row.get("vcp_confidence")]
        return " · ".join(esc(p) for p in parts if p)
    quality = row_vcp_metrics(row).get("vcp_numeric_quality")
    return {"not_requested": f"not requested (clear numeric {'setup' if quality in NUMERIC_POSITIVE else 'result'})",
            "review_off": "not run (review mode off)",
            "token_free": "not run (token-free mode)",
            "error": "analysis error"}.get(state)


def _vcp_summary_lines(counts, results):
    stage_g = [r for r in results if claude_state(r) is not None]
    computed = counts.get("vcp_analyzed", len(stage_g) if stage_g else None)
    if computed is None:
        return []
    states = [claude_state(r) for r in stage_g]
    if stage_g and all(s == "token_free" for s in states):
        computed_line = f"VCP metrics computed: {_fmt_int(computed)} (Claude vision off: token-free mode)"
    else:
        computed_line = f"VCP metrics computed: {_fmt_int(computed)}"
    numeric_positive = sum(row_vcp_metrics(r).get("vcp_numeric_quality") in NUMERIC_POSITIVE for r in stage_g)
    lines = [computed_line, f"Numeric VCP positive: {numeric_positive}"]
    reviewed = [r for r in stage_g if claude_state(r) == "reviewed"]
    if reviewed or any(s in ("not_requested", "error") for s in states):
        lines.append(f"Claude reviewed: {len(reviewed)}")
        lines.append(f"Claude VCP positive: {sum(bool(r.get('vcp_is_pattern')) for r in reviewed)}")
        errors = states.count("error")
        if errors:
            lines.append(f"Claude errors: {errors}")
    lines.append(f"VCP review union: {len(vcp_review_set(results))}")
    agreements = [a for a in (_row_agreement(r) for r in reviewed) if a]
    if agreements:
        agree = sum(a.startswith("agree") for a in agreements)
        lines.append(f"Numeric vs Claude: {agree} agree · {len(agreements) - agree} disagree")
    return lines


STAGE_SUMMARY_LINES = (
    ("universe", "Universe"),
    ("liquidity", "Liquidity"),
    ("market_cap", "Market cap"),
    ("stage_c", "Top sectors"),
    ("stage_d", "Stage D screened"),
    ("trend_template_pass", "Trend template 8/8"),
    ("ranked", "Ranked"),
)


# ── daily dashboard (main report) ─────────────────────────────────────────
# Presentation only. The buckets are read off the setup_plans rows Stage I
# persisted for the session (setup_gate state + the Layer-2/3 plan it stored);
# nothing here recomputes an entry, a stop or a size, and nothing is a new score.
# The state strings are the persisted values (setup_gate / entry_plan /
# risk_model import this module, so they cannot be imported back here).

PLANNABLE_STATES = ("READY", "BREAKOUT_TRIGGERED")
PLAN_OK = "OK"
USABLE_ENTRY_STATUSES = ("AWAIT_BREAKOUT", "BREAKOUT_ACTIONABLE", "BREAKOUT_VOLUME_UNCONFIRMED")
ENTRY_ACTIONS = {
    "AWAIT_BREAKOUT": "WAIT FOR BREAKOUT",
    "BREAKOUT_ACTIONABLE": "BREAKOUT — WITHIN ENTRY RANGE",
    "BREAKOUT_VOLUME_UNCONFIRMED": "BREAKOUT — VOLUME NOT CONFIRMED",
}

READY, WATCH, REVIEW, NO_SETUP = "ready", "watch", "review", "none"
BUCKET_TAGS = {READY: "🟢 Ready", WATCH: "🟡 Watch", REVIEW: "🟠 Review", NO_SETUP: "⚪ No VCP"}

# Callback prefixes of the dashboard buttons, answered by pipeline.telegram_trade
# (the bot process). They only start a journal conversation or show a record —
# there is no brokerage connection anywhere in this project.
OPEN_POSITION_PREFIX = "op"
VIEW_POSITION_PREFIX = "vp"


@dataclass
class SetupView:
    """Where one Stage-G row sits on the Action Board, and why."""
    symbol: str
    row: dict
    plan: Optional[dict]
    bucket: str
    action: Optional[str] = None
    reasons: list = None


def plan_is_usable(plan):
    """A persisted READY/BREAKOUT_TRIGGERED plan that can back a trade: the risk plan
    fits the ceiling, the entry is still inside the plan and it sizes to >0 shares."""
    return (bool(plan) and plan.get("setup_state") in PLANNABLE_STATES
            and plan.get("trade_plan_status") == PLAN_OK
            and plan.get("entry_status") in USABLE_ENTRY_STATUSES
            and float(plan.get("suggested_shares") or 0) > 0)


def _unusable_plan_reason(plan):
    status, entry = plan.get("trade_plan_status"), plan.get("entry_status")
    if status == "SKIP_RISK_TOO_WIDE":
        return ("NO TRADE — STOP TOO WIDE",
                f"⚠ stop needs {_fmt_num(plan.get('required_risk_pct'))}% risk, above the "
                f"{_fmt_num(plan.get('allowed_max_risk_pct'))}% allowed (SKIP_RISK_TOO_WIDE)")
    if status == "NO_PRICE_DATA":
        return "WAIT — PLAN INCOMPLETE", "⚠ no price data for a risk plan (NO_PRICE_DATA)"
    if entry == "EXTENDED_NO_CHASE":
        return "DO NOT CHASE", "⚠ price is beyond the maximum chase price"
    if entry == "NO_PIVOT":
        return "WAIT — PLAN INCOMPLETE", "⚠ no numeric pivot in the stored metrics"
    return "WAIT — PLAN INCOMPLETE", "⚠ the plan sizes this trade to 0 shares"


def _structure_reasons(m):
    """Readable ✓/⚠ lines from the stored detector components (no new analysis)."""
    good = NUMERIC_POSITIVE
    out = []
    tightening = m.get("tightening_quality")
    if tightening in good:
        out.append("✓ price contractions are tightening")
    elif tightening:
        out.append(f"⚠ price tightening is {esc(tightening.replace('_', ' '))}")
    volume = m.get("volume_dryup_quality")
    if volume in good:
        out.append("✓ volume dries up through the base")
    elif volume:
        out.append(f"⚠ volume confirmation is {'absent' if volume == 'none' else esc(volume)}")
    if m.get("base_coherence") == "weak":
        out.append("⚠ base structure is choppy (weak coherence)")
    if m.get("final_tightness") == "loose":
        out.append("⚠ final contraction is not tight")
    if m.get("final_contraction_confirmed") is False:
        out.append("⚠ final contraction still forming")
    p = m.get("pct_from_pivot")
    if m.get("pivot_state") == "far_below_pivot" and p is not None:
        out.append(f"⚠ price {abs(float(p)):.1f}% below pivot — not yet in entry range")
    elif m.get("pivot_state") == "extended_above_pivot" and p is not None:
        out.append(f"⚠ price {float(p):.1f}% above pivot — past the entry range")
    return out


def _claude_rationale(row, limit=200):
    text = _json_field(row, "vcp_verdict_json").get("rationale")
    if not text:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _review_reasons(row, quality):
    reviewed = claude_state(row) == "reviewed"
    q = (quality or "unrated").upper() if quality else "UNRATED"
    if reviewed and not row.get("vcp_is_pattern"):
        out = [f"Numeric detector rates it {esc(q)}, but Claude does not see one coherent VCP."]
    elif reviewed:
        stage = row.get("vcp_pattern_stage")
        out = [f"Numeric contraction quality is {esc(q)}, but Claude sees "
               f"{'a ' + esc(stage) if stage else 'a'} VCP."]
    else:
        out = [f"Numeric detector rates it {esc(q)}; no Claude review to confirm it."]
    rationale = _claude_rationale(row)
    if rationale:
        out.append(f"Claude: “{esc(rationale)}”")
    notes = row_vcp_metrics(row).get("vcp_quality_notes") or []
    out += [f"⚠ {esc(n)}" for n in notes[:2]]
    return out


def classify_setup(row, plan=None):
    """Action-Board bucket for one ranked row, read from its persisted Stage-I plan.

    READY   READY/BREAKOUT_TRIGGERED with a usable trade plan
    WATCH   promising but not actionable (numeric setup below the entry range,
            extended, or a plannable setup whose plan is not usable)
    REVIEW  MANUAL_REVIEW, or a numeric/Claude disagreement
    none    ranked stock with no VCP worth acting on (or never analysed)
    """
    symbol = row["symbol"]
    if claude_state(row) is None:
        return SetupView(symbol, row, plan, NO_SETUP)
    m = row_vcp_metrics(row)
    quality = m.get("vcp_numeric_quality")
    state = (plan or {}).get("setup_state")

    if state in PLANNABLE_STATES:
        if plan_is_usable(plan):
            return SetupView(symbol, row, plan, READY, ENTRY_ACTIONS[plan["entry_status"]], [])
        action, reason = _unusable_plan_reason(plan)
        return SetupView(symbol, row, plan, WATCH, action, _structure_reasons(m) + [reason])
    if state == "MANUAL_REVIEW":
        return SetupView(symbol, row, plan, REVIEW, "MANUAL REVIEW", _review_reasons(row, quality))
    if state == "MISSED_EXTENDED":
        return SetupView(symbol, row, plan, WATCH, "DO NOT CHASE", _structure_reasons(m))
    if _disagrees(row) or vcp_priority(row) == 3:
        action = "NOT READY" if quality in NUMERIC_POSITIVE else "MANUAL REVIEW"
        return SetupView(symbol, row, plan, REVIEW, action, _review_reasons(row, quality))
    if quality in NUMERIC_POSITIVE:
        reasons = _structure_reasons(m)
        if plan is None:
            reasons.append("⚠ no Stage-I plan recorded for this session")
        return SetupView(symbol, row, plan, WATCH, "WAIT", reasons)
    return SetupView(symbol, row, plan, NO_SETUP)


def classify_setups(results, plans=None):
    """{symbol: SetupView} for every ranked row. ``plans`` is {symbol: setup_plans row}."""
    plans = plans or {}
    return {r["symbol"]: classify_setup(r, plans.get(r["symbol"])) for r in results}


def _bucket_order(views, results, bucket):
    """Views of one bucket: VCP priority first, then Composite rank."""
    order = {r["symbol"]: i for i, r in enumerate(vcp_review_set(results))}
    picked = [v for v in views.values() if v.bucket == bucket]
    return sorted(picked, key=lambda v: (order.get(v.symbol, 10 ** 6), v.row.get("final_rank") or 10 ** 9))


def _fmt_num(value, digits=1):
    return "–" if value is None else f"{float(value):.{digits}f}"


def _score_text(row):
    if row.get("composite_score") is None:
        return None
    text = _fmt_score(row["composite_score"])
    if row.get("max_score") is not None:
        text += f"/{_fmt_score(row['max_score'])}"
    return text


def _rank_line(row):
    parts = [f"Rank #{row.get('final_rank')}", esc(row.get("sector")) if row.get("sector") else None,
             _score_text(row)]
    return " · ".join(p for p in parts if p)


def _open_map(open_trades):
    """{symbol: trade} from a list of trade dicts (or a dict / set already)."""
    if not open_trades:
        return {}
    if isinstance(open_trades, dict):
        return open_trades
    if isinstance(open_trades, (set, frozenset)):
        return {s: {} for s in open_trades}
    return {t["symbol"]: t for t in open_trades}


def _held_line(trade):
    if trade and trade.get("shares") is not None:
        return (f"💼 Already held: {_fmt_price(trade['shares'], 0)} @ {_fmt_price(trade.get('average_cost'))}"
                " — see View Position")
    return "💼 Already held — see View Position"


def _ready_block(view, held=None):
    row, plan, m = view.row, view.plan, row_vcp_metrics(view.row)
    entry, risk, sizing = plan.get("entry_plan") or {}, plan.get("risk_plan") or {}, plan.get("sizing") or {}
    title = "BREAKOUT" if plan.get("setup_state") == "BREAKOUT_TRIGGERED" else "READY"
    lines = [f"🟢 <b>{title} — {esc(view.symbol)}</b>", _rank_line(row),
             f"VCP: {esc(str(m.get('vcp_numeric_quality') or 'n/a').upper())} · {esc(_sequence(m))}"]
    if m.get("volume_dryup_quality"):
        lines.append(f"Volume: {esc(m['volume_dryup_quality'].upper())}")
    distance = entry.get("distance_to_pivot_pct", m.get("pct_from_pivot"))
    lines.append(f"Price: {_fmt_price(plan.get('current_price'))} · Pivot: {_fmt_price(plan.get('pivot_price'))}"
                 + (f" · {_fmt_pct(distance)}" if distance is not None else ""))

    lines.append("📌 <b>Entry plan</b>")
    lines.append(f"Trigger: {_fmt_price(plan.get('entry_trigger_price'))} · "
                 f"Max chase: {_fmt_price(plan.get('maximum_chase_price'))}")
    if entry.get("required_breakout_volume_ratio") is not None:
        lines.append(f"Breakout volume: ≥{float(entry['required_breakout_volume_ratio']):.2f}× 20D avg")
    lines.append(f"Status: {esc(str(plan.get('entry_status')).replace('_', ' '))}")

    lines.append("🛡 <b>Risk plan</b>")
    lines.append(f"Initial stop: {_fmt_price(plan.get('suggested_initial_stop'))}")
    lines.append(f"Required risk: {_fmt_num(plan.get('required_risk_pct'))}% · "
                 f"Allowed max: {_fmt_num(plan.get('allowed_max_risk_pct'))}%")
    if risk.get("atr_pct") is not None:
        lines.append(f"ATR: {float(risk['atr_pct']):.1f}%"
                     + (f" ({_fmt_price(risk['atr20'])})" if risk.get("atr20") is not None else ""))

    lines.append("📦 <b>Position plan</b>")
    portion = plan.get("position_portion_pct")
    lines.append(f"Suggested: {_fmt_price(plan.get('suggested_shares'), 0)} shares"
                 + (f" · {float(portion):.1f}% of portfolio" if portion is not None else ""))
    if sizing.get("binding_constraint"):
        binding = {"risk": "risk budget", "portion": "portfolio portion"}.get(sizing["binding_constraint"],
                                                                              sizing["binding_constraint"])
        lines.append(f"Binding constraint: {esc(binding)}")
    lines.append(f"Trade plan: {esc(plan.get('trade_plan_status'))}")

    claude = _claude_line(row)
    if claude:
        lines.append(f"Claude: {claude}")
    if held is not None:
        lines.append(_held_line(held))
    lines.append(f"Action: <b>{esc(view.action)}</b>")
    return "\n".join(lines)


def _watch_block(view):
    row, m = view.row, row_vcp_metrics(view.row)
    lines = [f"🟡 <b>WATCH — {esc(view.symbol)}</b>", _rank_line(row),
             f"VCP: {esc(str(m.get('vcp_numeric_quality') or 'n/a').upper())} · {esc(_sequence(m))}"]
    second = []
    if m.get("volume_dryup_quality"):
        second.append(f"Volume: {esc(m['volume_dryup_quality'].upper())}")
    if m.get("pct_from_pivot") is not None:
        second.append(f"Pivot: {_fmt_pct(m['pct_from_pivot'])}")
    if second:
        lines.append(" · ".join(second))
    if view.reasons:
        lines.append("Why watch:")
        lines += view.reasons[:6]
    if _claude_line(row):
        lines.append(f"Claude: {_claude_line(row)}")
    lines.append(f"Action: <b>{esc(view.action)}</b>")
    return "\n".join(lines)


def _review_block(view):
    row, m = view.row, row_vcp_metrics(view.row)
    lines = [f"🟠 <b>REVIEW — {esc(view.symbol)}</b>", _rank_line(row),
             f"Numeric: {esc(str(m.get('vcp_numeric_quality') or 'n/a').upper())} · {esc(_sequence(m))}"]
    claude = _claude_line(row)
    if claude:
        lines.append(f"Claude: {claude}")
    if view.reasons:
        lines.append("Why review:")
        lines += view.reasons
    lines.append(f"Action: <b>{esc(view.action)}</b>")
    return "\n".join(lines)


def _action_board(views, results, positions, open_trades):
    def names(bucket):
        picked = _bucket_order(views, results, bucket)
        return ", ".join(esc(v.symbol) for v in picked) if picked else "none"

    held = len(positions) if positions else len(_open_map(open_trades))
    attention = sum((s.get("monitor_state") or "HOLD") != "HOLD" for s in positions or [])
    lines = ["🎯 <b>ACTION BOARD</b>",
             f"🟢 Ready: {names(READY)}",
             f"🟡 Watch: {names(WATCH)}",
             f"🟠 Review: {names(REVIEW)}",
             f"💼 Open positions: {held}" + (f" ({attention} need attention)" if attention else "")]
    if not any(claude_state(r) is not None for r in results):
        lines.append("<i>No Stage-G VCP analysis in this run.</i>")
    return "\n".join(lines)


def _leaders_block(results, views, max_candidates, open_trades):
    candidates = list(results)[:max_candidates]
    if not candidates:
        return "📊 <b>OVERALL LEADERS</b>\n<i>No ranked candidates.</i>"
    held = _open_map(open_trades)
    lines = ["📊 <b>OVERALL LEADERS</b>"]
    for row in candidates:
        view = views.get(row["symbol"])
        tag = BUCKET_TAGS[view.bucket] if view and claude_state(row) is not None else "⚪ Not analysed"
        parts = [f"#{row.get('final_rank')} <b>{esc(row['symbol'])}</b>", _score_text(row),
                 esc(row.get("sector")) if row.get("sector") else None, tag]
        line = " · ".join(p for p in parts if p)
        if row["symbol"] in held:
            line += " · 💼"
        lines.append(line)
    return "\n".join(lines)


def _changes_block(trend):
    top_n = trend.top_n
    if not trend.has_history:
        return "📈 <b>WATCHLIST CHANGES</b>\n<i>First recorded session — no trend history yet.</i>"
    note = f"vs {esc(trend.previous_session)}"
    if trend.missing_sessions:
        note += f" · {len(trend.missing_sessions)} session(s) in lookback not recorded"
    lines = [f"📈 <b>WATCHLIST CHANGES</b> <i>({note})</i>"]
    by_symbol = {e.symbol: e for e in trend.entries}

    def new_entry(symbol):
        e = by_symbol.get(symbol)
        if e is None:
            return esc(symbol)
        came_from = f"#{e.previous_rank} → " if e.previous_rank is not None else ""
        return f"{esc(symbol)} {came_from}#{e.rank}"

    if trend.new_entries:
        lines.append("🆕 New: " + ", ".join(new_entry(s) for s in trend.new_entries))
    rising = [e for e in trend.by_category(ta.RISING) if e.symbol not in trend.new_entries]
    if rising:
        lines.append("⬆ Up: " + ", ".join(
            f"{esc(e.symbol)} #{e.previous_rank if e.previous_rank is not None else '–'} → #{e.rank}"
            for e in rising))
    falling = trend.by_category(ta.FALLING)
    if falling:
        lines.append("⬇ Down: " + ", ".join(f"{esc(e.symbol)} #{e.previous_rank} → #{e.rank}" for e in falling))
    if trend.dropped:
        lines.append(f"🚪 Left Top {top_n}: " + ", ".join(
            f"{esc(d.symbol)} #{d.previous_rank} → {'#' + str(d.current_rank) if d.current_rank else 'unranked'}"
            for d in trend.dropped))
    persistent = sorted(trend.by_category(ta.PERSISTENT), key=lambda e: e.rank)
    if persistent:
        lines.append("🔥 Persistent: " + ", ".join(esc(e.symbol) for e in persistent))
    if len(lines) == 1:
        lines.append(f"<i>No changes in the Top {top_n}.</i>")
    return "\n".join(lines)


def build_report(trading_date, stage_counts, results, trend, *, max_candidates=10, positions=None,
                 plans=None, open_trades=None):
    """Return the main (decision-making) report as a list of HTML message chunks.

    results      stock_results dicts for the run, ordered by rank
    trend        trend_analysis.TrendReport
    positions    Stage-I position_snapshots rows for the session. Open positions
                 are monitored independently of today's screen, so this section
                 can list symbols that are not in ``results`` at all.
    plans        {symbol: persisted setup_plans row} for the run — the only source
                 of READY / entry / risk / sizing values (never recomputed here)
    open_trades  the journal's OPEN trades (to mark what is already held)

    Pipeline/stage counts live in :func:`build_debug_report`, not here;
    ``stage_counts`` is accepted for signature compatibility.
    """
    header = f"📈 <b>NASDAQ Stage 2 / VCP</b>\nTrading session: <b>{esc(trading_date)}</b>"
    views = classify_setups(results, plans)
    held = _open_map(open_trades)
    blocks = [_action_board(views, results, positions, open_trades)]
    blocks += [_ready_block(v, held.get(v.symbol) if v.symbol in held else None)
               for v in _bucket_order(views, results, READY)]
    blocks += [_watch_block(v) for v in _bucket_order(views, results, WATCH)]
    blocks += [_review_block(v) for v in _bucket_order(views, results, REVIEW)]
    blocks += build_positions_blocks(positions)
    blocks.append(_leaders_block(results, views, max_candidates, open_trades))
    blocks.append(_changes_block(trend))
    return chunk_blocks(header, blocks)


# ── dashboard buttons ─────────────────────────────────────────────────────
# "Open Position" records the user's OWN fill in the local journal (after a
# price/shares conversation and a Confirm); "View Position" is /position SYMBOL.
# Neither places, modifies or cancels anything anywhere.

def position_button(symbol, held):
    symbol = str(symbol).upper()
    if held:
        return {"text": f"💼 View Position · {symbol}", "callback_data": f"{VIEW_POSITION_PREFIX}:{symbol}"}
    return {"text": f"🟢 Open Position · {symbol}", "callback_data": f"{OPEN_POSITION_PREFIX}:{symbol}"}


def position_keyboard(symbol, held):
    """Single-button keyboard for a READY chart (Open, or View when already held)."""
    button = position_button(symbol, held)
    button["text"] = button["text"].split(" · ")[0]
    return {"inline_keyboard": [[button]]}


def button_symbols(results, plans, open_trades):
    """READY symbols (usable plan) in dashboard order — the only ones that get a
    position button. MANUAL_REVIEW / NOT_READY / SKIP_RISK_TOO_WIDE /
    MISSED_EXTENDED never do."""
    views = classify_setups(results, plans)
    return [v.symbol for v in _bucket_order(views, results, READY)]


def build_report_keyboard(results, plans, open_trades, positions, *, exclude=()):
    """Inline keyboard for the last report chunk, or None.

    One button per READY setup (Open Position, or View Position when an OPEN trade
    exists) and one View Position per held symbol. ``exclude`` are READY symbols
    whose VCP chart carries the button instead, so no button appears twice.
    """
    held = _open_map(open_trades)
    buttons, seen = [], set(exclude)
    for symbol in button_symbols(results, plans, open_trades):
        if symbol not in seen:
            seen.add(symbol)
            buttons.append(position_button(symbol, symbol in held))
    order = {state: i for i, state in enumerate(_POSITION_ORDER)}
    for snap in sorted(positions or [], key=lambda s: (order.get(s.get("monitor_state"), 99), s.get("symbol") or "")):
        symbol = snap.get("symbol")
        if symbol and symbol not in seen and (not held or symbol in held):
            seen.add(symbol)
            buttons.append(position_button(symbol, True))
    if not buttons:
        return None
    return {"inline_keyboard": [buttons[i:i + 2] for i in range(0, len(buttons), 2)]}


# ── separate debug message (operational) ──────────────────────────────────
# Everything a developer wants to know about the run and nothing the daily
# decision needs. Never includes credentials, raw environment values, config
# JSON or local file paths.

_GATE_STATES = ("READY", "BREAKOUT_TRIGGERED", "NOT_READY", "MANUAL_REVIEW", "MISSED_EXTENDED")
_POSITION_LABELS = (("HOLD", "HOLD"), ("WATCH", "WATCH"), ("TIGHTEN_PROTECTION", "TIGHTEN"),
                    ("PARTIAL_PROFIT_REVIEW", "PARTIAL PROFIT REVIEW"), ("EXIT_REVIEW", "EXIT REVIEW"),
                    ("STOP_TRIGGERED", "STOP TRIGGERED"), ("CORPORATE_ACTION_REVIEW", "CORPORATE ACTION REVIEW"))


def _runtime(run):
    from datetime import datetime
    start, end = (run or {}).get("pipeline_started_at_utc"), (run or {}).get("completed_at_utc")
    try:
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None
    minutes, secs = divmod(int(max(0, seconds)), 60)
    return f"{minutes}m {secs:02d}s"


def build_debug_report(trading_date, stage_counts, results, *, run=None, plans=None, positions=None,
                       open_trade_count=None, corporate_actions=None, config=None, buttons_enabled=None,
                       report_status=None):
    """The separate 🛠 debug message: stage counts, Stage-G/Stage-I/monitor and
    corporate-action counts, and run facts. Returns HTML chunks."""
    header = f"🛠 <b>DEBUG — DAILY PIPELINE</b>\nSession: {esc(trading_date)}"
    counts = stage_counts or {}
    blocks = []

    stage = [f"{label}: {_fmt_int(counts.get(key))}" for key, label in STAGE_SUMMARY_LINES
             if counts.get(key) is not None]
    if stage:
        blocks.append("<b>Stage A–E</b>\n" + "\n".join(stage))
    vcp = _vcp_summary_lines(counts, results)
    if vcp:
        blocks.append("<b>Stage G — VCP</b>\n" + "\n".join(vcp))

    plan_rows = list((plans or {}).values()) if isinstance(plans, dict) else list(plans or [])
    stage_i = [f"Setup plans: {len(plan_rows)}"]
    if plan_rows:
        states = [p.get("setup_state") for p in plan_rows]
        stage_i += [f"{s}: {states.count(s)}" for s in _GATE_STATES]
        plannable = [p for p in plan_rows if p.get("setup_state") in PLANNABLE_STATES]
        statuses = [p.get("trade_plan_status") for p in plannable]
        stage_i.append(f"Trade plan OK: {statuses.count(PLAN_OK)}")
        stage_i.append(f"SKIP_RISK_TOO_WIDE: {statuses.count('SKIP_RISK_TOO_WIDE')}")
        if statuses.count("NO_PRICE_DATA"):
            stage_i.append(f"NO_PRICE_DATA: {statuses.count('NO_PRICE_DATA')}")
    views = classify_setups(results, {p.get("symbol"): p for p in plan_rows})
    buckets = [v.bucket for v in views.values()]
    stage_i.append(f"Dashboard: 🟢 {buckets.count(READY)} ready · 🟡 {buckets.count(WATCH)} watch · "
                   f"🟠 {buckets.count(REVIEW)} review")
    if buttons_enabled is not None:
        stage_i.append(f"READY setup buttons: {buckets.count(READY)}" if buttons_enabled
                       else "READY setup buttons: off (TRADE_BOT_ENABLED=0)")
    blocks.append("<b>Stage I — setups</b>\n" + "\n".join(stage_i))

    snaps = positions or []
    monitor = [f"Open positions: {open_trade_count if open_trade_count is not None else len(snaps)}",
               f"Positions monitored: {len(snaps)}"]
    monitor += [f"{label}: {sum(s.get('monitor_state') == state for s in snaps)}"
                for state, label in _POSITION_LABELS]
    blocks.append("<b>Positions</b>\n" + "\n".join(monitor))

    if corporate_actions is None:
        blocks.append("<b>Corporate actions</b>\n<i>Not re-checked for a resend.</i>")
    else:
        statuses = [a.get("status") for a in corporate_actions]
        blocks.append("<b>Corporate actions</b>\n" + "\n".join([
            f"Detected: {len(statuses)}",
            f"Applied automatically: {statuses.count('APPLIED')}",
            f"Review required: {sum(s in ('REVIEW_REQUIRED', 'AWAITING_CONFIRMATION') for s in statuses)}"]))

    run_lines = []
    if run:
        run_lines.append(f"Run id: <code>{esc(run.get('run_id'))}</code>")
        run_lines.append(f"Status: {esc(run.get('status'))}" + (" (canonical)" if run.get("is_canonical") else ""))
        if _runtime(run):
            run_lines.append(f"Runtime: {_runtime(run)}")
        if run.get("data_ready_latest_bar"):
            run_lines.append(f"Data readiness: latest bar {esc(run['data_ready_latest_bar'])}"
                             + (f" · {run['data_ready_attempts']} attempt(s)"
                                if run.get("data_ready_attempts") is not None else ""))
        if run.get("config_hash"):
            run_lines.append(f"Config hash: {esc(run['config_hash'])}")
    if config is not None:
        mode = config.get("vcp_llm_review_mode") if config.get("vcp_live_analysis", True) else "off (token-free)"
        run_lines.append(f"Claude mode: {esc(mode)}")
    if report_status:
        run_lines.append(f"Main report: {esc(report_status)}")
    if run_lines:
        blocks.append("<b>Run</b>\n" + "\n".join(run_lines))
    return chunk_blocks(header, blocks)


# ── open positions (Stage I, advisory) ────────────────────────────────────
# A separate section after the screener/VCP report, built from the stored
# position_snapshots of the session. Nothing here places, modifies or cancels an
# order — the wording must never suggest that it did.

POSITION_ICONS = {
    "HOLD": "", "WATCH": "⚠", "TIGHTEN_PROTECTION": "🔧",
    "PARTIAL_PROFIT_REVIEW": "💰", "EXIT_REVIEW": "🔴", "STOP_TRIGGERED": "🛑",
    "CORPORATE_ACTION_REVIEW": "🔄",
}
_REVIEW_STATES = ("EXIT_REVIEW", "STOP_TRIGGERED", "CORPORATE_ACTION_REVIEW")


def _fmt_price(value, digits=2):
    return "–" if value is None else f"{float(value):,.{digits}f}"


def _fmt_pct(value, digits=1):
    return "–" if value is None else f"{float(value):+.{digits}f}%"


def snapshot_reasons(snap):
    """The monitor's reasons for a snapshot row (live dict or stored JSON)."""
    if isinstance(snap.get("reasons"), list):
        return [str(r) for r in snap["reasons"]]
    raw = snap.get("reasons_json")
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _position_block(snap):
    state = snap.get("monitor_state") or "HOLD"
    icon = POSITION_ICONS.get(state, "")
    reasons = snapshot_reasons(snap)
    head = f"{icon} <b>{esc(snap.get('symbol'))}</b>".strip()
    held = f"{_fmt_price(snap.get('shares'), 0)} shares @ {_fmt_price(snap.get('average_cost'))}"
    close = f"Close: {_fmt_price(snap.get('close'))} · {_fmt_pct(snap.get('pnl_pct'))}"

    if state in _REVIEW_STATES:
        lines = [f"{head} — {esc(state.replace('_', ' '))}", held, close]
        if reasons:
            lines.append("Reason:")
            lines += [f"• {esc(r)}" for r in reasons]
        lines += _corporate_action_lines(snap)
        lines.append("Suggested action: review position")
        return "\n".join(lines)

    if state == "HOLD":
        # Normal positions: only what a daily decision needs.
        lines = [head, held, close]
        if snap.get("current_protective_level") is not None:
            lines.append(f"Protective level: {_fmt_price(snap['current_protective_level'])}")
        lines += _corporate_action_lines(snap)
        lines.append(f"Status: {esc(state)}")
        return "\n".join(lines)

    lines = [f"{head} — {esc(state.replace('_', ' '))}", held, close]
    if snap.get("highest_since_entry") is not None:
        lines.append(f"High since entry: {_fmt_price(snap['highest_since_entry'])}")
    if snap.get("atr_pct") is not None:
        lines.append(f"ATR: {float(snap['atr_pct']):.1f}%")
    if snap.get("ema10") is not None:
        lines.append(f"EMA10: {_fmt_price(snap['ema10'])}")
    if snap.get("ema21") is not None:
        lines.append(f"EMA21: {_fmt_price(snap['ema21'])}")
    if snap.get("initial_stop") is not None:
        lines.append(f"Initial stop: {_fmt_price(snap['initial_stop'])}")
    if snap.get("current_protective_level") is not None:
        lines.append(f"Protective level: {_fmt_price(snap['current_protective_level'])}")
    lines += _corporate_action_lines(snap)
    if reasons:
        lines.append("Why:")
        lines += [f"• {esc(r)}" for r in reasons[:4]]
    lines.append(f"Status: {esc(state)}")
    return "\n".join(lines)


def snapshot_corporate_actions(snap):
    """The corporate actions recorded against a snapshot (live dict or JSON)."""
    value = snap.get("corporate_actions")
    if value is None:
        raw = snap.get("corporate_actions_json")
        try:
            value = json.loads(raw) if isinstance(raw, str) and raw else raw
        except (TypeError, ValueError):
            value = None
    return [str(v) for v in value] if isinstance(value, list) else []


def _corporate_action_lines(snap):
    """§12: the P&L split and the action list — only when there is something to
    say. A position that has seen no corporate action reports exactly as before.
    """
    actions = snapshot_corporate_actions(snap)
    dividends = float(snap.get("dividend_income") or 0)
    if not actions and not dividends:
        return []
    lines = []
    if dividends:
        lines.append(f"Price P&L: {_fmt_price(snap.get('price_pnl'))} · "
                     f"Dividend income: {_fmt_price(dividends)} · "
                     f"Total P&L: {_fmt_price(snap.get('total_pnl'))}")
    if actions:
        lines.append("Corporate actions since entry:")
        lines += [f"• {esc(a)}" for a in actions[-4:]]
    return lines


# ── corporate-action alerts (§11) ─────────────────────────────────────────
# Sent as their own short messages after the report. Detection and the safe,
# unambiguous adjustments are read-only for the user and need no confirmation;
# anything that would require an uncertain journal mutation says so and waits.

_ADVISORY_FOOTER = "Journal adjustment only — no brokerage action performed."


def _qty_text(value):
    return "–" if value is None else f"{float(value):g}"


def build_corporate_action_alert(action):
    """One Telegram message describing what a corporate action did (or did not do)."""
    action_type = str(action.get("action_type") or "OTHER")
    symbol = esc(action.get("symbol"))
    effective = esc(action.get("effective_date"))
    status = action.get("status")

    if action_type in ("SPLIT", "REVERSE_SPLIT", "STOCK_DIVIDEND") and status == "APPLIED":
        before, after = action.get("before") or {}, action.get("after") or {}
        title = {"SPLIT": "Stock split", "REVERSE_SPLIT": "Reverse split",
                 "STOCK_DIVIDEND": "Stock dividend"}[action_type]
        ratio = action.get("ratio") or action.get("split_ratio")
        if action_type == "STOCK_DIVIDEND" and ratio:
            headline = f"{(float(ratio) - 1) * 100:g}% stock dividend effective {effective}"
        else:
            headline = f"{esc(_ratio_text(ratio))} split effective {effective}"
        lines = [f"🔄 <b>{symbol} — {title}</b>", "", headline, "",
                 f"Shares: {_qty_text(before.get('shares'))} → {_qty_text(after.get('shares'))}",
                 f"Average cost: {_fmt_price(before.get('average_cost'))} → "
                 f"{_fmt_price(after.get('average_cost'))}"]
        if before.get("initial_stop") is not None or after.get("initial_stop") is not None:
            lines.append(f"Initial stop: {_fmt_price(before.get('initial_stop'))} → "
                         f"{_fmt_price(after.get('initial_stop'))}")
        if before.get("pivot_price") is not None or after.get("pivot_price") is not None:
            lines.append(f"Pivot: {_fmt_price(before.get('pivot_price'))} → "
                         f"{_fmt_price(after.get('pivot_price'))}")
        lines += ["", "Economic position value unchanged.", _ADVISORY_FOOTER]
        return "\n".join(lines)

    if action_type in ("CASH_DIVIDEND", "SPECIAL_DIVIDEND") and status == "APPLIED":
        special = action_type == "SPECIAL_DIVIDEND"
        lines = [f"💰 <b>{symbol} — {'Special dividend' if special else 'Cash dividend'}</b>", "",
                 f"Ex-date: {effective}",
                 f"Dividend: {_fmt_price(action.get('per_share') or action.get('cash_amount'))}/share",
                 f"Eligible shares: {_qty_text(action.get('eligible_shares'))}",
                 f"Expected dividend income: {_fmt_price(action.get('income'))}", "",
                 "Shares and average cost unchanged."]
        if special:
            lines += ["", "⚠ A special distribution rebases the price series against your recorded "
                          "stop, pivot and protective level. They were NOT changed automatically — "
                          "review them, then clear the review."]
        lines.append(_ADVISORY_FOOTER)
        return "\n".join(lines)

    if action_type == "SYMBOL_CHANGE" and status == "APPLIED":
        return "\n".join([
            f"🔤 <b>{esc(action.get('old_symbol'))} — Ticker change</b>", "",
            f"{esc(action.get('old_symbol'))} → {esc(action.get('new_symbol'))} "
            f"effective {effective}", "",
            "The position keeps its trade id, fills, entry date and P&L history; only the symbol "
            "used for future market data changed.", _ADVISORY_FOOTER])

    if action_type == "DELISTING":
        return "\n".join([
            f"⚠ <b>Corporate action review</b>", "", symbol,
            f"Market data for this position stopped as of {effective}.", "",
            "This is NOT treated as a sale. Technical exit conclusions are suppressed until you "
            "resolve it.", _ADVISORY_FOOTER])

    detail = {"MERGER": "Merger/acquisition detected.", "SPINOFF": "Spin-off detected.",
              "RIGHTS": "Rights/warrant issue detected."}.get(action_type,
                                                              f"{esc(action_type)} detected.")
    lines = [f"⚠ <b>Corporate action review</b>", "", symbol, detail, f"Effective: {effective}"]
    if action.get("cash_amount"):
        lines.append(f"Stated cash terms: {_fmt_price(action.get('cash_amount'))}/share")
    lines += ["", "Automatic cost-basis allocation was not attempted.",
              "No exchange ratio or new share quantity was invented.", "Review required."]
    if status == "AWAITING_CONFIRMATION":
        lines.append(f"Once the terms are final, record the close yourself: "
                     f"<code>/close {esc(action.get('symbol'))} "
                     f"{float(action.get('cash_amount') or 0):g}</code>")
    lines.append(_ADVISORY_FOOTER)
    return "\n".join(lines)


def _ratio_text(ratio):
    """'2-for-1' / '1-for-5' without importing the corporate-action module."""
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return "share basis change"
    if value >= 1:
        whole = round(value)
        return f"{whole}-for-1" if abs(value - whole) < 1e-9 else f"{value:g}-for-1"
    inverse = 1.0 / value
    whole = round(inverse)
    return f"1-for-{whole}" if abs(inverse - whole) < 1e-9 else f"1-for-{inverse:g}"


#: Worst-first, so anything needing attention leads the section.
_POSITION_ORDER = ("CORPORATE_ACTION_REVIEW", "STOP_TRIGGERED", "EXIT_REVIEW",
                   "PARTIAL_PROFIT_REVIEW", "TIGHTEN_PROTECTION", "WATCH", "HOLD")


def build_positions_blocks(snapshots):
    """The "💼 OPEN POSITIONS" section (empty when nothing is held)."""
    if not snapshots:
        return []
    order = {state: i for i, state in enumerate(_POSITION_ORDER)}
    ranked = sorted(snapshots, key=lambda s: (order.get(s.get("monitor_state"), 99), s.get("symbol") or ""))
    return (["💼 <b>OPEN POSITIONS</b>"] + [_position_block(s) for s in ranked]
            + ["<i>Advisory only — this app records the fills you report and monitors public market "
               "data; it never places, modifies or cancels an order.</i>"])


_BUCKET_ICONS = {READY: "🟢", WATCH: "🟡", REVIEW: "🟠", NO_SETUP: "⚪"}
_BUCKET_LABELS = {READY: "READY", WATCH: "WATCH", REVIEW: "REVIEW", NO_SETUP: "NO VCP"}


def build_chart_caption(row, plan=None, held=None):
    """Caption for an existing Stage-G VCP verification chart.

    ``plan`` is the persisted setup_plans row: a READY setup's caption carries its
    stored entry trigger / stop / size (never recomputed). ``held`` is the OPEN
    trade for the symbol, if any."""
    view = classify_setup(row, plan)
    m = row_vcp_metrics(row)
    rank = row.get("final_rank")
    if view.bucket == READY:
        entry = plan.get("entry_plan") or {}
        distance = entry.get("distance_to_pivot_pct", m.get("pct_from_pivot"))
        title = "BREAKOUT" if plan.get("setup_state") == "BREAKOUT_TRIGGERED" else "READY"
        lines = [f"🟢 <b>{esc(row['symbol'])}</b> · {title} · Rank {rank}",
                 f"VCP: {esc(str(m.get('vcp_numeric_quality') or 'n/a').upper())} · {esc(_sequence(m))}",
                 f"Price: {_fmt_price(plan.get('current_price'))} · Pivot: {_fmt_price(plan.get('pivot_price'))}"
                 + (f" · {_fmt_pct(distance)}" if distance is not None else ""),
                 f"Entry trigger: {_fmt_price(plan.get('entry_trigger_price'))}",
                 f"Initial stop: {_fmt_price(plan.get('suggested_initial_stop'))}",
                 f"Suggested size: {_fmt_price(plan.get('suggested_shares'), 0)} shares"]
        if _claude_line(row):
            lines.append(f"Claude: {_claude_line(row)}")
        if held is not None:
            lines.append(_held_line(held))
        lines.append(f"Action: {esc(view.action)}")
        return "\n".join(lines)

    lines = [f"{_BUCKET_ICONS[view.bucket]} <b>{esc(row['symbol'])}</b> · {_BUCKET_LABELS[view.bucket]} · Rank {rank}",
             f"Numeric: {esc(str(m.get('vcp_numeric_quality') or 'n/a').upper())} · {esc(_sequence(m))}"]
    second = [p for p in ((f"Volume: {esc(m['volume_dryup_quality'].upper())}" if m.get("volume_dryup_quality") else None),
                          (f"Pivot: {_pivot_text(m, with_state=False)}" if _pivot_text(m) else None)) if p]
    if second:
        lines.append(" · ".join(second))
    if _claude_line(row):
        lines.append(f"Claude: {_claude_line(row)}")
    if claude_state(row) == "reviewed" and _row_agreement(row):
        lines.append("Agreement: DISAGREE" if _disagrees(row) else "Agreement: YES")
    if view.action:
        lines.append(f"Action: {esc(view.action)}")
    return "\n".join(lines)


def build_pipeline_failed_alert(trading_date, stage, log_path, error_summary=None):
    lines = ["⚠️ <b>Stage 2 Pipeline Failed</b>", "",
             f"Trading session: {esc(trading_date)}",
             f"Stage: {esc(stage or 'unknown')}",
             "Status: FAILED"]
    if error_summary:
        lines.append(f"Error: {esc(error_summary[:300])}")
    if log_path:
        lines += ["", "See local log:", esc(log_path)]
    return "\n".join(lines)


def build_data_not_ready_alert(trading_date, latest_bar, attempts, note=None):
    lines = [
        "⚠️ <b>Market data not ready</b>", "",
        f"Target session: {esc(trading_date)}",
        f"Latest available bar: {esc(latest_bar or 'none')}",
        f"Retries: {attempts}", "",
        "Pipeline not executed.",
    ]
    if note:
        lines.append(esc(note))
    return "\n".join(lines)
