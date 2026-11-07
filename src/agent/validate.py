"""
Validate the agent's JSON response against the 8 rules from
docs/prompt_uzasadnienie.md § "Dodatkowe reguły walidacji".

validate_response(response, universe, run_date, as_of, variant)
    → (is_valid: bool, errors: list[str])

Rules 1–6 and 8: is_valid=False on failure  → caller retries (max 2).
Rule 7 (web sources in no-web variant): is_valid stays True; the error
    is appended to the list as research data (H4 halucynacja).
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


def validate_response(
    response: dict,
    universe: list[str],
    run_date: str | date,
    as_of: str | date,
    variant: str = "claude_web",
) -> tuple[bool, list[str]]:
    """
    Validate a parsed agent response dict.

    Parameters
    ----------
    response    : dict – parsed JSON from the agent.
    universe    : list[str] – the 30 expected tickers.
    run_date    : str | date – e.g. "2026-11-08".
    as_of       : str | date – e.g. "2026-11-06".
    variant     : "claude_web" or "claude_noweb".

    Returns
    -------
    (is_valid, errors)
        is_valid – False if any hard rule (1–6, 8) fails.
        errors   – human-readable descriptions of all violations.
    """
    errors:  list[str] = []
    hard_fail = False

    stocks   = response.get("stocks", [])
    picks    = response.get("picks", [])
    sources  = response.get("sources", [])
    mv       = response.get("macro_view", {})

    # ── Rule 1: 30 unique tickers matching the universe ───────────────────
    tickers = [s.get("ticker") for s in stocks]
    if len(tickers) != 30 or len(set(tickers)) != 30:
        errors.append(
            f"Rule 1: expected 30 unique tickers, got {len(tickers)} "
            f"({len(set(tickers))} unique)"
        )
        hard_fail = True
    else:
        extra   = set(tickers) - set(universe)
        missing = set(universe) - set(tickers)
        if extra or missing:
            errors.append(
                f"Rule 1: tickers don't match universe – "
                f"extra={sorted(extra)}, missing={sorted(missing)}"
            )
            hard_fail = True

    # ── Rule 2: ranks are a permutation of 1–30 ───────────────────────────
    ranks = [s.get("rank") for s in stocks if isinstance(s.get("rank"), int)]
    if set(ranks) != set(range(1, 31)):
        errors.append(
            f"Rule 2: ranks must be a permutation of 1–30, got {sorted(ranks)}"
        )
        hard_fail = True

    # ── Rule 3: scores are integers in −5…+5 ─────────────────────────────
    bad_scores = [
        (s.get("ticker"), s.get("score"))
        for s in stocks
        if not isinstance(s.get("score"), int)
        or not (-5 <= s["score"] <= 5)
    ]
    if bad_scores:
        errors.append(
            f"Rule 3: invalid scores (must be int −5…+5): {bad_scores}"
        )
        hard_fail = True

    # ── Rule 4: consistency – higher score → lower (better) rank ─────────
    consistency_violations = []
    if len(stocks) == 30 and not bad_scores and set(ranks) == set(range(1, 31)):
        for i, a in enumerate(stocks):
            for b in stocks[i + 1:]:
                if a["score"] > b["score"] and a["rank"] > b["rank"]:
                    consistency_violations.append(
                        f"{a['ticker']}(score={a['score']},rank={a['rank']}) > "
                        f"{b['ticker']}(score={b['score']},rank={b['rank']})"
                    )
                elif a["score"] < b["score"] and a["rank"] < b["rank"]:
                    consistency_violations.append(
                        f"{b['ticker']}(score={b['score']},rank={b['rank']}) > "
                        f"{a['ticker']}(score={a['score']},rank={a['rank']})"
                    )
        if consistency_violations:
            errors.append(
                f"Rule 4: score/rank inconsistencies ({len(consistency_violations)}): "
                + "; ".join(consistency_violations[:5])
                + (" …" if len(consistency_violations) > 5 else "")
            )
            hard_fail = True

    # ── Rule 5: picks = tickers with ranks 1,2,3,4 in that order ─────────
    rank_map = {s["rank"]: s["ticker"] for s in stocks if isinstance(s.get("rank"), int)}
    expected_picks = [rank_map.get(r) for r in (1, 2, 3, 4)]
    if picks != expected_picks:
        errors.append(
            f"Rule 5: picks {picks} != expected {expected_picks} "
            f"(tickers at ranks 1–4)"
        )
        hard_fail = True

    # ── Rule 6: every source_id referenced exists in sources ─────────────
    declared_ids = {src["id"] for src in sources if isinstance(src.get("id"), str)}
    used_ids: set[str] = set()
    for s in stocks:
        used_ids.update(s.get("source_ids", []))
    used_ids.update(mv.get("source_ids", []))

    undefined = used_ids - declared_ids
    if undefined:
        errors.append(
            f"Rule 6: source_ids referenced but not declared: {sorted(undefined)}"
        )
        hard_fail = True

    # ── Rule 7: no-web variant must not contain web-type sources ─────────
    if variant == "claude_noweb":
        web_sources = [s["id"] for s in sources if s.get("type") == "web"]
        if web_sources:
            msg = (
                f"Rule 7 [content error, not retried]: "
                f"claude_noweb response contains web sources: {web_sources}"
            )
            errors.append(msg)
            logger.warning(msg)
            # is_valid stays True – this is research data for H4

    # ── Rule 7b: in claude_web, flag sources published after as_of ───────
    # Non-blocking. Logged for inspection; does not trigger a retry.
    if variant == "claude_web":
        future = []
        for src in sources:
            pd_ = src.get("published_date")
            if pd_ and str(pd_) > str(as_of):
                future.append(f"{src.get('id')} ({pd_})")
        if future:
            msg = (
                f"Rule 7b [content error, not retried]: "
                f"claude_web sources published after as_of {as_of}: {future}"
            )
            errors.append(msg)
            logger.warning(msg)
            # is_valid stays True

    # ── Rule 8: run_date and as_of match the run parameters ───────────────
    resp_run_date = response.get("run_date", "")
    resp_as_of    = response.get("as_of", "")
    if str(resp_run_date) != str(run_date):
        errors.append(
            f"Rule 8: run_date mismatch – response '{resp_run_date}' "
            f"!= expected '{run_date}'"
        )
        hard_fail = True
    if str(resp_as_of) != str(as_of):
        errors.append(
            f"Rule 8: as_of mismatch – response '{resp_as_of}' "
            f"!= expected '{as_of}'"
        )
        hard_fail = True

    is_valid = not hard_fail
    if errors:
        for e in errors:
            logger.info("Validation: %s", e)
    return is_valid, errors
