"""
Unit tests for src/agent/validate.py.
One broken-answer fixture per rule (1–8).
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from agent.validate import validate_response

UNIVERSE = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "WMT",
]
RUN_DATE = "2026-11-08"
AS_OF    = "2026-11-06"


def _make_valid() -> dict:
    """Minimal valid response (all rules satisfied)."""
    # Scores: first 4 get distinct positive scores for unambiguous picks;
    # rest get 0. Ranks mirror the order.
    stocks = []
    scores = [4, 3, 2, 1] + [0] * 26
    for i, tkr in enumerate(UNIVERSE):
        stocks.append({
            "ticker":    tkr,
            "score":     scores[i],
            "rank":      i + 1,
            "rationale": "placeholder",
            "key_factors": ["price_momentum"],
            "source_ids": ["PKG"],
        })
    return {
        "run_date": RUN_DATE,
        "as_of":    AS_OF,
        "macro_view": {
            "summary":            "ok",
            "main_themes":        ["growth"],
            "favoured_sectors":   [],
            "disfavoured_sectors": [],
            "source_ids":         ["PKG"],
        },
        "stocks":           stocks,
        "picks":            UNIVERSE[:4],
        "picks_rationale":  "top four",
        "sources":          [{"id": "PKG", "type": "data_package", "title": "Data package"}],
        "data_issues":      [],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid():
    ok, errs = validate_response(_make_valid(), UNIVERSE, RUN_DATE, AS_OF)
    assert ok,  f"Baseline should be valid; errors: {errs}"
    assert errs == []


def _invalid(resp, expected_rule_prefix):
    ok, errs = validate_response(resp, UNIVERSE, RUN_DATE, AS_OF)
    assert not ok, "Expected invalid response to be rejected"
    assert any(expected_rule_prefix in e for e in errs), (
        f"Expected error containing '{expected_rule_prefix}', got: {errs}"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_baseline():
    _valid()


def test_rule1_wrong_ticker():
    """Rule 1: a ticker not in the universe."""
    r = _make_valid()
    r["stocks"][0]["ticker"] = "FAKE"
    _invalid(r, "Rule 1")


def test_rule1_duplicate_ticker():
    """Rule 1: duplicate ticker (only 29 unique)."""
    r = _make_valid()
    r["stocks"][1]["ticker"] = r["stocks"][0]["ticker"]
    _invalid(r, "Rule 1")


def test_rule2_ranks_not_permutation():
    """Rule 2: two stocks share the same rank."""
    r = _make_valid()
    r["stocks"][1]["rank"] = r["stocks"][0]["rank"]   # duplicate rank 1
    _invalid(r, "Rule 2")


def test_rule3_score_out_of_range():
    """Rule 3: score = 6 (above +5)."""
    r = _make_valid()
    r["stocks"][0]["score"] = 6
    _invalid(r, "Rule 3")


def test_rule3_score_not_integer():
    """Rule 3: score = 2.5 (float)."""
    r = _make_valid()
    r["stocks"][0]["score"] = 2.5
    _invalid(r, "Rule 3")


def test_rule4_score_rank_inconsistency():
    """Rule 4: stock A has higher score but worse rank than stock B."""
    r = _make_valid()
    # Swap scores so rank 1 (AAPL, score 4) now has a lower score than rank 2 (AMGN, score 3)
    # but keep ranks unchanged → score 1 < score 2 yet rank 1 < rank 2 → violates rule 4
    r["stocks"][0]["score"] = 1   # AAPL: rank 1, score 1
    r["stocks"][1]["score"] = 3   # AMGN: rank 2, score 3  → score > AAPL but rank worse
    _invalid(r, "Rule 4")


def test_rule5_picks_wrong_order():
    """Rule 5: picks listed in wrong order (rank-2 first)."""
    r = _make_valid()
    r["picks"] = [UNIVERSE[1], UNIVERSE[0], UNIVERSE[2], UNIVERSE[3]]
    _invalid(r, "Rule 5")


def test_rule5_picks_wrong_tickers():
    """Rule 5: picks contain a ticker not ranked 1–4."""
    r = _make_valid()
    r["picks"] = [UNIVERSE[0], UNIVERSE[1], UNIVERSE[2], UNIVERSE[4]]  # UNIVERSE[4]=BA rank 5
    _invalid(r, "Rule 5")


def test_rule6_undefined_source_id():
    """Rule 6: a stock references source 'W1' which is not in sources."""
    r = _make_valid()
    r["stocks"][0]["source_ids"] = ["PKG", "W1"]
    _invalid(r, "Rule 6")


def test_rule7_web_source_in_noweb_variant():
    """Rule 7: web source in claude_noweb – content error, no retry, answer kept."""
    r = _make_valid()
    r["sources"].append({"id": "W1", "type": "web", "title": "Some article"})
    r["stocks"][0]["source_ids"] = ["PKG", "W1"]
    ok, errs = validate_response(r, UNIVERSE, RUN_DATE, AS_OF, variant="claude_noweb")
    assert ok, "Rule 7 must NOT reject the response (no retry)"
    rule7 = [e for e in errs if "Rule 7" in e]
    assert len(rule7) == 1, f"Expected exactly one Rule 7 content error, got: {errs}"
    assert "content error, not retried" in rule7[0]
    assert "W1" in rule7[0]


def test_rule7_web_source_ok_in_web_variant():
    """Rule 7: web sources produce NO error at all in claude_web."""
    r = _make_valid()
    r["sources"].append({"id": "W1", "type": "web", "title": "Article"})
    r["stocks"][0]["source_ids"] = ["PKG", "W1"]
    ok, errs = validate_response(r, UNIVERSE, RUN_DATE, AS_OF, variant="claude_web")
    assert ok
    assert errs == [], f"claude_web with web sources must yield no errors, got: {errs}"


def test_rule8_run_date_mismatch():
    """Rule 8: run_date in response doesn't match expected."""
    r = _make_valid()
    r["run_date"] = "2026-11-01"
    _invalid(r, "Rule 8")


def test_rule8_as_of_mismatch():
    """Rule 8: as_of in response doesn't match expected."""
    r = _make_valid()
    r["as_of"] = "2026-10-30"
    _invalid(r, "Rule 8")
