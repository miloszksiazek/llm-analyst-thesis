"""
Unit tests for aggregate_runs() and spearman_rank() in src/agent/run.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from agent.run import aggregate_runs, spearman_rank


def _mk_parsed(ranking: list[tuple[str, int]]) -> dict:
    """Build a minimal parsed-response dict from [(ticker, score), ...].

    Ranks are assigned by sort-by-score-descending with alphabetical tiebreak,
    which matches how an actual agent would be scored → ranked consistently.
    """
    ordered = sorted(ranking, key=lambda x: (-x[1], x[0]))
    stocks = [
        {"ticker": t, "score": s, "rank": i + 1}
        for i, (t, s) in enumerate(ordered)
    ]
    return {"stocks": stocks, "picks": [s["ticker"] for s in stocks[:4]]}


# ── spearman_rank ────────────────────────────────────────────────────────────

def test_spearman_identical_is_one():
    r = list(range(1, 31))
    assert spearman_rank(r, r) == pytest.approx(1.0)


def test_spearman_reverse_is_minus_one():
    r  = list(range(1, 31))
    rr = list(reversed(r))
    assert spearman_rank(r, rr) == pytest.approx(-1.0)


def test_spearman_known_value():
    # Textbook example: single swap of adjacent ranks in n=5 → ρ=0.9
    a = [1, 2, 3, 4, 5]
    b = [2, 1, 3, 4, 5]
    assert spearman_rank(a, b) == pytest.approx(0.9)


def test_spearman_requires_matching_length():
    with pytest.raises(ValueError):
        spearman_rank([1, 2], [1, 2, 3])


# ── aggregate_runs: basic path ───────────────────────────────────────────────

def test_aggregate_two_identical_runs():
    tickers = [f"T{i:02d}" for i in range(30)]
    scores  = list(range(29, -1, -1))   # T00 highest, T29 lowest
    r = _mk_parsed(list(zip(tickers, scores)))
    agg = aggregate_runs([r, r], ["r1", "r2"])

    assert agg["picks"] == ["T00", "T01", "T02", "T03"]
    assert agg["stocks"][0]["ticker"] == "T00"
    assert agg["stocks"][0]["final_rank"] == 1
    assert agg["stocks"][0]["mean_rank"] == 1.0
    assert agg["stocks"][0]["min_rank"] == 1
    assert agg["stocks"][0]["max_rank"] == 1

    assert len(agg["pairwise_spearman"]) == 1
    assert agg["pairwise_spearman"][0]["rho"] == pytest.approx(1.0)


def test_aggregate_rejects_single_run():
    r = _mk_parsed([("T00", 5)] + [(f"T{i:02d}", 0) for i in range(1, 30)])
    with pytest.raises(ValueError):
        aggregate_runs([r])


def test_aggregate_three_diverse_runs_picks_by_mean_rank():
    """
    Build 3 runs where ticker A is consistently top (ranks 1,1,2)
    and ticker B is consistently second-top (ranks 2,2,1).
    A's mean rank (1.33) < B's (1.67) → A finishes first.
    Also check that the final top-4 matches tickers with lowest mean rank.
    """
    tickers = [f"T{i:02d}" for i in range(30)]

    def permutation_with_swap(swap_i: int, swap_j: int) -> list[str]:
        order = tickers[:]
        order[swap_i], order[swap_j] = order[swap_j], order[swap_i]
        return order

    # Scores descending; rank derived from score
    runs = []
    for order in (tickers, permutation_with_swap(0, 1), permutation_with_swap(2, 3)):
        scored = [(t, 29 - idx) for idx, t in enumerate(order)]
        runs.append(_mk_parsed(scored))

    agg = aggregate_runs(runs)
    picks = agg["picks"]
    assert picks[0] == "T00" and picks[1] == "T01"
    assert set(picks[2:]) == {"T02", "T03"}


# ── aggregate_runs: tie-breakers ─────────────────────────────────────────────

def test_tiebreak_higher_mean_score_wins():
    """Two tickers with identical mean rank → the one with higher mean score wins."""
    # Run A: X rank 1 (score 5), Y rank 2 (score 4). Rest in order.
    # Run B: X rank 2 (score 3), Y rank 1 (score 5). Rest shifted.
    # Mean rank: X=1.5, Y=1.5. Mean score: X=4, Y=4.5 → Y should finish first.
    others = [(f"T{i:02d}", 0) for i in range(28)]

    # We'll construct the parsed dicts by hand because we need specific score pairings.
    def parsed(x_rank, x_score, y_rank, y_score):
        stocks = [{"ticker": "X", "rank": x_rank, "score": x_score},
                  {"ticker": "Y", "rank": y_rank, "score": y_score}]
        used = {x_rank, y_rank}
        avail = iter(r for r in range(1, 31) if r not in used)
        for t, _ in others:
            stocks.append({"ticker": t, "rank": next(avail), "score": 0})
        return {"stocks": stocks}

    r1 = parsed(1, 5, 2, 4)
    r2 = parsed(2, 3, 1, 5)
    agg = aggregate_runs([r1, r2])
    # Both have mean_rank 1.5; Y's mean_score 4.5 beats X's 4.0
    assert agg["stocks"][0]["ticker"] == "Y"
    assert agg["stocks"][1]["ticker"] == "X"


def test_tiebreak_alphabetical_when_mean_rank_and_score_equal():
    """Equal mean rank AND equal mean score → alphabetical."""
    def parsed(a_rank, b_rank, score_a, score_b):
        stocks = [{"ticker": "AAPL", "rank": a_rank, "score": score_a},
                  {"ticker": "ZOOM", "rank": b_rank, "score": score_b}]
        used = {a_rank, b_rank}
        avail = iter(r for r in range(1, 31) if r not in used)
        for i in range(28):
            stocks.append({"ticker": f"T{i:02d}", "rank": next(avail), "score": 0})
        return {"stocks": stocks}

    # mean rank = 1.5 both; mean score = 4 both
    r1 = parsed(1, 2, 4, 4)
    r2 = parsed(2, 1, 4, 4)
    agg = aggregate_runs([r1, r2])
    assert agg["stocks"][0]["ticker"] == "AAPL"
    assert agg["stocks"][1]["ticker"] == "ZOOM"


def test_aggregate_min_max_rank_tracked():
    """min_rank and max_rank must reflect the full range across runs."""
    others_a = [(f"T{i:02d}", 10) for i in range(28)]
    r1 = _mk_parsed([("TOP", 100), ("BOT", -100)] + others_a)
    r2 = _mk_parsed([("TOP", -100), ("BOT", 100)] + others_a)
    agg = aggregate_runs([r1, r2])
    top = next(s for s in agg["stocks"] if s["ticker"] == "TOP")
    bot = next(s for s in agg["stocks"] if s["ticker"] == "BOT")
    # TOP: rank 1 in r1, rank 30 in r2 → min=1, max=30
    assert top["min_rank"] == 1 and top["max_rank"] == 30
    assert bot["min_rank"] == 1 and bot["max_rank"] == 30


def test_pairwise_matrix_shape():
    """n_runs=3 → 3 pairs; n_runs=4 → 6 pairs."""
    tickers = [f"T{i:02d}" for i in range(30)]
    scores  = list(range(29, -1, -1))
    r = _mk_parsed(list(zip(tickers, scores)))
    agg3 = aggregate_runs([r, r, r])
    agg4 = aggregate_runs([r, r, r, r])
    assert len(agg3["pairwise_spearman"]) == 3
    assert len(agg4["pairwise_spearman"]) == 6
    for p in agg3["pairwise_spearman"]:
        assert p["rho"] == pytest.approx(1.0)
