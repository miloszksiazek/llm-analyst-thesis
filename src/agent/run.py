"""
Weekly agent run – one variant at a time.

run_variant(variant, package_path, run_date, out_dir, n_runs=None) builds
the prompt once, then executes `n_runs` independent calls to the model
(same input, no shared state) and writes every artifact to disk.

Layout under out_dir:

    n_runs == 1 (default)                   n_runs > 1
    -----------------------------           ----------------------------------
    <variant>/response.json                 <variant>/run_1/response.json
    <variant>/report.md                     <variant>/run_1/...
    <variant>/validation.json               <variant>/run_2/...
    <variant>/run.json                      <variant>/aggregate.json
    <variant>/searches.json                 <variant>/report.md    ← aggregate
    <variant>/thinking_summary.txt
    <variant>/attempt_01_*

Aggregation (n_runs > 1):
    - final rank = order by (mean_rank ↑, mean_score ↓, ticker)
    - final picks = tickers ranked 1..4 in that order
    - pairwise Spearman between every pair of valid runs
    - if fewer than 2 runs validated → fallback_applied=True, picks=[].
      Downstream portfolio code reads this and applies keep_prev_portfolio.

API parameters verified against anthropic==2.3.0 types:
    - thinking      = {"type": "adaptive", "display": "summarized"}
    - output_config = {"effort": <settings.effort>}
    - tools         = [web_search_20260318] for claude_web, else absent
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv

import sys
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from agent.prompt import render_prompt
from agent.validate import validate_response

logger = logging.getLogger(__name__)

_SETTINGS_FILE = _ROOT / "config" / "settings.yaml"
_UNIVERSE_FILE = _ROOT / "config" / "universe.yaml"

VARIANTS = ("claude_web", "claude_noweb")


# ── Settings & universe ───────────────────────────────────────────────────────

def _load_settings() -> dict[str, Any]:
    with open(_SETTINGS_FILE) as fh:
        return yaml.safe_load(fh)


def _load_universe_tickers() -> list[str]:
    with open(_UNIVERSE_FILE) as fh:
        u = yaml.safe_load(fh)
    return [s["ticker"] for s in u["universe"]]


# ── Cost model ────────────────────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    input_tokens:  int = 0
    output_tokens: int = 0
    web_searches:  int = 0
    input_usd:     float = 0.0
    output_usd:    float = 0.0
    search_usd:    float = 0.0
    total_usd:     float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _cost(usage: Any, pricing: dict[str, float]) -> CostBreakdown:
    inp  = int(getattr(usage, "input_tokens", 0) or 0)
    out  = int(getattr(usage, "output_tokens", 0) or 0)
    stu  = getattr(usage, "server_tool_use", None)
    ws   = int(getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    ic   = inp / 1_000_000 * pricing["input_per_mtok"]
    oc   = out / 1_000_000 * pricing["output_per_mtok"]
    sc   = ws  / 1_000     * pricing["search_per_1k"]
    return CostBreakdown(inp, out, ws, ic, oc, sc, ic + oc + sc)


# ── API response extraction ───────────────────────────────────────────────────

import re
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_final_text(response: Any) -> str:
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _collect_thinking(response: Any) -> list[str]:
    return [
        getattr(b, "thinking", "") or ""
        for b in response.content
        if getattr(b, "type", None) == "thinking"
    ]


def _collect_searches(response: Any) -> tuple[list[dict], list[dict]]:
    queries, sources = [], []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
            inp = getattr(block, "input", {}) or {}
            queries.append({"tool_use_id": getattr(block, "id", None), "query": inp.get("query")})
        elif btype == "web_search_tool_result":
            for item in getattr(block, "content", None) or []:
                if getattr(item, "type", None) == "web_search_result":
                    sources.append({
                        "url":      getattr(item, "url", None),
                        "title":    getattr(item, "title", None),
                        "page_age": getattr(item, "page_age", None),
                    })
    return queries, sources


# ── Core: one API call ────────────────────────────────────────────────────────

@dataclass
class Attempt:
    attempt:        int
    started_at:     str
    finished_at:    str
    stop_reason:    str | None
    raw_text:       str
    parsed:         dict | None
    validation_ok:  bool
    validation_errors: list[str]
    usage:          dict
    cost:           dict
    searches:       list[dict] = field(default_factory=list)
    retrieved:      list[dict] = field(default_factory=list)
    thinking:       list[str]  = field(default_factory=list)
    model_id:       str | None = None
    error:          str | None = None


def _build_request(variant: str, system: str, user: str, settings: dict) -> dict:
    req: dict[str, Any] = {
        "model":       settings["model"],
        "max_tokens":  settings["max_tokens"],
        "system":      system,
        "messages":    [{"role": "user", "content": user}],
        "thinking":    {"type": "adaptive", "display": settings["thinking_display"]},
        "output_config": {"effort": settings["effort"]},
    }
    if variant == "claude_web":
        req["tools"] = [{
            "type":     "web_search_20260318",
            "name":     "web_search",
            "max_uses": settings["max_searches"],
        }]
    return req


def _redact_request(req: dict) -> dict:
    return json.loads(json.dumps(req, default=str))


def _one_attempt(client, req, attempt_idx, universe, run_date, as_of,
                 variant, pricing, cost_guard) -> Attempt:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        response = client.messages.create(**req)
    except Exception as exc:
        finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.exception("API call failed on attempt %d", attempt_idx)
        return Attempt(
            attempt=attempt_idx, started_at=started, finished_at=finished,
            stop_reason=None, raw_text="", parsed=None,
            validation_ok=False, validation_errors=[f"API error: {exc}"],
            usage={}, cost={}, error=str(exc),
        )

    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cost = _cost(response.usage, pricing)
    if cost.total_usd > cost_guard:
        raise RuntimeError(
            f"Cost guard tripped: attempt {attempt_idx} cost "
            f"${cost.total_usd:.4f} > guard ${cost_guard:.2f}"
        )

    raw_text  = _extract_final_text(response)
    queries, retrieved = _collect_searches(response)
    thinking  = _collect_thinking(response)

    parsed:  dict | None = None
    errors:  list[str]   = []
    ok = False
    try:
        parsed = _parse_json(raw_text)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")

    if parsed is not None:
        ok, errors = validate_response(parsed, universe, run_date, as_of, variant)

    return Attempt(
        attempt=attempt_idx, started_at=started, finished_at=finished,
        stop_reason=response.stop_reason, raw_text=raw_text, parsed=parsed,
        validation_ok=ok, validation_errors=errors,
        usage=response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage),
        cost=cost.as_dict(),
        searches=queries, retrieved=retrieved, thinking=thinking,
        model_id=response.model,
    )


# ── One complete run of a variant (writes all artifacts to run_dir) ───────────

def _run_once(
    variant:      str,
    run_dir:      Path,
    req:          dict,
    settings:     dict,
    client:       anthropic.Anthropic,
    universe:     list[str],
    run_date_str: str,
    as_of:        str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "request.json").write_text(
        json.dumps(_redact_request(req), indent=2, default=str), encoding="utf-8",
    )

    pricing    = settings["pricing"]
    cost_guard = float(settings["cost_guard_usd"])
    max_total  = 1 + int(settings["max_retries"])

    attempts: list[Attempt] = []
    final: Attempt | None = None
    for i in range(1, max_total + 1):
        logger.info("[%s] attempt %d/%d", variant, i, max_total)
        a = _one_attempt(
            client, req, i, universe, run_date_str, as_of,
            variant, pricing, cost_guard,
        )
        attempts.append(a)
        if a.validation_ok:
            final = a
            break
        logger.warning("[%s] attempt %d failed: %s", variant, i, a.validation_errors)

    for a in attempts:
        (run_dir / f"attempt_{a.attempt:02d}_raw.txt").write_text(a.raw_text or "", encoding="utf-8")
        if a.parsed is not None:
            (run_dir / f"attempt_{a.attempt:02d}_parsed.json").write_text(
                json.dumps(a.parsed, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    chosen = final if final is not None else attempts[-1]

    (run_dir / "validation.json").write_text(
        json.dumps({
            "variant":          variant,
            "is_valid":         chosen.validation_ok,
            "errors":           chosen.validation_errors,
            "attempt_used":     chosen.attempt,
            "attempts_total":   len(attempts),
        }, indent=2), encoding="utf-8",
    )

    if chosen.parsed is not None:
        (run_dir / "response.json").write_text(
            json.dumps(chosen.parsed, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (run_dir / "report.md").write_text(
            _render_report(chosen.parsed, variant, run_date_str, as_of), encoding="utf-8",
        )

    (run_dir / "searches.json").write_text(
        json.dumps({"queries": chosen.searches, "retrieved": chosen.retrieved},
                   indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (run_dir / "thinking_summary.txt").write_text(
        "\n\n---\n\n".join(chosen.thinking), encoding="utf-8",
    )

    total_cost = sum(a.cost.get("total_usd", 0.0) for a in attempts)
    (run_dir / "run.json").write_text(
        json.dumps({
            "variant":          variant,
            "run_date":         run_date_str,
            "as_of":            as_of,
            "model_id":         chosen.model_id,
            "attempts": [
                {"attempt": a.attempt, "started_at": a.started_at,
                 "finished_at": a.finished_at, "stop_reason": a.stop_reason,
                 "usage": a.usage, "cost": a.cost,
                 "validation_ok": a.validation_ok, "errors": a.validation_errors,
                 "error": a.error}
                for a in attempts
            ],
            "final_attempt":    chosen.attempt,
            "is_valid":         chosen.validation_ok,
            "web_search_count": chosen.cost.get("web_searches", 0),
            "total_cost_usd":   total_cost,
        }, indent=2, default=str), encoding="utf-8",
    )

    return {
        "is_valid":       chosen.validation_ok,
        "errors":         chosen.validation_errors,
        "attempts":       len(attempts),
        "parsed":         chosen.parsed,
        "web_searches":   chosen.cost.get("web_searches", 0),
        "input_tokens":   chosen.cost.get("input_tokens", 0),
        "output_tokens":  chosen.cost.get("output_tokens", 0),
        "cost_usd":       chosen.cost.get("total_usd", 0.0),
        "total_cost_usd": total_cost,
        "model_id":       chosen.model_id,
    }


# ── Report rendering ──────────────────────────────────────────────────────────

def _render_report(parsed: dict, variant: str, run_date: str, as_of: str) -> str:
    stocks = sorted(parsed["stocks"], key=lambda s: s["rank"])
    mv = parsed["macro_view"]
    lines = [
        f"# Weekly assessment — {variant}",
        "",
        f"- **Run date:** {run_date}",
        f"- **As of:** {as_of}",
        f"- **Picks (rank 1–4):** {', '.join(parsed['picks'])}",
        "",
        "## Macro view",
        "",
        mv["summary"],
        "",
        f"- Themes: {', '.join(mv['main_themes'])}",
        f"- Favoured sectors: {', '.join(mv['favoured_sectors']) or '—'}",
        f"- Disfavoured sectors: {', '.join(mv['disfavoured_sectors']) or '—'}",
        "",
        "## Picks rationale",
        "",
        parsed["picks_rationale"],
        "",
        "## Ranking",
        "",
        "| Rank | Ticker | Score | Key factors | Rationale |",
        "|-----:|:-------|------:|:-----------|:----------|",
    ]
    for s in stocks:
        kf  = ", ".join(s["key_factors"])
        rat = s["rationale"].replace("|", "\\|")
        lines.append(f"| {s['rank']} | {s['ticker']} | {s['score']:+d} | {kf} | {rat} |")

    lines += ["", "## Sources", ""]
    for src in parsed["sources"]:
        extra = [x for x in (src.get("publisher"), src.get("published_date"), src.get("url")) if x]
        tail  = f" — {' · '.join(extra)}" if extra else ""
        lines.append(f"- **{src['id']}** ({src['type']}) {src['title']}{tail}")

    if parsed.get("data_issues"):
        lines += ["", "## Data issues", ""]
        for di in parsed["data_issues"]:
            lines.append(f"- {di}")

    return "\n".join(lines) + "\n"


def _render_aggregate_report(agg: dict, variant: str, run_date: str, as_of: str) -> str:
    lines = [
        f"# Aggregate assessment — {variant}",
        "",
        f"- **Run date:** {run_date}",
        f"- **As of:** {as_of}",
        f"- **Valid runs:** {agg['n_valid_runs']} / {agg['n_runs_requested']}",
    ]
    if agg["fallback_applied"]:
        lines += [
            f"- **Fallback applied:** {agg['fallback_rule']}",
            "",
            "Fewer than 2 runs were valid; no aggregate ranking produced.",
        ]
        return "\n".join(lines) + "\n"

    lines += [
        f"- **Picks (final):** {', '.join(agg['picks'])}",
        "",
        "## Pairwise Spearman ρ between runs",
        "",
        "| Pair | ρ |",
        "|------|---:|",
    ]
    for p in agg["pairwise_spearman"]:
        lines.append(f"| {p['a']} ↔ {p['b']} | {p['rho']:+.3f} |")

    lines += [
        "",
        "## Aggregate ranking",
        "",
        "| Final rank | Ticker | Mean rank | Mean score | Min rank | Max rank |",
        "|-----------:|:-------|----------:|-----------:|---------:|---------:|",
    ]
    for s in agg["stocks"]:
        lines.append(
            f"| {s['final_rank']} | {s['ticker']} | {s['mean_rank']:.2f} | "
            f"{s['mean_score']:+.2f} | {s['min_rank']} | {s['max_rank']} |"
        )
    return "\n".join(lines) + "\n"


# ── Spearman (pure Python, ranks only — no ties by validate.py Rule 2) ────────

def spearman_rank(ra: list[int], rb: list[int]) -> float:
    """Spearman ρ on integer rank vectors of equal length with no ties (= Pearson)."""
    n = len(ra)
    if n != len(rb) or n < 2:
        raise ValueError("rank vectors must be equal length ≥ 2")
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((a-ma)*(b-mb) for a, b in zip(ra, rb))
    da  = sum((a-ma)**2 for a in ra) ** 0.5
    db  = sum((b-mb)**2 for b in rb) ** 0.5
    if da == 0 or db == 0:
        raise ValueError("zero-variance rank vector")
    return num / (da * db)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_runs(parsed_runs: list[dict], run_ids: list[str] | None = None) -> dict:
    """
    Combine >=2 valid per-run responses into a single aggregate ranking.

    Returns a dict with: picks, stocks (per-ticker mean_rank, mean_score,
    min_rank, max_rank, final_rank) and pairwise_spearman.

    Final ranking order:
        1. mean_rank       ascending (lower is better)
        2. mean_score      descending (higher beats lower)
        3. ticker          alphabetical
    """
    if len(parsed_runs) < 2:
        raise ValueError("aggregate_runs needs at least 2 runs")

    if run_ids is None:
        run_ids = [f"run_{i+1}" for i in range(len(parsed_runs))]
    elif len(run_ids) != len(parsed_runs):
        raise ValueError("run_ids length must match parsed_runs")

    tickers = sorted({s["ticker"] for s in parsed_runs[0]["stocks"]})

    ranks:  dict[str, list[int]] = {t: [] for t in tickers}
    scores: dict[str, list[int]] = {t: [] for t in tickers}
    for r in parsed_runs:
        for s in r["stocks"]:
            t = s["ticker"]
            if t in ranks:
                ranks[t].append(s["rank"])
                scores[t].append(s["score"])

    stocks = []
    for t in tickers:
        rs = ranks[t]
        ss = scores[t]
        stocks.append({
            "ticker":     t,
            "mean_rank":  round(sum(rs)/len(rs), 4),
            "mean_score": round(sum(ss)/len(ss), 4),
            "min_rank":   min(rs),
            "max_rank":   max(rs),
        })

    # Final order: mean_rank ↑, mean_score ↓, ticker ↑
    stocks.sort(key=lambda x: (x["mean_rank"], -x["mean_score"], x["ticker"]))
    for i, s in enumerate(stocks, 1):
        s["final_rank"] = i

    picks = [s["ticker"] for s in stocks[:4]]

    # Pairwise Spearman
    pairwise = []
    for i, ida in enumerate(run_ids):
        for j in range(i + 1, len(run_ids)):
            idb = run_ids[j]
            ra  = [next(s for s in parsed_runs[i]["stocks"] if s["ticker"] == t)["rank"] for t in tickers]
            rb  = [next(s for s in parsed_runs[j]["stocks"] if s["ticker"] == t)["rank"] for t in tickers]
            pairwise.append({"a": ida, "b": idb, "rho": round(spearman_rank(ra, rb), 4)})

    return {
        "picks":             picks,
        "stocks":            stocks,
        "pairwise_spearman": pairwise,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_variant(
    variant:      str,
    package_path: Path | str,
    run_date:     date | str,
    out_dir:      Path | str,
    n_runs:       int | None = None,
) -> dict[str, Any]:
    """
    Run `variant` n_runs times on `package_path`. n_runs defaults to
    settings["n_runs"] (= 1 unless overridden).

    Layout:
        n_runs == 1: artifacts land directly in out_dir/<variant>/.
        n_runs  > 1: each independent run is in out_dir/<variant>/run_i/;
                     the aggregate is written to out_dir/<variant>/aggregate.json
                     and report.md.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    package_path = Path(package_path)
    run_date_str = str(run_date)
    out_dir      = Path(out_dir)
    variant_dir  = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    settings = _load_settings()
    universe = _load_universe_tickers()
    if n_runs is None:
        n_runs = int(settings.get("n_runs", 1))
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    pkg_json   = json.loads(package_path.read_text(encoding="utf-8"))
    as_of      = pkg_json["meta"]["as_of"]
    sha_file   = package_path.parent / "package.sha256"
    pkg_sha256 = sha_file.read_text(encoding="utf-8").strip()

    system, user = render_prompt(package_path, run_date_str, as_of, settings["max_searches"])
    req = _build_request(variant, system, user, settings)

    load_dotenv(_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)

    # ── Single run path – keep existing layout and return shape ───────────
    if n_runs == 1:
        r = _run_once(variant, variant_dir, req, settings, client,
                      universe, run_date_str, as_of)
        return {
            "variant":        variant,
            "n_runs":         1,
            "package_sha256": pkg_sha256,
            **r,
        }

    # ── Multi-run path ────────────────────────────────────────────────────
    per_run = []
    valid_parsed:  list[dict] = []
    valid_run_ids: list[str]  = []
    total_cost = 0.0
    for i in range(1, n_runs + 1):
        run_id  = f"run_{i}"
        run_dir = variant_dir / run_id
        r = _run_once(variant, run_dir, req, settings, client,
                      universe, run_date_str, as_of)
        total_cost += r["total_cost_usd"]
        per_run.append({
            "run_id":   run_id,
            "is_valid": r["is_valid"],
            "picks":    r["parsed"]["picks"] if r["parsed"] else None,
            "errors":   r["errors"],
            "cost_usd": r["total_cost_usd"],
            "input_tokens":  r["input_tokens"],
            "output_tokens": r["output_tokens"],
        })
        if r["is_valid"] and r["parsed"] is not None:
            valid_parsed.append(r["parsed"])
            valid_run_ids.append(run_id)

    n_valid = len(valid_parsed)

    if n_valid >= 2:
        agg = aggregate_runs(valid_parsed, valid_run_ids)
        aggregate = {
            "variant":          variant,
            "run_date":         run_date_str,
            "as_of":            as_of,
            "package_sha256":   pkg_sha256,
            "n_runs_requested": n_runs,
            "n_valid_runs":     n_valid,
            "fallback_applied": False,
            "fallback_rule":    settings["fallback_rule"],
            "picks":            agg["picks"],
            "stocks":           agg["stocks"],
            "pairwise_spearman": agg["pairwise_spearman"],
            "runs":             per_run,
            "total_cost_usd":   total_cost,
        }
    else:
        aggregate = {
            "variant":          variant,
            "run_date":         run_date_str,
            "as_of":            as_of,
            "package_sha256":   pkg_sha256,
            "n_runs_requested": n_runs,
            "n_valid_runs":     n_valid,
            "fallback_applied": True,
            "fallback_rule":    settings["fallback_rule"],
            "picks":            [],
            "stocks":           [],
            "pairwise_spearman": [],
            "runs":             per_run,
            "total_cost_usd":   total_cost,
        }
        logger.warning(
            "[%s] only %d/%d runs valid → fallback %s",
            variant, n_valid, n_runs, settings["fallback_rule"],
        )

    (variant_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (variant_dir / "report.md").write_text(
        _render_aggregate_report(aggregate, variant, run_date_str, as_of),
        encoding="utf-8",
    )

    return {
        "variant":         variant,
        "n_runs":          n_runs,
        "n_valid_runs":    n_valid,
        "fallback_applied": aggregate["fallback_applied"],
        "picks":           aggregate["picks"],
        "total_cost_usd":  total_cost,
        "package_sha256":  pkg_sha256,
        "aggregate":       aggregate,
        "runs":            per_run,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--out-dir",  type=Path, required=True)
    p.add_argument("--n-runs",   type=int, default=None)
    args = p.parse_args()
    s = run_variant(args.variant, args.package, args.run_date, args.out_dir, n_runs=args.n_runs)
    print(json.dumps({k: v for k, v in s.items() if k not in ("parsed", "aggregate")},
                     indent=2, default=str))


if __name__ == "__main__":
    _main()
