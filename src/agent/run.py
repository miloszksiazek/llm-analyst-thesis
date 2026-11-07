"""
Weekly agent run – one variant at a time.

run_variant(variant, package_path, run_date, out_dir) builds the prompt,
calls claude-opus-5-5 (adaptive thinking with fixed effort and summaries),
validates the JSON answer, retries up to max_retries times with the same
input, and writes every artifact of the run to out_dir/<variant>/.

The two variants differ ONLY in the tools attached to the API request:
    claude_web   → web_search tool with max_uses from settings.yaml
    claude_noweb → no tools

API parameters verified against anthropic==2.3.0 types:
    - thinking         = {"type": "adaptive", "display": "summarized"}
    - output_config    = {"effort": <settings.effort>}
    - tools            = [web_search_20260318] for claude_web, else absent
    - usage.server_tool_use.web_search_requests → search count
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_final_text(response: Any) -> str:
    """Concatenate all top-level text blocks from the response."""
    parts = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Parse the final text block as JSON; tolerate stray prose by extracting the outer {...}."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _collect_thinking(response: Any) -> list[str]:
    out = []
    for block in response.content:
        if getattr(block, "type", None) == "thinking":
            out.append(getattr(block, "thinking", "") or "")
    return out


def _collect_searches(response: Any) -> tuple[list[dict], list[dict]]:
    """Return (queries, retrieved_sources).  Each query is {query, tool_use_id};
    each source is {url, title, page_age}."""
    queries, sources = [], []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use":
            name = getattr(block, "name", "")
            if name == "web_search":
                inp = getattr(block, "input", {}) or {}
                queries.append({
                    "tool_use_id": getattr(block, "id", None),
                    "query":       inp.get("query"),
                })
        elif btype == "web_search_tool_result":
            content = getattr(block, "content", None)
            if not content:
                continue
            for item in content:
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
    """Return the kwargs for client.messages.create (minus the client itself)."""
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
    """Return a copy safe to persist (no API key is present in req; this is a hook for future)."""
    return json.loads(json.dumps(req, default=str))


def _one_attempt(
    client: anthropic.Anthropic,
    req: dict,
    attempt_idx: int,
    universe: list[str],
    run_date: str,
    as_of: str,
    variant: str,
    pricing: dict,
    cost_guard: float,
) -> Attempt:
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
        kf = ", ".join(s["key_factors"])
        rat = s["rationale"].replace("|", "\\|")
        lines.append(f"| {s['rank']} | {s['ticker']} | {s['score']:+d} | {kf} | {rat} |")

    lines += ["", "## Sources", ""]
    for src in parsed["sources"]:
        extra = []
        if src.get("publisher"):      extra.append(src["publisher"])
        if src.get("published_date"): extra.append(src["published_date"])
        if src.get("url"):            extra.append(src["url"])
        tail = f" — {' · '.join(extra)}" if extra else ""
        lines.append(f"- **{src['id']}** ({src['type']}) {src['title']}{tail}")

    if parsed.get("data_issues"):
        lines += ["", "## Data issues", ""]
        for di in parsed["data_issues"]:
            lines.append(f"- {di}")

    return "\n".join(lines) + "\n"


# ── Public entry point ────────────────────────────────────────────────────────

def run_variant(
    variant:      str,
    package_path: Path | str,
    run_date:     date | str,
    out_dir:      Path | str,
) -> dict[str, Any]:
    """
    Run one variant end-to-end. Returns a summary dict; writes all artifacts under
    out_dir/<variant>/.
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

    # Derive as_of from the package.
    pkg_json = json.loads(package_path.read_text(encoding="utf-8"))
    as_of    = pkg_json["meta"]["as_of"]

    # Package SHA-256 (from sibling file so we don't recompute).
    sha_file   = package_path.parent / "package.sha256"
    pkg_sha256 = sha_file.read_text(encoding="utf-8").strip()

    # Build the (variant-agnostic) prompt text.
    system, user = render_prompt(
        package_path, run_date_str, as_of, settings["max_searches"]
    )

    # Build the (variant-specific) request.
    req = _build_request(variant, system, user, settings)

    # Persist the request (minus future-proofed secrets).
    (variant_dir / "request.json").write_text(
        json.dumps(_redact_request(req), indent=2, default=str), encoding="utf-8",
    )

    load_dotenv(_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)

    pricing   = settings["pricing"]
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
        logger.warning("[%s] attempt %d failed validation: %s", variant, i, a.validation_errors)

    # Dump every attempt's raw response and parsed JSON.
    for a in attempts:
        (variant_dir / f"attempt_{a.attempt:02d}_raw.txt").write_text(a.raw_text or "", encoding="utf-8")
        if a.parsed is not None:
            (variant_dir / f"attempt_{a.attempt:02d}_parsed.json").write_text(
                json.dumps(a.parsed, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    chosen = final if final is not None else attempts[-1]

    # Validation summary (final attempt or best effort).
    (variant_dir / "validation.json").write_text(
        json.dumps({
            "variant":           variant,
            "is_valid":          chosen.validation_ok,
            "errors":            chosen.validation_errors,
            "attempt_used":      chosen.attempt,
            "attempts_total":    len(attempts),
            "fallback_applied":  (not chosen.validation_ok),
            "fallback_rule":     settings["fallback_rule"],
        }, indent=2), encoding="utf-8",
    )

    # Final parsed JSON (if any).
    if chosen.parsed is not None:
        (variant_dir / "response.json").write_text(
            json.dumps(chosen.parsed, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (variant_dir / "report.md").write_text(
            _render_report(chosen.parsed, variant, run_date_str, as_of), encoding="utf-8",
        )

    # Searches / sources / thinking.
    (variant_dir / "searches.json").write_text(
        json.dumps({
            "queries":  chosen.searches,
            "retrieved": chosen.retrieved,
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (variant_dir / "thinking_summary.txt").write_text(
        "\n\n---\n\n".join(chosen.thinking), encoding="utf-8",
    )

    # Usage, cost, model id, times, package hash.
    total_cost = sum(a.cost.get("total_usd", 0.0) for a in attempts)
    (variant_dir / "run.json").write_text(
        json.dumps({
            "variant":          variant,
            "run_date":         run_date_str,
            "as_of":            as_of,
            "package_path":     str(package_path),
            "package_sha256":   pkg_sha256,
            "model_id":         chosen.model_id,
            "settings":         settings,
            "attempts": [
                {
                    "attempt":       a.attempt,
                    "started_at":    a.started_at,
                    "finished_at":   a.finished_at,
                    "stop_reason":   a.stop_reason,
                    "usage":         a.usage,
                    "cost":          a.cost,
                    "validation_ok": a.validation_ok,
                    "errors":        a.validation_errors,
                    "error":         a.error,
                } for a in attempts
            ],
            "final_attempt":    chosen.attempt,
            "is_valid":         chosen.validation_ok,
            "web_search_count": chosen.cost.get("web_searches", 0),
            "total_cost_usd":   total_cost,
        }, indent=2, default=str), encoding="utf-8",
    )

    return {
        "variant":          variant,
        "is_valid":         chosen.validation_ok,
        "errors":           chosen.validation_errors,
        "attempts":         len(attempts),
        "final_attempt":    chosen.attempt,
        "parsed":           chosen.parsed,
        "web_searches":     chosen.cost.get("web_searches", 0),
        "input_tokens":     chosen.cost.get("input_tokens", 0),
        "output_tokens":    chosen.cost.get("output_tokens", 0),
        "cost_usd":         chosen.cost.get("total_usd", 0.0),
        "total_cost_usd":   total_cost,
        "model_id":         chosen.model_id,
        "package_sha256":   pkg_sha256,
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
    args = p.parse_args()
    s = run_variant(args.variant, args.package, args.run_date, args.out_dir)
    print(json.dumps({k: v for k, v in s.items() if k != "parsed"}, indent=2, default=str))


if __name__ == "__main__":
    _main()
