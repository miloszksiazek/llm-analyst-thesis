"""
Request shape for the weekly agent: structured outputs enabled on both variants.

What's being verified:

1. `_build_request` attaches `output_config.format = {type: "json_schema",
   schema: ...}` for BOTH `claude_web` and `claude_noweb`, using the same
   (identical) schema.

2. `output_config.effort` is preserved alongside `format` — the SDK's
   `OutputConfigParam` accepts both keys simultaneously.

3. The schema sent to the API is `anthropic.transform_schema(...)` applied
   to `prompts/output_schema.json`:
   - unsupported constraints (`minItems > 1`, `maxItems`, `maxLength`,
     `minimum`, `maximum`) are relocated into field descriptions, not
     silently dropped;
   - types, enums, `required`, `additionalProperties: false`, string
     `format`s are preserved so the API still enforces them.

4. `claude_web` adds the `web_search_20260318` tool; `claude_noweb` does
   not. Both still carry the same `output_config.format`.

The malformed-JSON failure on 2026-11-19 (run 1: validation error
"JSON parse error: Expecting ',' delimiter: line 1 column 2858") is the
regression this prevents — the API now rejects any model output that does
not parse as JSON matching the structural schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anthropic  # noqa: E402

from agent.run import VARIANTS, _build_request, _load_output_schema  # noqa: E402


# Settings matching config/settings.yaml — but we don't read the file so the
# test does not depend on frozen-after-pilot values moving.
_SETTINGS = {
    "model":            "claude-opus-5-5",
    "max_tokens":       16000,
    "effort":           "high",
    "thinking_display": "summarized",
    "max_searches":     10,
}


def _make_req(variant: str) -> dict:
    return _build_request(variant, system="SYS", user="USER", settings=_SETTINGS)


def test_both_variants_carry_output_config_format():
    for v in VARIANTS:
        req = _make_req(v)
        oc = req["output_config"]
        assert oc["effort"] == "high", "effort must be kept alongside format"
        fmt = oc["format"]
        assert fmt["type"] == "json_schema"
        assert isinstance(fmt["schema"], dict)
        # Required scalars at the top of the schema:
        assert fmt["schema"]["type"] == "object"


def test_schema_is_identical_across_variants():
    """Byte-identical schema across claude_web and claude_noweb."""
    web   = _make_req("claude_web")["output_config"]["format"]["schema"]
    noweb = _make_req("claude_noweb")["output_config"]["format"]["schema"]
    assert web == noweb, (
        "The prompt is byte-identical across variants (tests/test_prompt.py); "
        "the structural output schema must be too."
    )


def test_schema_is_transform_schema_of_file():
    """The schema sent to the API is the file passed through transform_schema."""
    raw_file = json.loads((ROOT / "prompts" / "output_schema.json").read_text())
    expected = anthropic.transform_schema(raw_file)
    assert _load_output_schema() == expected


def test_transform_schema_preserves_structural_enforcement():
    """`required`, `additionalProperties: false`, types, enums must survive the
    transform — they are what the API enforces for us."""
    s = _load_output_schema()
    assert s["additionalProperties"] is False
    for key in ("run_date", "as_of", "macro_view", "stocks", "picks",
                "picks_rationale", "sources", "data_issues"):
        assert key in s["required"], f"top-level `{key}` must stay required"

    stock_item = s["properties"]["stocks"]["items"]
    assert stock_item["additionalProperties"] is False
    for key in ("ticker", "score", "rank", "rationale", "key_factors", "source_ids"):
        assert key in stock_item["required"]

    # Enum on a stock's key_factors must be preserved.
    key_factors_enum = stock_item["properties"]["key_factors"]["items"]["enum"]
    assert "price_momentum" in key_factors_enum
    assert "macro_or_sector" in key_factors_enum


def test_transform_schema_relocates_unsupported_constraints():
    """Constraints the API can't enforce must move into descriptions, not
    silently vanish — otherwise client-side validation won't know to check
    them either."""
    s = _load_output_schema()

    # stocks: file has minItems: 30, maxItems: 30 — must appear in description.
    stocks = s["properties"]["stocks"]
    assert "minItems" not in stocks or int(stocks["minItems"]) <= 1
    assert "maxItems" not in stocks  # dropped from the schema body
    assert "30" in stocks.get("description", ""), stocks

    # picks: minItems: 4, maxItems: 4 — same treatment.
    picks = s["properties"]["picks"]
    assert "4" in picks.get("description", "")

    # stock.score: minimum: -5, maximum: 5 — must land in the field description.
    score = s["properties"]["stocks"]["items"]["properties"]["score"]
    assert "5" in score.get("description", "")

    # stock.rationale: maxLength 320 — must land in the field description.
    rationale = s["properties"]["stocks"]["items"]["properties"]["rationale"]
    assert "320" in rationale.get("description", "")


def test_web_variant_adds_web_search_tool_but_keeps_format():
    req = _make_req("claude_web")
    assert any(t.get("type") == "web_search_20260318" for t in req.get("tools", []))
    # format is NOT lost when tools are present
    assert "format" in req["output_config"]


def test_noweb_variant_has_no_tools():
    req = _make_req("claude_noweb")
    assert "tools" not in req
    assert "format" in req["output_config"]


def test_output_config_param_accepts_both_keys_in_sdk_types():
    """Light type-structural check: the SDK's OutputConfigParam TypedDict
    declares both `effort` and `format` as valid keys. If this ever changes
    (e.g. an upstream rename), the test fails at import time."""
    from anthropic.types.output_config_param import OutputConfigParam
    annotations = OutputConfigParam.__annotations__
    assert "effort" in annotations
    assert "format" in annotations
