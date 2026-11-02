"""One-off smoke test: verify Anthropic API access with web_search tool."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

MODEL = "claude-opus-5-5"
PROMPT = (
    "Find the most recent FOMC decision and summarize it in 3 sentences, with sources."
)

# pricing
INPUT_COST_PER_M = 4.0
OUTPUT_COST_PER_M = 20.0
SEARCH_COST_PER_1K = 10.0


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            }
        ],
        messages=[{"role": "user", "content": PROMPT}],
    )

    # --- metrics ---
    model_id = response.model
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    n_searches = response.usage.server_tool_use.web_search_requests

    input_cost = input_tokens / 1_000_000 * INPUT_COST_PER_M
    output_cost = output_tokens / 1_000_000 * OUTPUT_COST_PER_M
    search_cost = n_searches / 1_000 * SEARCH_COST_PER_1K
    total_cost = input_cost + output_cost + search_cost

    print(f"model:          {model_id}")
    print(f"input tokens:   {input_tokens:,}")
    print(f"output tokens:  {output_tokens:,}")
    print(f"web searches:   {n_searches}")
    print(
        f"estimated cost: ${total_cost:.4f} "
        f"(input ${input_cost:.4f} + output ${output_cost:.4f} + search ${search_cost:.4f})"
    )

    # --- save raw response ---
    out_dir = _PROJECT_ROOT / "data" / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"smoke_test_{timestamp}.json"

    raw = response.model_dump()
    out_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    print(f"raw response:   {out_path}")


if __name__ == "__main__":
    main()
