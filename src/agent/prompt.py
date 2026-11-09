"""
Render the agent prompt from prompts/agent_prompt.md.

render_prompt(package_path, run_date, as_of, max_searches) → (system: str, user: str)

The rendered texts are byte-identical for claude_web and claude_noweb;
the only difference between variants is whether the web-search tool is
attached to the API request.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_ROOT        = Path(__file__).parents[2]
_PROMPT_FILE = _ROOT / "prompts" / "agent_prompt.md"
_SCHEMA_FILE = _ROOT / "prompts" / "output_schema.json"

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER   = "=== USER ==="


def _strip_html_comment(text: str) -> str:
    """Remove the leading <!-- ... --> documentation block."""
    return re.sub(r"^\s*<!--.*?-->\s*", "", text, flags=re.DOTALL)


def _split_parts(text: str) -> tuple[str, str]:
    """Split on the SYSTEM / USER markers; return (system_body, user_body)."""
    if _SYSTEM_MARKER not in text or _USER_MARKER not in text:
        raise ValueError(
            f"Prompt file is missing '{_SYSTEM_MARKER}' or '{_USER_MARKER}' markers"
        )
    after_system = text.split(_SYSTEM_MARKER, 1)[1]
    system_body, user_body = after_system.split(_USER_MARKER, 1)
    return system_body.strip(), user_body.strip()


def _fill(template: str, mapping: dict[str, str]) -> str:
    """Replace {key} placeholders using plain string substitution (safe with JSON braces)."""
    result = template
    for key, value in mapping.items():
        result = result.replace("{" + key + "}", value)
    return result


def render_prompt(
    package_path: Path | str,
    run_date: date | str,
    as_of: date | str,
    max_searches: int,
) -> tuple[str, str]:
    """
    Render the agent prompt for one weekly run.

    Parameters
    ----------
    package_path : Path | str
        Path to package.json written by build_package().
    run_date : date | str
        The Sunday the pipeline runs (e.g. 2026-11-08).
    as_of : date | str
        Last US close before run_date (e.g. 2026-11-06).
    max_searches : int
        Maximum number of web searches the agent may perform.

    Returns
    -------
    (system_prompt, user_message) : tuple[str, str]
        Both parts with all placeholders filled and the HTML comment stripped.
        Identical for claude_web and claude_noweb.
    """
    package_path = Path(package_path)
    sha_path     = package_path.parent / "package.sha256"

    if not package_path.exists():
        raise FileNotFoundError(f"Package not found: {package_path}")
    if not sha_path.exists():
        raise FileNotFoundError(f"SHA-256 file not found: {sha_path}")

    package_json     = package_path.read_text(encoding="utf-8")
    package_sha256   = sha_path.read_text(encoding="utf-8").strip()
    output_schema    = _SCHEMA_FILE.read_text(encoding="utf-8")

    raw   = _PROMPT_FILE.read_text(encoding="utf-8")
    clean = _strip_html_comment(raw)
    system_tpl, user_tpl = _split_parts(clean)

    run_date_obj = date.fromisoformat(str(run_date))
    run_weekday  = run_date_obj.strftime("%A")

    mapping = {
        "run_date":          str(run_date),
        "run_weekday":       run_weekday,
        "as_of":             str(as_of),
        "max_searches":      str(max_searches),
        "package_json":      package_json,
        "package_sha256":    package_sha256,
        "output_schema_json": output_schema,
    }

    system = _fill(system_tpl, mapping)
    user   = _fill(user_tpl,   mapping)
    return system, user
