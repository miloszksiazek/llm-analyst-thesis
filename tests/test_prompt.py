"""
render_prompt() must return byte-identical (same SHA-256) output for
claude_web and claude_noweb. The variants differ only in whether the
web-search tool is attached to the API request; the prompt text is the same.

This is enforced by design: render_prompt() does not take a `variant` argument.
This test documents and locks in that invariant.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from agent.prompt import render_prompt


def _make_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "package.json"
    pkg.write_text(
        '{"meta": {"run_date": "2026-11-01", "as_of": "2026-10-30"}}\n',
        encoding="utf-8",
    )
    sha = hashlib.sha256(pkg.read_bytes()).hexdigest()
    (tmp_path / "package.sha256").write_text(sha + "\n", encoding="utf-8")
    return pkg


def _sha(parts: tuple[str, str]) -> str:
    joined = parts[0].encode("utf-8") + b"\x00" + parts[1].encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def test_render_prompt_byte_identical_for_variants(tmp_path):
    """Same inputs → same bytes. Both variants call render_prompt with identical args."""
    pkg = _make_package(tmp_path)

    web      = render_prompt(pkg, "2026-11-01", "2026-10-30", max_searches=10)
    noweb    = render_prompt(pkg, "2026-11-01", "2026-10-30", max_searches=10)

    assert web == noweb, "render_prompt must be deterministic across calls"
    assert _sha(web) == _sha(noweb), "SHA-256 of rendered prompt must match across variants"


def test_render_prompt_fills_run_weekday(tmp_path):
    """{run_weekday} is derived from run_date and must appear in the system prompt."""
    pkg = _make_package(tmp_path)
    # 2026-11-01 is a Sunday; 2026-11-02 is a Monday.
    sys_sun, _ = render_prompt(pkg, "2026-11-01", "2026-10-30", max_searches=10)
    sys_mon, _ = render_prompt(pkg, "2026-11-02", "2026-10-30", max_searches=10)
    assert "2026-11-01 (Sunday)" in sys_sun
    assert "2026-11-02 (Monday)" in sys_mon
    assert "{run_weekday}" not in sys_sun, "placeholder must be substituted"


def test_render_prompt_has_no_variant_parameter():
    """render_prompt() must not accept a `variant` kwarg – variants share one prompt."""
    import inspect
    sig = inspect.signature(render_prompt)
    assert "variant" not in sig.parameters, (
        "render_prompt must not take a variant parameter; "
        "the rendered text is the same for claude_web and claude_noweb"
    )
