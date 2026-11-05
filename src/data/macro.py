"""
Fetch the latest macro observations from FRED for the agent's weekly data package.

Each series is defined in config/macro_series.yaml.
Returns a list of observation dicts; the caller saves them to the weekly folder.
"""

from __future__ import annotations

import os
import yaml
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "config" / "macro_series.yaml"


def _load_series_config() -> list[dict]:
    with open(_CONFIG) as fh:
        return yaml.safe_load(fh)["series"]


def fetch_macro(api_key: str | None = None) -> list[dict]:
    """
    Fetch the latest observation for each series in macro_series.yaml.

    Parameters
    ----------
    api_key : str | None
        FRED API key. Falls back to FRED_API_KEY env variable.

    Returns
    -------
    list of dict, one per series:
        id, name, unit, transform, value, obs_date, retrieved_at
    """
    load_dotenv()
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY not set")

    fred = Fred(api_key=key)
    series_cfg = _load_series_config()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []

    for cfg in series_cfg:
        sid = cfg["id"]
        transform = cfg["transform"]

        try:
            raw = fred.get_series(sid)
            raw = raw.dropna()

            if transform == "latest":
                obs_date = raw.index[-1].date()
                value = round(float(raw.iloc[-1]), 4)

            elif transform == "yoy_pct":
                last_dt  = raw.index[-1]
                last_val = float(raw.iloc[-1])
                prior_dt = last_dt - pd.DateOffset(years=1)
                prior_s  = raw[raw.index <= prior_dt]
                if prior_s.empty:
                    raise ValueError(f"No data one year before {last_dt.date()}")
                prior_val = float(prior_s.iloc[-1])
                value     = round((last_val / prior_val - 1) * 100, 3)
                obs_date  = last_dt.date()

            else:
                raise ValueError(f"Unknown transform '{transform}' for {sid}")

            results.append({
                "id":           sid,
                "name":         cfg["name"],
                "unit":         cfg["unit"] if transform == "latest" else "%",
                "transform":    transform,
                "value":        value,
                "obs_date":     str(obs_date),
                "retrieved_at": retrieved_at,
            })
            logger.info("%s: %s  (%s)", sid, value, obs_date)

        except Exception as exc:
            logger.error("%s: fetch failed – %s", sid, exc)
            results.append({
                "id":           sid,
                "name":         cfg["name"],
                "unit":         cfg.get("unit", ""),
                "transform":    transform,
                "value":        None,
                "obs_date":     None,
                "retrieved_at": retrieved_at,
                "error":        str(exc),
            })

    return results


def macro_table(observations: list[dict]) -> str:
    """Format fetch_macro() output as a plain-text table."""
    lines = [f"  {'Series':<12} {'Name':<45} {'Value':>10}  {'Obs date':<12}"]
    lines.append("  " + "-" * 85)
    for obs in observations:
        unit  = "%" if obs["transform"] in ("latest", "yoy_pct") else obs["unit"]
        label = "(YoY)" if obs["transform"] == "yoy_pct" else ""
        val   = f"{obs['value']}" if obs["value"] is not None else "N/A"
        lines.append(
            f"  {obs['id']:<12} {obs['name']:<45} {val:>10}  {obs['obs_date'] or '':12}  {label}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    obs = fetch_macro()
    print("\nMacro snapshot\n")
    print(macro_table(obs))
