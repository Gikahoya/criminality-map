"""
load_cma.py
-----------
Loads the Statistics Canada CMA employment CSV:
  "Employment by industry and census metropolitan area,
   three-month moving average, unadjusted for seasonality"

Returns a clean DataFrame with one row per (period, city, sector),
keeping only leaf-level industry sectors (no occupations, no aggregates).

Output columns:
    period      str     e.g. "2024-01"
    city        str     e.g. "Toronto, Ontario"
    province    str     e.g. "Ontario"  (extracted from city string)
    sector      str     canonical CMA sector label
    employment  float   persons in thousands (None if suppressed)
"""

import pandas as pd
from pathlib import Path

from backend.etl.transformers.sector_crosswalk import is_industry_sector

# Maps city GEO strings to their province
# Statistics Canada includes the province in the city name, e.g.
# "Toronto, Ontario" — we split on the last comma to extract it.
def _extract_province(geo: str) -> str:
    """
    Extracts province name from a GEO string like 'Toronto, Ontario'.
    Falls back to the full string if no comma is found.
    """
    parts = geo.rsplit(",", 1)
    return parts[-1].strip() if len(parts) == 2 else geo.strip()


def load_cma(filepath: str | Path) -> pd.DataFrame:
    """
    Reads the CMA employment CSV and returns a clean DataFrame.

    Args:
        filepath: path to the raw Statistics Canada CSV file

    Returns:
        pd.DataFrame with columns:
            period, city, province, sector, employment
    """
    raw = pd.read_csv(filepath, dtype=str)

    # --- Step 1: Rename columns we care about ---
    raw = raw.rename(columns={
        "REF_DATE": "period",
        "GEO": "city",
        "Employment characteristics": "sector",
        "VALUE": "employment",
        "STATUS": "status",
    })

    # --- Step 2: Keep only the columns we need ---
    raw = raw[["period", "city", "sector", "employment", "status"]]

    # --- Step 3: Filter to leaf industry sectors only ---
    # is_industry_sector() returns True for sectors in our crosswalk + "Total employed"
    # This drops occupation rows, aggregate groupings, and employment-type rows
    raw = raw[raw["sector"].apply(is_industry_sector)].copy()

    # --- Step 4: Handle suppressed values ---
    # Statistics Canada uses STATUS = "x" when a value is suppressed for privacy.
    # We convert those to NaN instead of treating them as 0.
    raw["employment"] = pd.to_numeric(raw["employment"], errors="coerce")
    # STATUS "F" means too unreliable to publish — also treat as NaN
    suppressed_mask = raw["status"].isin(["x", "F"])
    raw.loc[suppressed_mask, "employment"] = float("nan")

    # --- Step 5: Extract province from city string ---
    raw["province"] = raw["city"].apply(_extract_province)

    # --- Step 6: Drop the status column (no longer needed) ---
    raw = raw.drop(columns=["status"])

    # --- Step 7: Sort for readability ---
    raw = raw.sort_values(["period", "city", "sector"]).reset_index(drop=True)

    return raw