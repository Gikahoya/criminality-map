"""
load_vacancy.py
---------------
Loads two Statistics Canada vacancy CSVs:

  1. "Job vacancies, payroll employees, and job vacancy rate
      by industry sector, monthly, unadjusted for seasonality"
     → national-level vacancy rate per sector

  2. "Job vacancies, payroll employees, and job vacancy rate
      by provinces and territories, monthly, unadjusted for seasonality"
     → province-level total vacancy rate (all sectors combined)

Both files share the same long format where each row is one statistic.
We pivot on the "Statistics" column to get one row per (period, geo, sector).

Output of load_vacancy_by_sector():
    period          str     e.g. "2024-01"
    sector          str     vacancy dataset sector label (NAICS name)
    job_vacancies   float   number of open positions (None if suppressed)
    payroll_empl    float   number of payroll employees (None if suppressed)
    vacancy_rate    float   percentage (None if suppressed)

Output of load_vacancy_by_province():
    period          str     e.g. "2024-01"
    geo             str     "Canada" or province name
    job_vacancies   float
    payroll_empl    float
    vacancy_rate    float
"""

import pandas as pd
from pathlib import Path


# Statistics Canada STATUS codes that mean suppressed/unreliable
_SUPPRESSED_STATUSES = {"x", "F", "E", "D", "C", "B"}
# Note: B/C/D/E are quality indicators (B=good, C=acceptable, D=use with caution,
# E=use with caution, F=too unreliable). We keep B and A as reliable, suppress F.
# For this project we keep all numeric values regardless of quality flag but
# mark F as NaN since StatCan explicitly says F should not be published.
_FORCE_NULL_STATUSES = {"x", "F"}


def _pivot_statcan(raw: pd.DataFrame, sector_col: str | None) -> pd.DataFrame:
    """
    Pivots a long-format StatCan CSV so each (period, geo, sector) becomes one row
    with columns: job_vacancies, payroll_empl, vacancy_rate.

    Args:
        raw:        raw DataFrame from pd.read_csv
        sector_col: name of the sector column, or None if no sector column
    """
    raw = raw.rename(columns={
        "REF_DATE": "period",
        "GEO": "geo",
        "Statistics": "statistic",
        "VALUE": "value",
        "STATUS": "status",
    })

    # Suppress F values before pivoting
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw.loc[raw["status"].isin(_FORCE_NULL_STATUSES), "value"] = float("nan")

    # Build index columns for pivot
    index_cols = ["period", "geo"]
    if sector_col:
        raw = raw.rename(columns={sector_col: "sector"})
        index_cols.append("sector")

    # Keep only columns we need
    keep = index_cols + ["statistic", "value"]
    raw = raw[keep]

    # Pivot: rows = index_cols, columns = statistic names
    pivoted = raw.pivot_table(
        index=index_cols,
        columns="statistic",
        values="value",
        aggfunc="first",   # there should be exactly one value per cell
    ).reset_index()

    # Rename pivoted statistic columns to clean names
    pivoted.columns.name = None
    rename_map = {
        "Job vacancies": "job_vacancies",
        "Payroll employees": "payroll_empl",
        "Job vacancy rate": "vacancy_rate",
    }
    pivoted = pivoted.rename(columns=rename_map)

    # Ensure all three stat columns exist (some may be missing if all suppressed)
    for col in ["job_vacancies", "payroll_empl", "vacancy_rate"]:
        if col not in pivoted.columns:
            pivoted[col] = float("nan")

    return pivoted


def load_vacancy_by_sector(filepath: str | Path) -> pd.DataFrame:
    """
    Loads the vacancy-by-industry-sector CSV.
    Filters to Canada-level rows only (GEO == "Canada").

    Returns DataFrame with columns:
        period, sector, job_vacancies, payroll_empl, vacancy_rate
    """
    raw = pd.read_csv(filepath, dtype=str)

    # This file has a NAICS column — find its exact name
    naics_col = [c for c in raw.columns if "North American Industry" in c]
    if not naics_col:
        raise ValueError(
            "Could not find NAICS column in vacancy-by-sector file. "
            f"Columns found: {list(raw.columns)}"
        )
    naics_col = naics_col[0]

    # Filter to Canada only — we only need national vacancy rates by sector
    raw = raw[raw["GEO"] == "Canada"].copy()

    # Remove the "Total, all industries" row — handled separately
    raw = raw[raw[naics_col] != "Total, all industries"].copy()

    df = _pivot_statcan(raw, sector_col=naics_col)

    # Drop the geo column (it's always "Canada" here)
    df = df.drop(columns=["geo"])

    return df.sort_values(["period", "sector"]).reset_index(drop=True)


def load_vacancy_total_canada(filepath: str | Path) -> pd.DataFrame:
    """
    Loads the Canada-total vacancy rate (all sectors combined) from the
    vacancy-by-sector file. Used as VR(Canada, ALL) in the EVR formula.

    Returns DataFrame with columns:
        period, vacancy_rate
    """
    raw = pd.read_csv(filepath, dtype=str)

    naics_col = [c for c in raw.columns if "North American Industry" in c][0]

    # Keep only the Canada total row
    mask = (raw["GEO"] == "Canada") & (raw[naics_col] == "Total, all industries")
    raw = raw[mask].copy()

    df = _pivot_statcan(raw, sector_col=None)

    return df[["period", "vacancy_rate"]].rename(
        columns={"vacancy_rate": "vr_canada_total"}
    ).sort_values("period").reset_index(drop=True)


def load_vacancy_by_province(filepath: str | Path) -> pd.DataFrame:
    """
    Loads the vacancy-by-province CSV.
    Keeps all provinces + Canada total row.

    Returns DataFrame with columns:
        period, geo, job_vacancies, payroll_empl, vacancy_rate
    """
    raw = pd.read_csv(filepath, dtype=str)

    df = _pivot_statcan(raw, sector_col=None)

    return df.sort_values(["period", "geo"]).reset_index(drop=True)