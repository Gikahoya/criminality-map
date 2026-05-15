"""
compute_evr.py
--------------
Computes Estimated Vacancy Rate (EVR) per (period, province, sector).

Formula:
    EVR(province, sector) = VR(Canada, sector) × ( VR(province, ALL) / VR(Canada, ALL) )

This synthesizes a province+sector vacancy rate since Statistics Canada
does not publish that combination directly.

For CMA sectors that map to multiple vacancy sectors (e.g. "Wholesale and
retail trade" maps to both "Wholesale trade [41]" and "Retail trade [44-45]"),
we average the EVR across the constituent vacancy sectors.

Inputs:
    vacancy_sector_df:   output of load_vacancy_by_sector()
                         columns: period, sector, vacancy_rate
    vacancy_province_df: output of load_vacancy_by_province()
                         columns: period, geo, vacancy_rate
    vr_canada_total_df:  output of load_vacancy_total_canada()
                         columns: period, vr_canada_total

Output DataFrame columns:
    period          str
    province        str
    cma_sector      str     canonical CMA sector name
    evr             float   estimated vacancy rate for this province+sector
"""

import pandas as pd
import numpy as np

from backend.etl.transformers.sector_crosswalk import CROSSWALK


def compute_evr(
    vacancy_sector_df: pd.DataFrame,
    vacancy_province_df: pd.DataFrame,
    vr_canada_total_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes EVR for every (period, province, cma_sector) combination.

    Args:
        vacancy_sector_df:   national vacancy rates by sector
        vacancy_province_df: vacancy rates by province (all sectors combined)
        vr_canada_total_df:  Canada total vacancy rate by period

    Returns:
        DataFrame with columns: period, province, cma_sector, evr
    """

    # --- Step 1: Get VR(Canada, sector) for each vacancy sector ---
    # vacancy_sector_df already filtered to Canada only
    vr_canada_sector = vacancy_sector_df[["period", "sector", "vacancy_rate"]].rename(
        columns={"vacancy_rate": "vr_canada_sector"}
    )

    # --- Step 2: Get VR(province, ALL) for each province ---
    # Exclude the Canada row — we have that separately
    vr_province = vacancy_province_df[
        vacancy_province_df["geo"] != "Canada"
    ][["period", "geo", "vacancy_rate"]].rename(
        columns={"geo": "province", "vacancy_rate": "vr_province_total"}
    )

    # --- Step 3: Build a flat table of all (period, province, vacancy_sector) combos ---
    # Cross join province × vacancy_sector, then attach both VR values
    all_periods = vr_province["period"].unique()

    # All vacancy sectors we need
    all_vacancy_sectors = vr_canada_sector["sector"].unique()

    # Build cartesian product: period × province × vacancy_sector
    combos = pd.MultiIndex.from_product(
        [all_periods, vr_province["province"].unique(), all_vacancy_sectors],
        names=["period", "province", "sector"]
    )
    combos_df = pd.DataFrame(index=combos).reset_index()

    # Join VR(Canada, sector)
    combos_df = combos_df.merge(vr_canada_sector, on=["period", "sector"], how="left")

    # Join VR(province, ALL)
    combos_df = combos_df.merge(vr_province, on=["period", "province"], how="left")

    # Join VR(Canada, ALL)
    combos_df = combos_df.merge(vr_canada_total_df, on="period", how="left")

    # --- Step 4: Compute EVR per (period, province, vacancy_sector) ---
    # EVR = VR(Canada, sector) × ( VR(province, ALL) / VR(Canada, ALL) )
    combos_df["evr_raw"] = (
        combos_df["vr_canada_sector"]
        * (combos_df["vr_province_total"] / combos_df["vr_canada_total"])
    )

    # --- Step 5: Map vacancy sectors → CMA sectors and average if needed ---
    # Build a lookup: vacancy_sector → [cma_sectors that include it]
    vacancy_to_cma_rows = []
    for cma_sector, vacancy_sectors in CROSSWALK.items():
        for v_sector in vacancy_sectors:
            vacancy_to_cma_rows.append({
                "sector": v_sector,
                "cma_sector": cma_sector,
            })
    sector_map = pd.DataFrame(vacancy_to_cma_rows)

    # Join CMA sector labels onto our EVR rows
    combos_df = combos_df.merge(sector_map, on="sector", how="inner")

    # Average EVR across constituent vacancy sectors for each CMA sector
    evr_df = (
        combos_df
        .groupby(["period", "province", "cma_sector"], as_index=False)["evr_raw"]
        .mean()
        .rename(columns={"evr_raw": "evr"})
    )

    return evr_df.sort_values(["period", "province", "cma_sector"]).reset_index(drop=True)