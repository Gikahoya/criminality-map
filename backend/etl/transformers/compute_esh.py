"""
compute_esh.py
--------------
Computes Employment Share (ESh) for each (period, city, sector).

Formula:
    ESh(city, sector) = E(city, sector) / E(city, ALL)

Where:
    E(city, sector) = employment in thousands for that city+sector
    E(city, ALL)    = total employment in thousands for that city
                      (the "Total employed" row in the CMA data)

Input:
    cma_df: output of load_cma() — DataFrame with columns:
            period, city, province, sector, employment

Output:
    Same DataFrame with two new columns:
        e_city_all  float   total employment for that city+period
        esh         float   employment share (0.0 to 1.0), None if missing data
"""

import pandas as pd


def compute_esh(cma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ESh (employment share) to the CMA DataFrame.

    Args:
        cma_df: cleaned CMA DataFrame from load_cma()

    Returns:
        DataFrame with added columns: e_city_all, esh
    """

    # --- Step 1: Extract E(city, ALL) from "Total employed" rows ---
    total_mask = cma_df["sector"] == "Total employed"
    totals = (
        cma_df[total_mask][["period", "city", "employment"]]
        .rename(columns={"employment": "e_city_all"})
    )

    # --- Step 2: Keep only sector rows (drop the "Total employed" rows) ---
    # We don't want ESh for "Total employed" itself — it would always be 1.0
    sectors_df = cma_df[~total_mask].copy()

    # --- Step 3: Join totals onto sector rows ---
    df = sectors_df.merge(totals, on=["period", "city"], how="left")

    # --- Step 4: Compute ESh ---
    # If either E(city, sector) or E(city, ALL) is NaN, result is NaN
    df["esh"] = df["employment"] / df["e_city_all"]

    # --- Step 5: Sanity check ---
    # ESh should be between 0 and 1. Flag anything outside that range.
    invalid = df[(df["esh"].notna()) & ((df["esh"] < 0) | (df["esh"] > 1))]
    if not invalid.empty:
        print(
            f"[compute_esh] WARNING: {len(invalid)} rows have ESh outside [0, 1]. "
            f"Sample:\n{invalid.head(3)}"
        )

    return df.sort_values(["period", "city", "sector"]).reset_index(drop=True)