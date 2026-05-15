"""
compute_scores.py
-----------------
Computes the final scores for each (period, city, sector):

    S(city, sector)   = EVR(province, sector) × ESh(city, sector)
    NBS(sector)       = VR(Canada, sector) × ESh(Canada, sector)
    RI(city, sector)  = S(city, sector) / NBS(sector)
    Z(city, sector)   = (S - μ_S) / σ_S     ← population std, computed across all S values

Classification thresholds for Z-score (set empirically, adjust after seeing distribution):
    Z > 0.5   → IN_DEMAND
    Z < -0.5  → SATURATED
    otherwise → BALANCED

Inputs:
    esh_df:              output of compute_esh()
                         columns: period, city, province, sector, employment,
                                  e_city_all, esh
    evr_df:              output of compute_evr()
                         columns: period, province, cma_sector, evr
    vacancy_sector_df:   output of load_vacancy_by_sector()
                         columns: period, sector, vacancy_rate
    vacancy_canada_df:   national employment by sector for ESh(Canada, sector)
                         columns: period, sector, payroll_empl  (we derive ESh from this)

Output DataFrame columns:
    period, city, province, sector,
    employment, e_city_all, esh,
    evr, score, nbs, ri, z_score, classification
"""

import pandas as pd
import numpy as np

from backend.etl.transformers.sector_crosswalk import CROSSWALK


# --- Classification thresholds (tweak after inspecting real distribution) ---
Z_IN_DEMAND_THRESHOLD = 0.5
Z_SATURATED_THRESHOLD = -0.5


def _classify(z: float | None) -> str | None:
    if z is None or np.isnan(z):
        return None
    if z > Z_IN_DEMAND_THRESHOLD:
        return "IN_DEMAND"
    if z < Z_SATURATED_THRESHOLD:
        return "SATURATED"
    return "BALANCED"


def compute_nbs(vacancy_sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes NBS(sector) = VR(Canada, sector) × ESh(Canada, sector)

    ESh(Canada, sector) = payroll_empl(sector) / payroll_empl(ALL sectors)

    This uses the vacancy-by-sector file which has both payroll employees
    and vacancy rate at the national level.

    Returns DataFrame with columns: period, sector, nbs
    """
    df = vacancy_sector_df.copy()

    # Compute total payroll employees across all sectors for each period
    total_payroll = (
        df.groupby("period")["payroll_empl"]
        .sum()
        .reset_index()
        .rename(columns={"payroll_empl": "total_payroll"})
    )

    df = df.merge(total_payroll, on="period", how="left")

    # ESh(Canada, sector) = payroll in sector / total payroll
    df["esh_canada"] = df["payroll_empl"] / df["total_payroll"]

    # NBS = VR(Canada, sector) × ESh(Canada, sector)
    df["nbs"] = df["vacancy_rate"] * df["esh_canada"]

    # Map vacancy sector names to CMA sector names for joining later
    vacancy_to_cma = {
        v_sector: cma_sector
        for cma_sector, v_sectors in CROSSWALK.items()
        for v_sector in v_sectors
    }
    df["cma_sector"] = df["sector"].map(vacancy_to_cma)

    # For CMA sectors that aggregate multiple vacancy sectors, average NBS
    nbs_df = (
        df.groupby(["period", "cma_sector"], as_index=False)["nbs"]
        .mean()
        .rename(columns={"cma_sector": "sector"})
    )

    return nbs_df


def compute_scores(
    esh_df: pd.DataFrame,
    evr_df: pd.DataFrame,
    vacancy_sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes S, NBS, RI, Z-score, and classification for every city+sector.

    Args:
        esh_df:            output of compute_esh()
        evr_df:            output of compute_evr()
        vacancy_sector_df: output of load_vacancy_by_sector()

    Returns:
        Final scored DataFrame
    """

    # --- Step 1: Join ESh with EVR ---
    # EVR is keyed on (period, province, cma_sector)
    # ESh is keyed on (period, city, sector) where city includes province info
    df = esh_df.merge(
        evr_df.rename(columns={"cma_sector": "sector"}),
        on=["period", "province", "sector"],
        how="left",
    )

    # --- Step 2: Compute S = EVR × ESh ---
    df["score"] = df["evr"] * df["esh"]

    # --- Step 3: Compute NBS per sector ---
    nbs_df = compute_nbs(vacancy_sector_df)

    df = df.merge(nbs_df, on=["period", "sector"], how="left")

    # --- Step 4: Compute RI = S / NBS ---
    df["ri"] = df["score"] / df["nbs"]

    # --- Step 5: Compute Z-score across ALL (period, city, sector) pairs ---
    # Use population std (ddof=0) since we have the complete dataset
    mu_s = df["score"].mean()
    sigma_s = df["score"].std(ddof=0)

    print(f"[compute_scores] μ_S = {mu_s:.6f}, σ_S = {sigma_s:.6f}")
    print(f"[compute_scores] S range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")

    if sigma_s == 0:
        raise ValueError(
            "Standard deviation of S is 0 — all scores are identical. "
            "Check that your data loaded correctly."
        )

    df["z_score"] = (df["score"] - mu_s) / sigma_s

    # --- Step 6: Classify ---
    df["classification"] = df["z_score"].apply(_classify)

    # --- Step 7: Final column selection and sort ---
    output_cols = [
        "period", "city", "province", "sector",
        "employment", "e_city_all", "esh",
        "evr", "score", "nbs", "ri", "z_score", "classification",
    ]
    df = df[output_cols].sort_values(["period", "city", "sector"]).reset_index(drop=True)

    # --- Step 8: Summary stats for sanity check ---
    print(f"[compute_scores] Total rows: {len(df)}")
    print(f"[compute_scores] Classification breakdown:\n{df['classification'].value_counts()}")

    return df