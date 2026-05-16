"""
run_etl.py
----------
ETL entry point. Run this to load all Statistics Canada CSVs,
compute scores, and write results to PostgreSQL.

Usage:
    python -m backend.etl.run_etl

Environment variables (set in .env):
    DATABASE_URL   PostgreSQL connection string
                   e.g. postgresql://user:password@localhost:5432/jobmarket

CSV files (place in backend/data/raw/):
    cma_employment.csv          Employment by industry and CMA
    vacancy_by_sector.csv       Job vacancies by industry sector
    vacancy_by_province.csv     Job vacancies by province
"""

import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from backend.etl.loaders.load_cma import load_cma
from backend.etl.loaders.load_vacancy import (
    load_vacancy_by_sector,
    load_vacancy_by_province,
    load_vacancy_total_canada,
)
from backend.etl.transformers.compute_esh import compute_esh
from backend.etl.transformers.compute_evr import compute_evr
from backend.etl.transformers.compute_scores import compute_scores


# --- Config ---
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

CSV_FILES = {
    "cma":              DATA_DIR / "cma_employment.csv",
    "vacancy_sector":   DATA_DIR / "vacancy_by_sector.csv",
    "vacancy_province": DATA_DIR / "vacancy_by_province.csv",
}


def get_db_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set. "
            "Add it to your .env file."
        )
    return create_engine(url)


def validate_files():
    """Check all input CSVs exist before starting."""
    missing = [str(p) for p in CSV_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSV files:\n" + "\n".join(f"  {f}" for f in missing) +
            f"\n\nPlace your Statistics Canada CSVs in: {DATA_DIR}"
        )


def main():
    print("=" * 60)
    print("Canada Job Market ETL")
    print("=" * 60)

    # --- Validate inputs ---
    validate_files()
    print("[1/6] Input files found ✓")

    # --- Load raw data ---
    print("[2/6] Loading CSVs...")
    cma_df = load_cma(CSV_FILES["cma"])
    print(f"      CMA:             {len(cma_df):,} rows, "
          f"{cma_df['city'].nunique()} cities, "
          f"{cma_df['period'].nunique()} periods")

    vacancy_sector_df = load_vacancy_by_sector(CSV_FILES["vacancy_sector"])
    print(f"      Vacancy/sector:  {len(vacancy_sector_df):,} rows, "
          f"{vacancy_sector_df['sector'].nunique()} sectors")

    vacancy_province_df = load_vacancy_by_province(CSV_FILES["vacancy_province"])
    print(f"      Vacancy/province:{len(vacancy_province_df):,} rows, "
          f"{vacancy_province_df['geo'].nunique()} geos")

    vr_canada_total_df = load_vacancy_total_canada(CSV_FILES["vacancy_sector"])
    print(f"      Canada VR total: {len(vr_canada_total_df):,} periods")

    # --- Compute ESh ---
    print("[3/6] Computing Employment Share (ESh)...")
    esh_df = compute_esh(cma_df)
    null_esh = esh_df["esh"].isna().sum()
    print(f"      {len(esh_df):,} rows | {null_esh} null ESh values (suppressed data)")

    # --- Compute EVR ---
    print("[4/6] Computing Estimated Vacancy Rate (EVR)...")
    evr_df = compute_evr(vacancy_sector_df, vacancy_province_df, vr_canada_total_df)
    print(f"      {len(evr_df):,} rows")

    # --- Compute scores ---
    print("[5/6] Computing S, NBS, RI, Z-score...")
    scores_df = compute_scores(esh_df, evr_df, vacancy_sector_df)

    # --- Write to database ---
    print("[6/6] Writing to PostgreSQL...")
    engine = get_db_engine()

    scores_df.to_sql(
        name="job_market_scores",
        con=engine,
        if_exists="replace",   # drop and recreate on each ETL run
        index=False,
        method="multi",        # batch inserts for speed
        chunksize=1000,
    )

    print(f"      Wrote {len(scores_df):,} rows to 'job_market_scores' table ✓")
    print()
    print("ETL complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()