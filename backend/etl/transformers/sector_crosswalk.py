"""
sector_crosswalk.py
--------------------
Maps CMA employment sector names → vacancy dataset sector names.

Some CMA sectors are aggregates of multiple vacancy sectors (e.g. CMA lumps
Wholesale + Retail together, but vacancy data splits them). In those cases,
we average the vacancy rates across the constituent sectors when computing EVR.

Structure:
    CROSSWALK = {
        "<CMA sector name>": ["<vacancy sector name 1>", "<vacancy sector name 2>", ...]
    }

    A list with one item  → direct 1-to-1 mapping
    A list with 2+ items  → aggregate: average their vacancy rates
"""

CROSSWALK: dict[str, list[str]] = {
    # --- 1-to-1 mappings ---
    "Construction [23]": [
        "Construction [23]"
    ],
    "Manufacturing [31-33]": [
        "Manufacturing [31-33]"
    ],
    "Utilities [22]": [
        "Utilities [22]"
    ],
    "Transportation and warehousing [48-49]": [
        "Transportation and warehousing [48-49]"
    ],
    "Educational services [61]": [
        "Educational services [61]"
    ],
    "Health care and social assistance [62]": [
        "Health care and social assistance [62]"
    ],
    "Accommodation and food services [72]": [
        "Accommodation and food services [72]"
    ],
    "Other services (except public administration) [81]": [
        "Other services (except public administration) [81]"
    ],
    "Public administration [91]": [
        "Public administration [91]"
    ],
    "Professional, scientific and technical services [54]": [
        "Professional, scientific and technical services [54]"
    ],
    "Business, building and other support services [55-56]": [
        "Administrative and support, waste management and remediation services [56]"
        # Note: NAICS 55 (Management of companies) exists in vacancy data separately
        # but CMA lumps 55+56 together. We use [56] as the dominant proxy since
        # [55] is tiny and often suppressed.
    ],

    # --- Aggregated mappings (CMA combines, vacancy splits) ---

    # CMA: "Agriculture [111-112, 1100, 1151-1152]"
    # Vacancy: "Agriculture, forestry, fishing and hunting [11]"
    # The CMA also has a separate forestry/fishing/mining row, so we map
    # agriculture CMA rows to the [11] vacancy row only.
    "Agriculture [111-112, 1100, 1151-1152]": [
        "Agriculture, forestry, fishing and hunting [11]"
    ],

    # CMA: "Forestry, fishing, mining, quarrying, oil and gas [21, 113-114, 1153, 2100]"
    # Vacancy splits this into [11] (agri/forestry/fishing) and [21] (mining/oil/gas).
    # We average both since the CMA mixes them.
    "Forestry, fishing, mining, quarrying, oil and gas [21, 113-114, 1153, 2100]": [
        "Agriculture, forestry, fishing and hunting [11]",
        "Mining, quarrying, and oil and gas extraction [21]",
    ],

    # CMA: "Wholesale and retail trade [41, 44-45]"
    # Vacancy splits into Wholesale [41] and Retail [44-45] separately.
    "Wholesale and retail trade [41, 44-45]": [
        "Wholesale trade [41]",
        "Retail trade [44-45]",
    ],

    # CMA: "Finance, insurance, real estate, rental and leasing [52-53]"
    # Vacancy splits into Finance+insurance [52] and Real estate [53] separately.
    "Finance, insurance, real estate, rental and leasing [52-53]": [
        "Finance and insurance [52]",
        "Real estate and rental and leasing [53]",
    ],

    # CMA: "Information, culture and recreation [51, 71]"
    # Vacancy splits into Information [51] and Arts/entertainment [71] separately.
    "Information, culture and recreation [51, 71]": [
        "Information and cultural industries [51]",
        "Arts, entertainment and recreation [71]",
    ],
}

# Reverse lookup: vacancy sector name → canonical CMA sector name
# Useful for joining vacancy data back to CMA data
VACANCY_TO_CMA: dict[str, str] = {}
for cma_sector, vacancy_sectors in CROSSWALK.items():
    for v in vacancy_sectors:
        # If a vacancy sector maps to multiple CMA sectors (like [11] maps to
        # both Agriculture and Forestry/fishing rows), we keep the first mapping.
        # This is acceptable because the aggregation happens in compute_evr.py,
        # not here.
        if v not in VACANCY_TO_CMA:
            VACANCY_TO_CMA[v] = cma_sector


def get_vacancy_sectors(cma_sector: str) -> list[str]:
    """
    Given a CMA sector name, return the list of vacancy sector names to use.
    Raises KeyError if the CMA sector is not in the crosswalk.
    """
    if cma_sector not in CROSSWALK:
        raise KeyError(
            f"CMA sector '{cma_sector}' not found in crosswalk. "
            f"Available sectors: {list(CROSSWALK.keys())}"
        )
    return CROSSWALK[cma_sector]


def is_industry_sector(label: str) -> bool:
    """
    Returns True if the Employment characteristics label is a leaf-level
    industry sector we want to keep. Filters out:
      - Aggregate groupings (Goods-producing sector, Services-producing sector)
      - Occupation rows (Management occupations, Health occupations, etc.)
      - Employment type rows (Employees, Self-employed, Public sector employees)
    """
    return label in CROSSWALK or label == "Total employed"