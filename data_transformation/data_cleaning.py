from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas import DataFrame

# Default File Paths
INPUT_CSV = Path("backend/datasets/Resale.csv")
OUTPUT_CSV = Path("backend/datasets/Cleaned_Resale_Data.csv")

# Flat Type Mapping
FLAT_TYPE_MAP: Dict[str, int] = {
    "1 ROOM": 0,
    "2 ROOM": 1,
    "3 ROOM": 2,
    "4 ROOM": 3,
    "5 ROOM": 4,
    "EXECUTIVE": 5,
    "MULTI-GENERATION": 6,
}

# Clean Data

def clean_data(df: DataFrame) -> DataFrame:
    """
    Transforms raw HDB Resale DataFrame into cleaned format.

    Steps:
      1. Combine 'block' and 'street_name' into 'Address'.
      2. Extract lower bound of 'storey_range' into 'Storey'.
      3. Split 'month' into 'Year' and 'Month'.
      4. Convert 'remaining_lease' into fractional years.
      5. Map 'flat_type' to ordered integer categories.
      6. Select and rename relevant columns.

    Args:
        df: Raw DataFrame containing HDB resale data.

    Returns:
        Cleaned DataFrame with features:
        ['Year', 'Month', 'Town', 'Flat_Type', 'Address',
         'Storey', 'Floor_Area', 'Remaining_Lease', 'Price']
    """
    df_clean = df.copy()

    # Combine Block and Street Name to get Address
    df_clean["Address"] = df_clean["block"].astype(str) + " " + df_clean["street_name"].astype(str)
    
    # Take Lower Bound of Storey Range
    df_clean["Storey"] = pd.to_numeric(
        df_clean["storey_range"].str.extract(r"^(\d+)")[0], errors="coerce"
    )

    # Convert 'month' column to Quarter
    df_clean['Quarter'] = (
        pd.to_datetime(df_clean['month'], format='%Y-%m', errors='coerce')
        .dt.to_period('Q')
        .astype(str)
        .str.replace(r'Q', '-Q')
    )

    # Express Remaining Lease in Years
    df_clean["Remaining_Lease"] = df_clean["remaining_lease"].apply(
        lambda x: round(int(x.split()[0]) + (int(x.split()[2]) if "month" in x else 0)/12, 3)
        )
    
    # Convert Flat Type to Ordered Factor
    df_clean["Flat_Type"] = df_clean["flat_type"].map(FLAT_TYPE_MAP)
    
    # Select Relevant Columns
    selected = [
        "Quarter", "town", "Flat_Type", "Address",
        "Storey", "floor_area_sqm", "Remaining_Lease", "resale_price"
    ]
    df_clean = df_clean[selected].rename(columns={
        "town": "Town",
        "floor_area_sqm": "Floor_Area",
        "resale_price": "Price"
    })

    return df_clean

def main() -> None:
    """
    Read raw data, clean, and save.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()