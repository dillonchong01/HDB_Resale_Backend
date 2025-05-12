import pandas as pd
from pathlib import Path
from typing import Any, Dict

from data_cleaning import clean_data

"""
Pipeline is to be used when there are updates to Raw Data, RPI, or HDB Features
"""

# Default File Paths
RAWDATA_CSV = Path("datasets/Resale.csv")
CLEANDATA_CSV = Path("datasets/Cleaned_Resale_Data.csv")
HDB_FEATURES_CSV = Path("datasets/HDB_Features.csv")
RPI_CSV = Path("datasets/RPI.csv")

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL", "CLEMENTI",
    "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH"
]

if __name__ == "__main__":
    # Convert Raw Data to Clean Data
    if not RAWDATA_CSV.exists():
        raise FileNotFoundError(f"Raw Data File not found: {RAWDATA_CSV}")
    df_raw = pd.read_csv(RAWDATA_CSV)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(CLEANDATA_CSV, index=False)

    # Add Engineered Features (Distances, RPI) to obtain Final Dataset
    if not HDB_FEATURES_CSV.exists():
        raise FileNotFoundError(f"HDB EngineeredFeatures File not found: {HDB_FEATURES_CSV}")
    if not RPI_CSV.exists():
        raise FileNotFoundError(f"RPI File not found: {RPI_CSV}")
    
    hdbs = pd.read_csv(HDB_FEATURES_CSV)
    rpi = pd.read_csv(RPI_CSV)
    final_df = pd.merge(df_clean, hdbs[["Address", "Distance_MRT", "Distance_Mall", "Within_1km_of_Pri"]], on='Address', how='left')
    final_df = pd.merge(final_df, rpi, on='Quarter', how='left')

    # Classify Towns into Mature/Non-Mature Estate
    final_df["Mature"] = final_df["Town"].isin(MATURE_ESTATES)
    
    final_df.drop(columns=['Quarter', 'Town', 'Address'], inplace=True)

    final_df.to_csv("datasets/Final_Resale_Data.csv", index=False)
