import os
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from data_transformation.feature_engineering import authenticate, get_distance, nearest_loc
from data_transformation.coordinate_api_caller import get_lat_long


# Config
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/lgbm_model.pkl"))
HDB_FEATURE_PATH = Path(os.getenv("HDB_FEATURE_PATH", "datasets/HDB_Features.csv"))
MRT_COORD_PATH = Path(os.getenv("MRT_COORD_PATH", "datasets/coordinates/MRT_LatLong.csv"))
MALL_COORD_PATH = Path(os.getenv("MALL_COORD_PATH", "datasets/coordinates/Mall_LatLong.csv"))
SCHOOL_COORD_PATH = Path(os.getenv("SCHOOL_COORD_PATH", "datasets/coordinates/School_LatLong.csv"))
MATURE_ESTATES = set([
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH"
])
FLAT_TYPE_MAP = {
    "1 Room": 0,
    "2 Room": 1,
    "3 Room": 2,
    "4 Room": 3,
    "5 Room": 4,
    "Executive": 5,
    "Multi-Gen": 6
}

# Load Model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load Lookup Datasets
hdb_features = pd.read_csv(HDB_FEATURE_PATH).set_index("Address") if HDB_FEATURE_PATH.exists() else None
mrts = pd.read_csv(MRT_COORD_PATH)
malls = pd.read_csv(MALL_COORD_PATH)
schools = pd.read_csv(SCHOOL_COORD_PATH)

# Authenticate
api_token = authenticate()

# Get Location Features
def get_location_features(address: str) -> Dict[str, Any]:
    """
    Determines location-based features based on user input of address

    Args:
        address (str): Full address string

    Returns:
        Dictionary with keys: Distance_MRT, Distance_Mall, Within_1km_of_Pri, Mature
    """
    address = address.upper().replace("BLOCK", "").replace("BLK", "").strip()
    # If Address exists in HDB_Features, Extract Location Values
    if hdb_features is not None and address in hdb_features.index:
        row = hdb_features.loc[address]
        return {
            "Distance_MRT": row["Distance_MRT"],
            "Distance_Mall": row["Distance_Mall"],
            "Within_1km_of_Pri": row["Within_1km_of_Pri"]
        }
    
    # Else, use OneMap API to get information
    df = get_lat_long([address], api_token)
    hdb_lat, hdb_long = df.loc[0, ["Lat", "Long"]]

    # Get Distance to Nearest MRT
    _, mrt_lat, mrt_long, _ = nearest_loc(hdb_lat, hdb_long, mrts)
    mrt_dist = get_distance(hdb_lat, hdb_long, mrt_lat, mrt_long, api_token)
    # Get Distance to Nearest Mall
    _, mall_lat, mall_long, _ = nearest_loc(hdb_lat, hdb_long, malls)
    mall_dist = get_distance(hdb_lat, hdb_long, mall_lat, mall_long, api_token)
    # Return True if there is oversubscribed Primary School within 1km
    _, _, _, school_1km = nearest_loc(hdb_lat, hdb_long, schools)

    return {
        "Distance_MRT": mrt_dist,
        "Distance_Mall": mall_dist,
        "Within_1km_of_Pri": school_1km
    }

def predict_price(input_features: Dict[str, Any]) -> float:
    """
    Predicts the resale price of an HDB flat using a trained LightGBM model.

    Args:
        input_features (Dict[str, Any]): Dictionary containing the following keys:
            - 'Flat_Type': Categorical type of the flat (encoded as integer)
            - 'Storey': Floor level of the unit
            - 'Floor_Area': Size of the unit in square meters
            - 'Remaining_Lease': Remaining lease in years
            - 'RPI': Resale Price Index at the time of transaction
            - 'Address': Full address string used to derive location features
            - 'Mature': Town name (used to check if estate is in mature list)

    Returns:
        float: Predicted resale price in SGD (rounded to 2 decimal places)
    """
    # Get Location Features
    location_features = get_location_features(input_features["Address"])

    features = {
        "Flat_Type": FLAT_TYPE_MAP.get(input_features["Flat_Type"]),
        "Storey": input_features["Storey"],
        "Floor_Area": input_features["Floor_Area"],
        "Remaining_Lease": input_features["Remaining_Lease"],
        "RPI": input_features["RPI"],
        "Distance_MRT": location_features["Distance_MRT"],
        "Distance_Mall": location_features["Distance_Mall"],
        "Within_1km_of_Pri": location_features["Within_1km_of_Pri"],
        "Mature": input_features["Town"].strip().upper() in MATURE_ESTATES
    }

    # Prepare Input and Predict Price
    input_df = pd.DataFrame([features])
    pred_log = model.predict(input_df, num_iteration=model.best_iteration)
    pred_price = np.expm1(pred_log[0])

    return float(round(pred_price / 1000.0) * 1000)
