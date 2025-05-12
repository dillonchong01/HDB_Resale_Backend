from typing import List, Tuple
from data_transformation.locations import STATIONS, MALLS, SCHOOLS
import requests
import pandas as pd

# OneMap API Configuration
EMAIL = "dillonchong01@gmail.com"
PASSWORD = "T0126546BClash"
TOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"

# Authenticate OneMap API
def authenticate() -> str:
    """
    Authenticate with OneMap API and returns access token
    """
    payload = {"email": EMAIL, "password": PASSWORD}
    try:
        response = requests.post(TOKEN_URL, json=payload)
        response.raise_for_status()
        return response.json().get("access_token", None)
    except requests.RequestException as e:
        print(f"Authentication failed: {e}")
        return None
    
# Obtain Latitude and Longitude from OneMaps
def get_lat_long(locations: List[str], api_token: str) -> pd.DataFrame:
    """
    Given a list of location names, retrieve latitude and longitude from OneMap.

    Args:
        locations: List of string addresses to geocode.
        api_token: Auth token for OneMap API.

    Returns:
        DataFrame with columns ['Address', 'Lat', 'Long'].
    """
    lat_long_data: List[Tuple[str, float, float]] = []

    # Get Lat - Long for each Unique Address
    for address in locations:
        try:
            response = requests.get(
                SEARCH_URL,
                params={"searchVal": address, "returnGeom": "Y", "getAddrDetails": "N"},
                headers={"Authorization": f"{api_token}"}
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            if results:
                latitude = float(results[0]["LATITUDE"])
                longitude = float(results[0]["LONGITUDE"])
            else:
                latitude, longitude = 0.0, 0.0

        except Exception as e:
            print(f"Request failed for '{address}': {e}")
            latitude, longitude = 0.0, 0.0
            
        lat_long_data.append((address, latitude, longitude))

    return pd.DataFrame(lat_long_data, columns=["Address", "Lat", "Long"])

# THIS CODE IS USED TO GET LAT LONG OF HDBS, MRTS, MALLS AND SCHOOLS
if __name__ == "__main__":
    # Authenticate and get API Token
    api_token = authenticate()
    if api_token is None:
        print("Authentication failed. Exiting...")
        exit(1)

    df = pd.read_csv("datasets/Cleaned_Resale_Data.csv")

    # Get Lat/Long of Resale HDB
    hdb_locations = df["Address"].unique()
    hdb_lat_long_df = get_lat_long(hdb_locations, api_token)
    hdb_lat_long_df.to_csv("datasets/coordinates/HDB_LatLong.csv", index=False)

    # Get Lat/Long of MRTs
    mrt_lat_long_df = get_lat_long(STATIONS, api_token)
    mrt_lat_long_df.to_csv("datasets/coordinates/MRT_LatLong.csv", index=False)

    # Get Lat/Long of Malls
    mall_lat_long_df = get_lat_long(MALLS, api_token)
    mall_lat_long_df.to_csv("datasets/coordinates/Mall_LatLong.csv", index=False)

    # Get Lat/Long of Primary Schools (that are oversubscribed)
    school_lat_long_df = get_lat_long(SCHOOLS, api_token)
    school_lat_long_df.to_csv("datasets/coordinates/School_LatLong.csv", index=False)
