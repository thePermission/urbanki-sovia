import json
import ssl
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from duckdb import connect
from pyproj import Transformer
from shapely.geometry import Point

all_polygons_tablename = "polygons"
rvr_shapes_tablename = "rvr_shapes"
rvr_polygons_tablename = "rvr_polygons"
dachumbauten_tablename = "dachumbauten_polygons"
manual_polygons_tablename = "manual_coordinates_polygons"
wms_links_viewname = "wms_links"
scoring_tablename = "scoring"
label_tablename = "label"
labeled_data_tablename = "first_training"


def create_connection():
    con = connect(f"{get_path_to_data(__file__)}/database/solardachproject.duckdb")
    con.sql("INSTALL spatial")
    con.sql("LOAD spatial")
    return con


def fetch_coordinates_from_api(query: str, lat: str, lon: str) -> list:
    SSL_CONTEXT = ssl._create_unverified_context()

    """Fetches coordinates from the Photon API for a given query."""
    encoded_query = quote(query)
    url = f"http://photon.komoot.io/api/?q={encoded_query}&limit=1&lat={lat}&lon={lon}"
    try:
        with urlopen(url, context=SSL_CONTEXT) as response:
            if response.status == 200:
                response_data = json.loads(response.read())
                features = response_data.get("features", [])
                if features:
                    return features[0]["geometry"]["coordinates"]
    except Exception as e:
        print(f"Error fetching coordinates for '{query}': {e}")
    return []


def transform_coordinates_to_points(coord: list) -> list:
    TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    """Transforms raw coordinates into transformed Point objects."""
    try:
        if coord:
            x_trans, y_trans = TRANSFORMER.transform(coord[0], coord[1])
            return Point(x_trans, y_trans)
    except Exception as e:
        print(f"Error transforming coordinates {coord}: {e}")
        return None


def get_path_to_data(file: str):
    current_file_path = Path(file)
    for parent in current_file_path.parents:
        if (parent / 'data').is_dir():
            return parent / 'data'
