import io
import zipfile

import requests
from shapely.geometry import Point

from sovia.data_preparation.export_data_from_labelingstudio import load_labels
from sovia.data_preparation.utils import create_connection, all_polygons_tablename, \
    rvr_shapes_tablename, rvr_polygons_tablename, manual_polygons_tablename, wms_links_viewname, dachumbauten_tablename, \
    scoring_tablename, fetch_coordinates_from_api, transform_coordinates_to_points, get_path_to_data


def download_nrw_shapefiles():
    print("Downloading NRW Shapefiles")
    url = "https://www.opengeodata.nrw.de/produkte/geobasis/lk/akt/hu_shp/hu_EPSG25832_Shape.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(f"{get_path_to_data(__file__)}/input/all_shapes")


def load_all_shapes():
    filepath = f"{get_path_to_data(__file__)}/input/all_shapes/hu_shp.shp"
    with create_connection() as con:
        print(f"Loading shapes into duckdb in table {all_polygons_tablename}")
        con.sql(f"""CREATE OR REPLACE TABLE {all_polygons_tablename} (
            AGS VARCHAR NOT NULL,
            OI VARCHAR NOT NULL,
            GFK VARCHAR NOT NULL,
            AKTUALITAE VARCHAR NOT NULL,
            geom GEOMETRY NOT NULL,
            PRIMARY KEY (OI)
        )""")
        con.sql(f"INSERT INTO {all_polygons_tablename} SELECT * FROM ST_Read('{filepath}')")


def load_rvr_shapes():
    filepath = f"{get_path_to_data(__file__)}/input/rvr_gebiet/rvr_gebiet.shp"
    with create_connection() as con:
        print(f"Loading rvr shapes into duckdb in table {all_polygons_tablename}")
        con.sql(f"CREATE OR REPLACE TABLE {rvr_shapes_tablename} AS (SELECT * FROM ST_Read('{filepath}'))")
        print(f"Creating rvr polygons table {rvr_polygons_tablename}")
        con.sql(f"""
            CREATE OR REPLACE TABLE {rvr_polygons_tablename} AS 
                SELECT p.OI, p.geom FROM {all_polygons_tablename} as p, {rvr_shapes_tablename} as rp 
                WHERE ST_Covers(rp.geom, p.geom)""")


def load_dachumbauten():
    filepath = f"{get_path_to_data(__file__)}/input/dachumbau/Dachumbau_gruen.csv"
    LAT_HERTEN_MITTE = "51.59638"
    LON_HERTEN_MITTE = "7.14387"
    print(f"Loading {dachumbauten_tablename} into duckdb")
    with create_connection() as con:
        df = con.sql(f"SELECT Lage FROM read_csv('{filepath}')").fetchdf()
        df["Lage"] = (df["Lage"]
                      .str.replace("\r\n", ",", regex=True)
                      .str.split("/", n=1).str[0]
                      .dropna())
        df["coordinates"] = df["Lage"].apply(
            lambda x: fetch_coordinates_from_api(x, LAT_HERTEN_MITTE, LON_HERTEN_MITTE))
        df["point"] = df["coordinates"].apply(lambda x: transform_coordinates_to_points(x))
        con.sql(
            f"""
            CREATE OR REPLACE TABLE {dachumbauten_tablename} as 
                SELECT OI, Lage, point 
                FROM {all_polygons_tablename} as p, df as d WHERE ST_Covers(p.geom, ST_GeomFromText(d.point))
            """)


def load_manual_coordinate():
    print(f"Loading {manual_polygons_tablename} into duckdb")
    with create_connection() as con:
        df = con.sql(
            f"SELECT * FROM read_csv('{get_path_to_data(__file__)}/input/dachumbau/manuelle_koordinaten_epsg25832.csv')").fetchdf()
        df["point"] = df.apply(lambda x: Point(x['x'], x['y']), axis=1)
        con.sql(f"""
        CREATE OR REPLACE TABLE {manual_polygons_tablename} AS 
            SELECT OI, point 
            FROM {all_polygons_tablename} as p, df as d 
            WHERE ST_Covers(p.geom, ST_GeomFromText(d.point))
        """)


def create_link_view():
    AUFLOESUNG = 800 * 800
    print(f"Creating {wms_links_viewname} in duckdb")
    with create_connection() as con:
        query = f"""
        CREATE OR REPLACE VIEW {wms_links_viewname} AS (
            WITH format as (
                SELECT
                    OI,
                    ST_XMin(ST_Envelope(geom)) as x1,
                    ST_YMIN(ST_Envelope(geom)) as y1,
                    ST_XMAX(ST_Envelope(geom)) as x2,
                    ST_YMAX(ST_Envelope(geom)) as y2,
                    round(sqrt({AUFLOESUNG}/((x2-x1)/(y2-y1))), 0)::INTEGER as height,
                    round({AUFLOESUNG}/sqrt({AUFLOESUNG}/((x2-x1)/(y2-y1))), 0)::INTEGER as width
                FROM {rvr_polygons_tablename}
            )
            SELECT
                p.OI,
                'https://geodaten.metropoleruhr.de/dop/top_2020?language=ger&width=' || width || '&height=' || height || '&bbox=' || x1 || '%2C' || y1 || '%2C' || x2 || '%2C' || y2 || '&crs=EPSG%3A25832&format=image%2Fpng&request=GetMap&service=WMS&styles=&transparent=true&version=1.3.0&layers=top_2020' as link_2020,
                'https://geodaten.metropoleruhr.de/dop/dop_2021?language=ger&width=' || width || '&height=' || height || '&bbox=' || x1 || '%2C' || y1 || '%2C' || x2 || '%2C' || y2 || '&crs=EPSG%3A25832&format=image%2Fpng&request=GetMap&service=WMS&styles=&transparent=true&version=1.3.0&layers=dop_2021' as link_2021,
                'https://geodaten.metropoleruhr.de/dop/top_2022?language=ger&width=' || width || '&height=' || height || '&bbox=' || x1 || '%2C' || y1 || '%2C' || x2 || '%2C' || y2 || '&crs=EPSG%3A25832&format=image%2Fpng&request=GetMap&service=WMS&styles=&transparent=true&version=1.3.0&layers=top_2022' as link_2022,
                'https://geodaten.metropoleruhr.de/dop/top_2023?language=ger&width=' || width || '&height=' || height || '&bbox=' || x1 || '%2C' || y1 || '%2C' || x2 || '%2C' || y2 || '&crs=EPSG%3A25832&format=image%2Fpng&request=GetMap&service=WMS&styles=&transparent=true&version=1.3.0&layers=top_2023' as link_2023,
                'https://geodaten.metropoleruhr.de/dop/top_2024?language=ger&width=' || width || '&height=' || height || '&bbox=' || x1 || '%2C' || y1 || '%2C' || x2 || '%2C' || y2 || '&crs=EPSG%3A25832&format=image%2Fpng&request=GetMap&service=WMS&styles=&transparent=true&version=1.3.0&layers=top_2024' as link_2024
            FROM {rvr_polygons_tablename} as p
            LEFT JOIN format as f ON f.OI = p.OI
        )"""
        con.sql(query)


def create_scoring_table():
    print(f"Creating {scoring_tablename} in duckdb")
    with create_connection() as con:
        con.sql(
            f"""
            CREATE OR REPLACE TABLE {scoring_tablename} (
                OI VARCHAR NOT NULL,
                score DOUBLE NOT NULL,
                PRIMARY KEY (OI)
            )
            """
        )
        con.sql(
            f"""
                    INSERT INTO {scoring_tablename} (SELECT * FROM read_csv('{get_path_to_data(__file__)}/input/scores.csv'))
                    """
        )


def main():
    download_nrw_shapefiles()
    load_all_shapes()
    load_rvr_shapes()
    load_dachumbauten()
    load_manual_coordinate()
    create_link_view()
    create_scoring_table()
    load_labels("first_5000_labels.csv")
    load_labels("second_labels.csv")


if __name__ == "__main__":
    main()
