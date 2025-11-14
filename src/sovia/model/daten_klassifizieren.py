import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame

from sovia.data_preparation.utils import get_path_to_data, create_connection, wms_links_viewname, \
    rvr_polygons_tablename
from sovia.model.SiameseNetworkTrainingLoop2 import SiameseNetwork, SimpleEmbeddingNet
from sovia.model.image_loader import ImageLoader


def _load_model(model, weights_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = get_path_to_data(__file__) / f"input/trained_models/saved_states/{weights_name}.pth"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _init_table(name: str):
    with create_connection() as con:
        con.sql(
            f"CREATE TABLE IF NOT EXISTS {name}_klassifizierung (OI VARCHAR, klassifizierung DOUBLE, PRIMARY KEY (OI))")


def _insert_into_table(name: str, data: DataFrame):
    with create_connection() as con:
        sql = f"INSERT OR REPLACE INTO {name}_klassifizierung SELECT * FROM data"
        con.sql(sql)


def klassifiziere_alle(model, image_loader, name: str):
    _init_table(name)
    loaded_model = _load_model(model, weights_name)
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", 'Starte Klassifizierung')
    while True:
        with create_connection() as con:
            sql = f"""
                SELECT 
                    wlv.OI as oi,
                    2020 as year_1,
                    link_2020 as link_1,
                    2024 as year_2,
                    link_2024 as link_2,
                    ST_AsText(apt.geom) as geom
                FROM {rvr_polygons_tablename} as apt
                LEFT JOIN {wms_links_viewname} as wlv
                    ON apt.oi = wlv.oi
                LEFT JOIN {name}_klassifizierung as k on k.oi = wlv.oi
                WHERE k.oi IS NULL
                LIMIT 10000
            """
            data = con.sql(sql).fetchdf()
            data_sets = np.array_split(data, 10)
            threads = []
            for np_data_set in data_sets:
                data_set = pd.DataFrame(np_data_set)
                thread = threading.Thread(target=klassifiziere_set, args=(data_set, loaded_model, image_loader, name))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            # data["klassifizierung"] = data[["oi", "year_1", "link_1", "year_2", "link_2", "geom"]].apply(
            #     lambda x: klassifiziere_row(loaded_model, image_loader, *x), axis=1)
            # _insert_into_table(name, data[["oi", "klassifizierung"]])
            if len(data) == 0:
                return
            anzahl = con.sql(f"SELECT COUNT(*) FROM {name}_klassifizierung").fetchone()[0]
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Klassifizierung abgeschlossen. {anzahl} Hausumringe klassifiziert.")


def klassifiziere_set(data: DataFrame, loaded_model, image_loader, name):
    data["klassifizierung"] = data[["oi", "year_1", "link_1", "year_2", "link_2", "geom"]].apply(
        lambda x: klassifiziere_row(loaded_model, image_loader, *x), axis=1)
    _insert_into_table(name, data[["oi", "klassifizierung"]])
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Thread Done")


def klassifiziere_row(model, image_loader, oi, year_1, link_1, year_2, link_2, geom):
    try:
        img1, img2 = image_loader.load(oi, year_1, link_1, year_2, link_2, geom)
        output1, output2 = model(img1.unsqueeze(0), img2.unsqueeze(0))
        return float(nn.functional.pairwise_distance(output1, output2))
    except Exception:
        return float(9999)


if __name__ == "__main__":
    image_loader = ImageLoader()
    model = SiameseNetwork(SimpleEmbeddingNet())
    weights_name = "first_training"
    klassifiziere_alle(model, image_loader, weights_name)
