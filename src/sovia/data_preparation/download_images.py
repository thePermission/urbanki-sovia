from threading import Thread

import numpy as np
import pandas as pd
from pandas import DataFrame

from sovia.data_preparation.utils import create_connection, labeled_data_tablename, wms_links_viewname, get_path_to_data
from sovia.model.image_loader import ImageLoader

img_loader = ImageLoader()

def download_all_images():
    with create_connection() as con:
        data = con.sql(f"SELECT * FROM read_csv('{get_path_to_data(__file__)}/input/training_data/second_training.csv')").to_df()
        batches = np.array_split(data, 4)
        threads = []
        for batch in batches:
            thread = Thread(target=load_images_for_batch, args=(pd.DataFrame(batch),))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

def load_images_for_batch(batch: DataFrame):
    batch[["oi", "year_1", "link_1", "year_2", "link_2", "geom"]].apply(
        lambda x: img_loader.load(*x), axis=1)

if __name__ == '__main__':
    download_all_images()