import urllib
import threading
import torch
import open_clip
import cv2
from duckdb.duckdb import DuckDBPyConnection
from sentence_transformers import util
from PIL import Image
import numpy as np
import time

from sovia.data_preparation.utils import wms_links_viewname, scoring_tablename, create_connection

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
model.to(device)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def image_encoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def generate_score(url1, url2):
    try:
        test_img = url_to_image(url1)
        data_img = url_to_image(url2)
        img1 = image_encoder(test_img)
        img2 = image_encoder(data_img)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        return round(float(cos_scores[0][0]) * 100, 2)
    except Exception as ex:
        print(f"Fehler bei:\n{url1}\n{url2}")
        return 404

def score(offset: int, limit: int, connection: DuckDBPyConnection):
    print(f"start scoring {offset}")
    with connection.cursor() as con:
        data = con.sql(f"SELECT * FROM {wms_links_viewname} ORDER BY OI LIMIT {limit} OFFSET {offset}").fetchdf()
    data["score"] = data[["link_2020", "link_2024"]].apply(lambda x: generate_score(*x), axis=1)
    with connection.cursor() as con:
        con.sql(f"INSERT OR REPLACE INTO {scoring_tablename} (SELECT OI, score FROM data)")

def scoring_in_threads(anzahl_thread: int, rows_pro_thread: int) -> list[threading.Thread]:
    threads = []
    connection = create_connection()
    for i in range(0, anzahl_thread):
        offset = i * rows_pro_thread
        thread = threading.Thread(target=score, args=(offset, rows_pro_thread, connection))
        print(f"Starting Thread {i}")
        threads.append(thread)
        thread.start()
    return threads


if __name__ == "__main__":
    start = time.time()
    anzahl_thread = 100
    rows_pro_thread = 10000
    threads = scoring_in_threads(anzahl_thread, rows_pro_thread)
    for thread in threads:
        thread.join()
    end = time.time()
    print(f"Time: {end - start}")
