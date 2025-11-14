import os
import zipfile
from pathlib import Path

from sovia.data_preparation.utils import get_path_to_data

tmp_filepath = get_path_to_data(__file__) / "tmp"


def find_shape_file(files: list[str]) -> Path | None:
    for file in files:
        if file.endswith(".shp"):
            return tmp_filepath / file
    return None


def zwischenspeichern(file) -> list[str]:
    with zipfile.ZipFile(file) as zip_file:
        zip_file.extractall(tmp_filepath)
        return zip_file.namelist()


def temp_dateien_loeschen(names: list[str]):
    for name in names:
        os.remove(tmp_filepath / name)
