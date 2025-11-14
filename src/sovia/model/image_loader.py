import os
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import shapely.wkt
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from cachetools.func import fifo_cache
from torch import Tensor
from torchvision import transforms


class ImageLoader:
    img_cache_path = Path(__file__).parent.parent.parent.parent / "data/input/image_cache"

    image_size = (224, 224)

    def __init__(self):
        self.transformer = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @fifo_cache(maxsize=1000)
    def load(self, oi, year_1, link_1, year_2, link_2, geom) -> tuple[Tensor, Tensor]:
        image_1 = self._load_image(oi, year_1, link_1)
        image_2 = self._load_image(oi, year_2, link_2)
        mask_tensor = self._prepare_mask(geom)
        return self._prepare_image(image_1, mask_tensor), self._prepare_image(image_2, mask_tensor)

    def _load_image(self, polygon_id, year, link) -> ImageFile:
        if not self._file_exists(polygon_id, year):
            self._download_from_url(polygon_id, year, link)
        return self._load_img_from_file(polygon_id, year)

    def _file_exists(self, polygon_id: str, year: str):
        return os.path.isfile(self._get_filepath(polygon_id, year))

    def _download_from_url(self, polygon_id: str, year: str, link: str):
        while True:
            try:
                response = requests.get(link, stream=True)
                response.raise_for_status()  # Überprüfung auf HTTP-Fehler
                # Bild laden und speichern
                image = Image.open(BytesIO(response.content))
                image.save(self._get_filepath(polygon_id, year))
                break
            except Exception as e:
                print("Bild konnte nicht geladen werden.", e)
                time.sleep(1)

    def _load_img_from_file(self, polygon_id: str, year: str) -> ImageFile:
        return Image.open(self._get_filepath(polygon_id, year))

    def _get_filepath(self, polygon_id: str, year: str) -> Path:
        return self.img_cache_path / str(year) / f"{polygon_id}.png"

    def _prepare_mask(self, polygon_points: str, activation_value=255) -> Tensor:
        height, width = self.image_size
        # Validate image size
        if not (isinstance(height, (int, np.integer)) and isinstance(width, (int, np.integer))):
            raise ValueError("image_shape must be a tuple (height:int, width:int)")
        if height <= 0 or width <= 0:
            raise ValueError(f"image_shape must be positive; got {(height, width)}")

        mask = np.zeros((height, width), dtype=np.uint8)

        # Parse polygon robustly
        try:
            geom = shapely.wkt.loads(polygon_points) if isinstance(polygon_points, str) else None
        except Exception:
            geom = None

        if geom is None:
            # Nothing to draw
            return mask

        # Handle MultiPolygon by selecting the largest
        try:
            from shapely.geometry import Polygon, MultiPolygon
            if isinstance(geom, MultiPolygon):
                if len(geom.geoms) == 0:
                    return mask
                geom = max(geom.geoms, key=lambda g: g.area)
            if not isinstance(geom, Polygon):
                return mask
        except Exception:
            return mask

        # Extract exterior coordinates
        try:
            points = list(geom.exterior.coords)
            # Drop duplicate closing point
            if len(points) >= 2 and np.allclose(points[0], points[-1], atol=1e-9):
                points = points[:-1]
        except Exception:
            return mask

        # Must have at least a triangle
        if len(points) < 3:
            return mask

        polygon_np = np.array(points, dtype=np.float32)
        # Normalize to local origin
        x_min = float(np.min(polygon_np[:, 0]))
        y_min = float(np.min(polygon_np[:, 1]))
        polygon_np[:, 0] -= x_min
        polygon_np[:, 1] -= y_min

        # Compute extents; guard against zero (degenerate)
        x_max = float(np.max(polygon_np[:, 0]))
        y_max = float(np.max(polygon_np[:, 1]))
        if x_max <= 0.0 or y_max <= 0.0:
            # Degenerate (line or point) -> return empty mask
            return mask

        # Map to pixel grid [0, width-1] and [0, height-1]
        # Use np.clip to ensure indices are in-bounds
        polygon_px = polygon_np.copy()
        polygon_px[:, 0] = np.round((polygon_px[:, 0] / x_max) * (width - 1))
        polygon_px[:, 1] = np.round((polygon_px[:, 1] / y_max) * (height - 1))
        polygon_px[:, 0] = np.clip(polygon_px[:, 0], 0, width - 1)
        polygon_px[:, 1] = np.clip(polygon_px[:, 1], 0, height - 1)

        polygon_px = polygon_px.astype(np.int32)

        # Additional safety: ensure we still have >=3 unique points after rounding
        if len(np.unique(polygon_px, axis=0)) < 3:
            return mask

        cv2.fillPoly(mask, [polygon_px], int(activation_value))
        mask = np.flip(mask, axis=0).copy()
        mask_tensor = torch.from_numpy(np.asarray(mask)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.contiguous()
        return mask_tensor

    def _prepare_image(self, image: ImageFile, mask_tensor: Tensor) -> Tensor:
        image_rgb = image.convert("RGB")
        image_transformed = self.transformer(image_rgb)
        return torch.cat([image_transformed, mask_tensor], dim=0)
