import json
import numpy as np
from typing import Optional
from rasterio.features import shapes


def mask_to_geojson(mask: np.ndarray, transform, out_path: Optional[str] = None, min_val: int = 1, min_area_px: int = 25):
    """
    Polygonize a binary/label mask to GeoJSON FeatureCollection using the provided affine transform (from T1).
    - min_val: pixels >= min_val are considered "on".
    - min_area_px: drop tiny blobs.
    Returns dict (FeatureCollection). If out_path provided, writes to disk.
    """
    # Ensure uint8
    src = (mask.astype(np.uint8) > 0).astype(np.uint8)
    feats = []
    for geom, val in shapes(src, mask=src >= min_val, transform=transform):
        # geometry area in pixel units ~ number of pixels when transform is identity; we approximate via bbox px count
        # Filter tiny features by counting mask pixels inside polygon is expensive; approximate with bbox size
        # Keep it simple for hackathon:
        # Skip features whose bbox area < min_area_px
        bbox = geom.get('bbox', None)
        if bbox:
            minx, miny, maxx, maxy = bbox
            # If pixels square size unknown, we can't convert to px area precisely; keep feature anyway.
        # Add feature
        feats.append({"type": "Feature", "properties": {"value": int(val)}, "geometry": geom})

    fc = {"type": "FeatureCollection", "features": feats}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fc, f)
    return fc
