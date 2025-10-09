import os
import argparse
from glob import glob
from typing import List, Tuple

import numpy as np
import rasterio
import cv2


def _list_band_file(folder: str, band_tag: str, exts=("tif", "tiff", "jp2")) -> str:
    # case-insensitive search
    patterns = [f"*{band_tag}*.{ext}" for ext in exts] + [f"*{band_tag.lower()}*.{ext}" for ext in exts] + \
        [f"*{band_tag.upper()}*.{ext}" for ext in exts]
    hits = []
    for pat in patterns:
        hits.extend(glob(os.path.join(folder, pat)))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"Band '{band_tag}' not found in {folder}")
    return hits[0]


def _read_band(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)  # single band
        profile = src.profile
    return arr, profile


def _resample_to(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if arr.shape == (th, tw):
        return arr
    # nearest for categorical; OK for bands here to keep georeg consistent
    return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)


def merge_bands(input_folder: str, output_tiff: str, bands: List[str]):
    # Resolve files for requested bands in given order
    files = [_list_band_file(input_folder, b) for b in bands]
    print("[INFO] Band order:", bands)
    for b, f in zip(bands, files):
        print(f"  {b}: {os.path.basename(f)}")

    # Read all bands
    arrays = []
    profiles = []
    for f in files:
        arr, prof = _read_band(f)
        arrays.append(arr)
        profiles.append(prof)

    # Reference (size/CRS) = first band
    ref = profiles[0]
    ref_hw = (ref["height"], ref["width"])

    # Check CRS consistency
    crs_set = {(p["crs"].to_string() if p.get("crs") else None) for p in profiles}
    if len(crs_set) > 1:
        raise ValueError(f'''Bands have different CRS: {
                         crs_set}. Reproject to a common CRS before stacking.''')

    # Resample any size mismatches to reference size
    arrays = [_resample_to(a, ref_hw) for a in arrays]

    # Stack to (count, H, W)
    data = np.stack(arrays, axis=0)

    # Build output profile
    out_profile = ref.copy()
    out_profile.update(
        count=len(bands),
        dtype=str(data.dtype),
        driver="GTiff"
    )
    # Some single-band sources include photometric tags; drop extras to avoid write issues
    for k in ("photometric", "tiled"):
        if k in out_profile:
            out_profile.pop(k, None)

    # Write stacked multiband GeoTIFF
    with rasterio.open(output_tiff, "w", **out_profile) as dst:
        dst.write(data)

    print(f"[OK] Merged {len(bands)} bands -> {output_tiff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stack single-band TIFF/JP2 files into a multi-band GeoTIFF.")
    ap.add_argument("--input-folder", required=True,
                    help="Folder containing per-band files (e.g., .../IMG_DATA)")
    ap.add_argument("--output", required=True, help="Output multi-band GeoTIFF path")
    ap.add_argument("--bands", nargs="+",
                    default=["B02", "B03", "B04", "B08"], help="Band tags in desired order")
    args = ap.parse_args()
    merge_bands(args.input_folder, args.output, args.bands)
