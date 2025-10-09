import numpy as np
import cv2
import rasterio


def read_raster(path):
    with rasterio.open(path) as src:
        img = src.read()
        profile = src.profile
    # img shape: (bands, H, W)
    return img, profile


def to_uint8(x):
    x = x.astype(np.float32)
    x = (255 * (x - x.min()) / (x.max() - x.min() + 1e-6)).clip(0, 255)
    return x.astype(np.uint8)


def compute_ndvi(nir_band, red_band):
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi


def align_by_resize(img_ref, img_mov):
    """Naive align: resize moving image to ref HxW."""
    H, W = img_ref.shape[-2:]
    if img_mov.shape[-2:] != (H, W):
        img_mov_resized = cv2.resize(np.moveaxis(img_mov, 0, -1),
                                     (W, H), interpolation=cv2.INTER_LINEAR)
        img_mov_resized = np.moveaxis(img_mov_resized, -1, 0)
        return img_mov_resized
    return img_mov


def compute_ndwi(green_band, nir_band):
    g = green_band.astype(np.float32)
    n = nir_band.astype(np.float32)
    ndwi = (g - n) / (g + n + 1e-6)
    return ndwi


def uint8_norm(x):
    x = x.astype(np.float32)
    x = (x - np.nanpercentile(x, 2)) / (np.nanpercentile(x, 98) - np.nanpercentile(x, 2) + 1e-6)
    x = (x.clip(0, 1) * 255.0).astype(np.uint8)
    return x
