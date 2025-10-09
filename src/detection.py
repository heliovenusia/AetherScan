import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def delta_ndvi_mask(ndvi_t1, ndvi_t2):
    delta = ndvi_t2 - ndvi_t1
    # scale absolute delta for thresholding
    abs_delta = np.abs(delta)
    abs_u8 = ((abs_delta - abs_delta.min()) / (abs_delta.max() -
              abs_delta.min() + 1e-6) * 255).astype(np.uint8)
    thr, mask = cv2.threshold(abs_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return delta, mask


def ssim_change(gray_t1, gray_t2):
    score, diff = ssim(gray_t1, gray_t2, full=True)
    diff = (1 - diff)  # high=more change
    diff_u8 = (255 * (diff / (diff.max() + 1e-6))).astype(np.uint8)
    thr, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return score, mask


def apply_water_mask(mask, ndwi, thresh=0.15):
    """Zero out changes over water (NDWI high)."""
    water = (ndwi > thresh).astype(np.uint8) * 255
    inv_water = cv2.bitwise_not(water)
    return cv2.bitwise_and(mask, mask, mask=inv_water)


def change_stats(mask):
    total = mask.size
    changed = int((mask > 0).sum())
    pct = 100.0 * changed / max(total, 1)
    return changed, total, pct


def ndvi_gain_loss_stats(ndvi_t1, ndvi_t2, ndwi=None, water_thresh=0.15, delta_eps=1e-3):
    """
    Returns (loss_px, gain_px, total_px, loss_pct, gain_pct, net_pct).
    loss: NDVI decreased beyond +/−eps; gain: NDVI increased beyond eps.
    If NDWI provided, masks out water pixels (ndwi > thresh).
    """
    delta = ndvi_t2 - ndvi_t1
    if ndwi is not None:
        water = (ndwi > water_thresh)
        delta = np.where(water, 0.0, delta)

    total_px = int(delta.size)
    loss_px = int((delta < -delta_eps).sum())
    gain_px = int((delta > delta_eps).sum())
    loss_pct = 100.0 * loss_px / max(total_px, 1)
    gain_pct = 100.0 * gain_px / max(total_px, 1)
    net_pct = gain_pct - loss_pct
    return loss_px, gain_px, total_px, loss_pct, gain_pct, net_pct
