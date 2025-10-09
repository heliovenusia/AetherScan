import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import json

from src.detection import ndvi_gain_loss_stats  # NEW
from src.geoexport import mask_to_geojson       # NEW


from src.preprocessing import read_raster, uint8_norm, compute_ndvi, compute_ndwi, align_by_resize
from src.detection import delta_ndvi_mask, ssim_change, apply_water_mask, change_stats
from src.visualization import overlay_mask, stack_h

st.set_page_config(page_title="AetherScan", layout="wide")
st.title("🛰️ AetherScan — Satellite Change Intelligence (MVP)")

with st.sidebar:
    st.header("Controls")
    red = st.number_input("RED band index (0-based)", min_value=0, value=2, step=1)
    nir = st.number_input("NIR band index (0-based)", min_value=0, value=3, step=1)
    green = st.number_input("GREEN band index (for NDWI)", min_value=0, value=1, step=1)
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    ndwi_thresh = st.slider("NDWI water threshold", 0.0, 0.6, 0.15, 0.01)
    show_ndvi = st.checkbox("Compute NDVI change", value=True)
    water_mask_on = st.checkbox("Apply water mask", value=True)

t1_file = st.file_uploader("Upload T1 (GeoTIFF)", type=["tif", "tiff"])
t2_file = st.file_uploader("Upload T2 (GeoTIFF)", type=["tif", "tiff"])


def to_rgb(img):
    if img.shape[0] >= 3:
        rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        rgb = uint8_norm(rgb)
    else:
        g = uint8_norm(img[0])
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return rgb


if t1_file and t2_file:
    # Use OS-safe temp folder
    tempdir = tempfile.gettempdir()
    t1_path = os.path.join(tempdir, "aetherscan_t1.tif")
    t2_path = os.path.join(tempdir, "aetherscan_t2.tif")

    # Write uploads to disk
    with open(t1_path, "wb") as f:
        f.write(t1_file.getbuffer())
    with open(t2_path, "wb") as f:
        f.write(t2_file.getbuffer())

    try:
        img1, profile1 = read_raster(t1_path)
        img2, _ = read_raster(t2_path)

    except Exception as e:
        st.error(f"Failed to read GeoTIFFs: {e}")
        st.stop()

    img2 = align_by_resize(img1, img2)
    rgb1, rgb2 = to_rgb(img1), to_rgb(img2)

    legends, panels = [], []
    ndwi = None

    if show_ndvi and img1.shape[0] > max(red, nir) and img2.shape[0] > max(red, nir):
        red1, nir1 = img1[red], img1[nir]
        red2, nir2 = img2[red], img2[nir]
        ndvi1 = compute_ndvi(nir1, red1)
        ndvi2 = compute_ndvi(nir2, red2)
        _, mask_ndvi = delta_ndvi_mask(ndvi1, ndvi2)
        loss_px, gain_px, tot_px, loss_pct, gain_pct, net_pct = ndvi_gain_loss_stats(
            ndvi1, ndvi2, ndwi if water_mask_on else None, water_thresh=ndwi_thresh)

        # Stats Card UI
        c1, c2, c3 = st.columns(3)
        c1.metric("Vegetation LOSS", f"{loss_pct:.2f}%", f"-{loss_px} px")
        c2.metric("Vegetation GAIN", f"{gain_pct:.2f}%", f"+{gain_px} px")
        delta_sign = "+" if net_pct >= 0 else ""
        c3.metric("NET CHANGE", f"{delta_sign}{net_pct:.2f}%", None)

        if img1.shape[0] > green and img2.shape[0] > green:
            ndwi1 = compute_ndwi(img1[green], nir1)
            ndwi2 = compute_ndwi(img2[green], nir2)
            ndwi = (ndwi1 + ndwi2) / 2.0

        if water_mask_on and ndwi is not None:
            mask_ndvi = apply_water_mask(mask_ndvi, ndwi, thresh=ndwi_thresh)

        over1 = overlay_mask(rgb1, mask_ndvi, alpha=alpha)
        over2 = overlay_mask(rgb2, mask_ndvi, alpha=alpha)
        ch, tot, pct = change_stats(mask_ndvi)
        legends.append(f"NDVI change: {ch}/{tot} px = {pct:.2f}%")
        panels.extend([over1, over2])

        g1, g2 = uint8_norm(red1), uint8_norm(red2)
    else:
        g1, g2 = uint8_norm(img1[0]), uint8_norm(img2[0])

    _, mask_ssim = ssim_change(g1, g2)
    if water_mask_on and ndwi is not None:
        mask_ssim = apply_water_mask(mask_ssim, ndwi, thresh=ndwi_thresh)

    over3 = overlay_mask(rgb2, mask_ssim, alpha=alpha)
    ch2, tot2, pct2 = change_stats(mask_ssim)
    legends.append(f"SSIM change: {ch2}/{tot2} px = {pct2:.2f}%")

    # Compose safely: 0/1/2 panels + SSIM
    if len(panels) == 0:
        comp = over3
    elif len(panels) == 1:
        comp = stack_h(panels[0], over3)
    else:
        comp = stack_h(*panels, over3)

    # GeoJSON exports
    transform = profile1.get("transform", None)
    if transform is not None:
        st.subheader("Export Detected Change as GeoJSON")
    # NDVI mask export (only if computed)
    if show_ndvi and 'mask_ndvi' in locals():
        fc_ndvi = mask_to_geojson(mask_ndvi, transform)
        st.download_button(
            label="Download NDVI-change polygons (GeoJSON)",
            file_name="aetherscan_ndvi_change.geojson",
            mime="application/geo+json",
            data=bytes(json.dumps(fc_ndvi), "utf-8")
        )

    # SSIM mask export (always available)
    fc_ssim = mask_to_geojson(mask_ssim, transform)
    st.download_button(
        label="Download SSIM-change polygons (GeoJSON)",
        file_name="aetherscan_ssim_change.geojson",
        mime="application/geo+json",
        data=bytes(json.dumps(fc_ssim), "utf-8")
    )

    st.image(comp, use_container_width=True, caption="Change overlays")
    if legends:
        st.caption(" | ".join(legends))


else:
    st.info("Upload two GeoTIFFs (same area, different dates). Tip: Use bands B02,B03,B04,B08 (red=2, nir=3).")
