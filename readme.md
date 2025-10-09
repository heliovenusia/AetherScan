# AetherScan
**AI-Driven Satellite Imagery Intelligence for Environmental & Defence Applications**

> Detect deforestation, illegal construction, and land-use anomalies using open satellite data + explainable AI.

## What it does
- **Change detection:** ΔNDVI + SSIM-based change masks between two dates (T1 vs T2).
- **Visual overlays:** Heatmaps on RGB composites for instant interpretation.
- **Extensible:** Plug in YOLOv8 / SAM / ViTs for object and semantic layers.
- **GeoJSON export:** Polygonized change masks (NDVI/SSIM) for GIS integration (QGIS/Leaflet/Mapbox).
- **Stats:** Vegetation loss/gain and net change (%), water-masked via NDWI.

## Why it matters (Thales alignment)
- **Defence & Security:** Border encroachment & unauthorized build-up detection.
- **Aeronautics & Space:** Satellite/UAV imagery processing & analytics.
- **Cybersecurity & Digital Identity:** Secure, responsible AI deployment.

## Architecture
Data → Preprocess → ΔNDVI/SSIM → Threshold+Morphology → Overlays → CLI/Streamlit App


## Quickstart
```bash
git clone https://github.com/heliovenusia/AetherScan.git
cd AetherScan
pip install -r requirements.txt
python src/main.py --t1 data/sample_T1.tif --t2 data/sample_T2.tif --out demo/output.png
```

## Streamlit Demo
Coming soon: hosted link

## Run locally:
streamlit run streamlit_app.py

## Prepare Data (optional)
To merge Sentinel-2 .jp2 bands into GeoTIFFs:
python tools/merge_bands.py --input-folder <path_to_folder_with_B02_B03_B04_B08_jp2> --output data/sample_T1.tif

## Repo Layout
src/                # preprocessing, detection, visualization, main.py
data/               # sample tiles (add small test GeoTIFFs here)
models/             # (optional) YOLO/weights later
notebooks/          # exploration
demo/               # screenshots and outputs
LICENSE             # MIT

## Roadmap
* Cloud mask & co-registration upgrade
* YOLOv8 for man-made object detection
* Semantic segmentation (DeepLabV3+/SAM)
* REST API (FastAPI) + GeoJSON export
* Eval metrics & benchmark report

### Tips
- For Sentinel-2 bands stacked as **B02,B03,B04,B08**, pass `--red 2 --nir 3`.
- Enable water mask (NDWI) to suppress false positives over lakes/rivers.
- Use the Streamlit sidebar sliders to adjust overlay opacity and view % change.




