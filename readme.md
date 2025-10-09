# AetherScan: Change Intelligence for Earth Observation
**AI-Driven Satellite Imagery Intelligence for Environmental & Defence Applications**
***Turning raw satellite pixels into actionable intelligence.***

## Overview
AetherScan is an AI-driven remote-sensing platform that automatically detects, quantifies, and visualizes landscape and infrastructure changes using multi-temporal satellite imagery.
It combines NDVI, SSIM, and NDWI analytics to distinguish vegetation loss, urban growth, deforestation, and structural changes - all within a single unified workflow.

## Key Capabilities

| Feature | Description |
|----------|--------------|
| **Hybrid AI Engine** | Combines NDVI-based spectral change detection with SSIM-based structural analysis for accurate surface intelligence. |
| **Water Masking via NDWI** | Automatically suppresses false positives over lakes, rivers, and reservoirs. |
| **Change Statistics Dashboard** | Quantifies vegetation loss/gain, net change, and total impact — instantly displayed in a live stats card. |
| **GeoJSON Export** | Generates polygonized change masks for GIS platforms (QGIS, Leaflet, ArcGIS). |
| **Streamlit Cloud Deployment** | Fully interactive, zero-setup web dashboard. Upload T1/T2 GeoTIFFs and visualize change instantly. |
| **Open-Source & Lightweight** | Built entirely with open-source Python (rasterio, OpenCV, scikit-image, Streamlit). |


## Why it matters (Thales alignment)
- **Defence & Security:** Border encroachment & unauthorized build-up detection.
- **Aeronautics & Space:** Satellite/UAV imagery processing & analytics.
- **Cybersecurity & Digital Identity:** Secure, responsible AI deployment.

## Tech Stack
- **Python 3.10+**
- **Streamlit**
- **Rasterio**
- **OpenCV (headless)**
- **NumPy / scikit-image**


## Quickstart
### Run Locally
```bash
pip install -r requirements.txt
python src/main.py --t1 data/sample_T1.tif --t2 data/sample_T2.tif --red 2 --nir 3 --out demo/output.png
```
### Streamlit Demo
https://aetherscan-p3cw48qzje2zwcj9mxdakb.streamlit.app/

### Prepare Data (optional)
To merge Sentinel-2 .jp2 bands into GeoTIFFs:
```bash
python tools/merge_bands.py --input-folder <path_to_folder_with_B02_B03_B04_B08_jp2> --output data/sample_T1.tif
```


## Future Extensions
- Sentinel-2 API integration for automated temporal data ingestion
- Real-time alert pipeline for border or deforestation surveillance
- Integration with Thales secure cloud for classified analytics

## Tips
- For Sentinel-2 bands stacked as **B02,B03,B04,B08**, pass `--red 2 --nir 3`.
- Enable water mask (NDWI) to suppress false positives over lakes/rivers.
- Use the Streamlit sidebar sliders to adjust overlay opacity and view % change.




