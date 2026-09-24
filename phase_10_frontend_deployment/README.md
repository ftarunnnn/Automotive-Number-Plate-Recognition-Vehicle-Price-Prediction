# Phase 10 — Frontend & Deployment

## Overview
Phase 10 provides an interactive **Streamlit** Web Interface allowing users to upload vehicle images, visualize real-time YOLO number plate bounding box annotations, view cropped license plates, inspect extracted registration text via OCR, and receive ML-driven resale price estimates.

## Interface Features
1. **ANPR Valuation Tab**: Upload vehicle images or select from synthetic datasets $\rightarrow$ View detected bounding box overlays $\rightarrow$ Inspect extracted license plate $\rightarrow$ Get instant price valuation.
2. **EDA Market Analytics Tab**: Interactive charts for price distributions, brand valuation breakdown, and mileage-depreciation curves.
3. **ML Benchmarking & Optimization Tab**: Model comparison tables (MAE, RMSE, $R^2$) and 5-Fold Cross Validation metrics.
4. **Vehicle Records Registry Tab**: Searchable table of preprocessed vehicle specifications.

## Local Execution
Start Streamlit application:
```bash
streamlit run phase_10_frontend_deployment/app_ui.py
```

## Production Docker Deployment
Build and launch containerized dual-service stack (FastAPI Backend + Streamlit Frontend):
```bash
docker build -t automotive-anpr-val .
docker run -p 8000:8000 -p 8501:8501 automotive-anpr-val
```
Access points:
- Streamlit Web Dashboard: `http://localhost:8501`
- FastAPI Documentation: `http://localhost:8000/docs`
