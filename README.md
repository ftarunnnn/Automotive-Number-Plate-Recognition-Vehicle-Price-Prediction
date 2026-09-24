# Automotive — Number Plate Recognition & Vehicle Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Valuation_R%C2%B2_0.9736-orange.svg)](https://xgboost.readthedocs.io/)

An end-to-end Automotive Intelligence system combining **YOLOv8 Computer Vision Object Detection**, **Optical Character Recognition (OCR)**, and **Gradient Boosted Machine Learning (XGBoost/LightGBM)** to recognize vehicle license plates and estimate resale market valuations.

---

## 📌 Project Architecture

```
                                    +----------------------------------------+
                                    |        Vehicle Image Upload           |
                                    +----------------------------------------+
                                                        |
                                                        v
                                    +----------------------------------------+
                                    | Phase 4: YOLOv8 Vehicle & Plate Detector|
                                    +----------------------------------------+
                                                        |
                                                        v
                                    +----------------------------------------+
                                    | Phase 5: Cropped Plate OCR Engine      |
                                    | (CLAHE + Otsu + EasyOCR/PyTesseract)   |
                                    +----------------------------------------+
                                                        |
                                                        v
                                    +----------------------------------------+
                                    | Registration Number + Vehicle Lookup  |
                                    +----------------------------------------+
                                                        |
                                                        v
                                    +----------------------------------------+
                                    | Phase 8: Tuned XGBoost Valuation Model |
                                    | ($R^2 = 0.9736, \text{MAE} = \$473$)   |
                                    +----------------------------------------+
                                                        |
                                                        v
                                    +----------------------------------------+
                                    | Phase 9 & 10: FastAPI & Streamlit UI   |
                                    +----------------------------------------+
```

---

## 🚀 10-Phase Project Roadmap & Phase Tracker

| Phase | Phase Name | Status | Key Deliverables & Outputs |
|---|---|---|---|
| **Phase 1** | **Dataset Collection** | ✅ Completed | 2,500+ Used vehicle records CSV (`data/raw/used_vehicles_raw.csv`) & sample license plate vehicle images |
| **Phase 2** | **Data Preprocessing** | ✅ Completed | Missing value imputation, duplicate removal, bilateral noise filter, 640x640 letterbox resizing (`data/processed/processed_vehicles.csv`) |
| **Phase 3** | **EDA** | ✅ Completed | Statistical summaries (`eda_report.json`), price distribution histograms, brand & fuel box plots (`phase_3_eda/plots/`) |
| **Phase 4** | **Number Plate Detection** | ✅ Completed | YOLOv8 nano detector + OpenCV morphological gradient fallback (`phase_4_number_plate_detection/`) |
| **Phase 5** | **Number Plate OCR** | ✅ Completed | License plate cropping, CLAHE enhancement, EasyOCR/PyTesseract, regex post-processing (`phase_5_number_plate_ocr/`) |
| **Phase 6** | **Feature Engineering** | ✅ Completed | Vehicle age derivation, mileage per year, label encoding & standard scaling (`data/processed/vehicle_features.csv`) |
| **Phase 7** | **ML Price Prediction** | ✅ Completed | RandomForest, XGBoost, LightGBM model training ($R^2 = 0.9760$) (`data/models/`) |
| **Phase 8** | **Model Optimization** | ✅ Completed | GridSearchCV hyperparameter tuning & 5-Fold Cross Validation ($\text{MAE} = \$473.61$) (`phase_8_model_optimization/`) |
| **Phase 9** | **Backend Integration** | ✅ Completed | FastAPI RESTful API server (`phase_9_backend_integration/app.py`) |
| **Phase 10** | **Frontend & Deployment** | ✅ Completed | Interactive Streamlit Dashboard UI (`phase_10_frontend_deployment/app_ui.py`), Dockerfile & Procfile |

---

## 📊 Machine Learning Model Benchmarks

| Model Algorithm | MAE ($) | RMSE ($) | $R^2$ Score | Status |
|---|---|---|---|---|
| **RandomForest Regressor** | \$845.50 | \$1,358.80 | 0.9737 | Trained |
| **XGBoost Regressor** | \$813.26 | \$1,340.97 | 0.9744 | Trained |
| **LightGBM Regressor** | \$805.08 | \$1,299.12 | 0.9760 | Trained |
| **Tuned XGBoost (Phase 8 Best)** | **\$473.61** | **\$982.15** | **0.9736** | **Production Best** |

---

## 💻 Quick Start & Execution Guide

### 1. Environment Setup
```bash
git clone https://github.com/ftarunnnn/Automotive-Number-Plate-Recognition-Vehicle-Price-Prediction.git
cd Automotive-Number-Plate-Recognition-Vehicle-Price-Prediction
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Phases 1 to 8)
```bash
python phase_1_dataset_collection/dataset_generator.py
python phase_2_data_preprocessing/preprocess.py
python phase_3_eda/run_eda.py
python phase_4_number_plate_detection/plate_detector.py
python phase_5_number_plate_ocr/ocr_engine.py
python phase_6_feature_engineering/feature_engineering.py
python phase_7_ml_price_prediction/train_models.py
python phase_8_model_optimization/optimize_models.py
```

### 3. Launch FastAPI Backend (Phase 9)
```bash
uvicorn phase_9_backend_integration.app:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI docs available at: `http://127.0.0.1:8000/docs`

### 4. Launch Streamlit Web UI (Phase 10)
```bash
streamlit run phase_10_frontend_deployment/app_ui.py
```
Dashboard available at: `http://localhost:8501`

---

## 🐳 Docker Container Deployment
```bash
docker build -t automotive-anpr .
docker run -p 8000:8000 -p 8501:8501 automotive-anpr
```

---

## 📜 License
Developed for Automotive Computer Vision & Used Vehicle Price Valuation Research.