# Phase 9 — Backend Integration

## Overview
Phase 9 integrates Computer Vision (YOLO detection + OCR) and Machine Learning (Tuned Price Predictor) into a high-performance **FastAPI** web server.

## API Endpoints

### 1. `GET /`
- Service status and health check.

### 2. `POST /api/detect_and_predict`
- **Payload**: Multipart file upload (`file: UploadFile`) with optional fallback parameters.
- **Process**:
  1. Accepts uploaded vehicle image.
  2. Runs YOLOv8 object detector to draw bounding boxes around vehicle & license plate.
  3. Crops license plate and extracts registration number using OCR Engine.
  4. Queries memory database for historical registration records.
  5. Computes ML price estimation using tuned XGBoost model.
- **Response**: Base64 encoded annotated image, cropped plate, extracted registration number, vehicle specs, and predicted price.

### 3. `POST /api/predict_price`
- **Payload**: JSON vehicle specifications (`brand`, `model`, `year`, `mileage`, `fuel_type`, `transmission`, `engine_cc`, `owner_type`).
- **Response**: Resale price valuation.

### 4. `GET /api/vehicles`
- Returns historical database records.

## Execution
Start the FastAPI server:
```bash
python phase_9_backend_integration/app.py
```
Or with Uvicorn:
```bash
uvicorn phase_9_backend_integration.app:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI documentation available at: `http://127.0.0.1:8000/docs`
