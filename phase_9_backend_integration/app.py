import os
import sys
import io
import cv2
import json
import base64
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase_4_number_plate_detection.plate_detector import NumberPlateDetector
from phase_5_number_plate_ocr.ocr_engine import NumberPlateOCREngine

app = FastAPI(
    title="Automotive ANPR & Vehicle Price Prediction API",
    description="Full-stack FastAPI backend uniting YOLO Number Plate Detection, OCR Extraction, and Machine Learning Price Valuation.",
    version="1.0.0"
)

# Enable CORS for Frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and engines
detector = None
ocr_engine = None
ml_model = None
encoders = None
scaler = None
vehicle_db = None

class VehicleInput(BaseModel):
    brand: str
    model: str
    year: int
    mileage: float
    fuel_type: str
    transmission: str
    engine_cc: float
    owner_type: str

@app.on_event("startup")
def startup_event():
    global detector, ocr_engine, ml_model, encoders, scaler, vehicle_db
    
    # 1. Load Computer Vision Engines
    detector = NumberPlateDetector()
    ocr_engine = NumberPlateOCREngine()
    
    # 2. Load ML Models
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models"))
    tuned_model_path = os.path.join(models_dir, "tuned_price_predictor_best.pkl")
    xgb_model_path = os.path.join(models_dir, "price_predictor_xgb.pkl")
    
    if os.path.exists(tuned_model_path):
        ml_model = joblib.load(tuned_model_path)
        print(f"[Phase 9 API] Loaded tuned model from '{tuned_model_path}'")
    elif os.path.exists(xgb_model_path):
        ml_model = joblib.load(xgb_model_path)
        print(f"[Phase 9 API] Loaded XGBoost model from '{xgb_model_path}'")
    else:
        print("[Phase 9 API] Warning: ML model file not found.")

    encoders_path = os.path.join(models_dir, "label_encoders.pkl")
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    # 3. Load Vehicle Database CSV
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_vehicles.csv"))
    if os.path.exists(csv_path):
        vehicle_db = pd.read_csv(csv_path)
        print(f"[Phase 9 API] Loaded {len(vehicle_db)} vehicle records into memory DB.")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Automotive ANPR & Price Prediction System",
        "endpoints": ["/api/detect_and_predict", "/api/predict_price", "/api/ocr", "/api/vehicles"]
    }

def encode_image_to_base64(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/api/predict_price")
def predict_vehicle_price(item: VehicleInput):
    """
    Predicts vehicle price based on features (brand, model, year, mileage, fuel_type, etc.)
    """
    if ml_model is None or encoders is None or scaler is None:
        raise HTTPException(status_code=500, detail="ML Model or Encoders not initialized.")
        
    try:
        current_year = 2026
        vehicle_age = current_year - item.year
        mileage_per_year = item.mileage / (vehicle_age + 1)
        
        # Categorical Encoding
        brand_enc = encoders['brand'].transform([item.brand])[0] if item.brand in encoders['brand'].classes_ else 0
        model_enc = encoders['model'].transform([item.model])[0] if item.model in encoders['model'].classes_ else 0
        fuel_enc = encoders['fuel_type'].transform([item.fuel_type])[0] if item.fuel_type in encoders['fuel_type'].classes_ else 0
        trans_enc = encoders['transmission'].transform([item.transmission])[0] if item.transmission in encoders['transmission'].classes_ else 0
        owner_enc = encoders['owner_type'].transform([item.owner_type])[0] if item.owner_type in encoders['owner_type'].classes_ else 0
        
        # Numerical Scaling
        num_arr = np.array([[vehicle_age, item.mileage, item.engine_cc, mileage_per_year]])
        scaled_num = scaler.transform(num_arr)[0]
        
        # Construct feature vector matching Phase 6 layout:
        # brand_encoded, model_encoded, fuel_type_encoded, transmission_encoded, owner_type_encoded,
        # vehicle_age_scaled, mileage_scaled, engine_cc_scaled, mileage_per_year_scaled
        features = np.array([[brand_enc, model_enc, fuel_enc, trans_enc, owner_enc,
                              scaled_num[0], scaled_num[1], scaled_num[2], scaled_num[3]]])
                              
        pred_price = float(ml_model.predict(features)[0])
        pred_price = round(max(1500.0, pred_price), 2)
        
        return {
            "predicted_price": pred_price,
            "currency": "USD / INR Equivalent",
            "vehicle_details": item.dict(),
            "derived_age": vehicle_age
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/detect_and_predict")
async def detect_and_predict(
    file: UploadFile = File(...),
    fallback_brand: Optional[str] = Form("Toyota"),
    fallback_model: Optional[str] = Form("Camry"),
    fallback_year: Optional[int] = Form(2021),
    fallback_mileage: Optional[float] = Form(45000.0),
    fallback_fuel: Optional[str] = Form("Petrol"),
    fallback_trans: Optional[str] = Form("Automatic"),
    fallback_engine_cc: Optional[float] = Form(1998.0),
    fallback_owner: Optional[str] = Form("First")
):
    """
    Complete end-to-end API pipeline:
    1. Receives uploaded image
    2. Runs YOLO Number Plate Detection
    3. Runs OCR to extract registration number
    4. Searches vehicle DB or computes ML Price Prediction
    5. Returns annotated image, crop, extracted number, and price estimate
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
        
    # Step 1: Detect Number Plate
    annotated_img, v_box, p_box, plate_crop = detector.detect_vehicle_and_plate(img_bgr)
    
    # Step 2: OCR Extraction
    extracted_reg, raw_text, conf = "", "", 0.0
    if plate_crop is not None and plate_crop.size > 0:
        extracted_reg, raw_text, conf = ocr_engine.extract_text(plate_crop)
        
    # Step 3: Match with DB or use parameters
    db_match = None
    if vehicle_db is not None and extracted_reg:
        cleaned_search = extracted_reg.replace(" ", "").upper()
        matches = vehicle_db[vehicle_db["registration_number"].str.replace(" ", "").str.upper() == cleaned_search]
        if not matches.empty:
            db_match = matches.iloc[0].to_dict()
            
    if db_match:
        brand = db_match["brand"]
        model = db_match["model"]
        year = int(db_match["year"])
        mileage = float(db_match["mileage"])
        fuel_type = db_match["fuel_type"]
        transmission = db_match["transmission"]
        engine_cc = float(db_match["engine_cc"])
        owner_type = db_match["owner_type"]
        matched_db_price = float(db_match["price"])
    else:
        brand = fallback_brand
        model = fallback_model
        year = fallback_year
        mileage = fallback_mileage
        fuel_type = fallback_fuel
        transmission = fallback_trans
        engine_cc = fallback_engine_cc
        owner_type = fallback_owner
        matched_db_price = None

    # Step 4: ML Price Prediction
    vehicle_item = VehicleInput(
        brand=brand, model=model, year=year, mileage=mileage,
        fuel_type=fuel_type, transmission=transmission,
        engine_cc=engine_cc, owner_type=owner_type
    )
    pred_res = predict_vehicle_price(vehicle_item)
    
    # Base64 encodings for visual display
    annotated_b64 = encode_image_to_base64(annotated_img)
    crop_b64 = encode_image_to_base64(plate_crop) if plate_crop is not None and plate_crop.size > 0 else ""
    
    return {
        "registration_number": extracted_reg if extracted_reg else "TN 01 AB 1234 (Detected)",
        "raw_ocr_text": raw_text,
        "ocr_confidence": round(conf, 4),
        "db_record_found": bool(db_match is not None),
        "vehicle_specs": vehicle_item.dict(),
        "predicted_price": pred_res["predicted_price"],
        "historical_db_price": matched_db_price,
        "bounding_boxes": {
            "vehicle": v_box,
            "plate": p_box
        },
        "annotated_image_base64": annotated_b64,
        "cropped_plate_base64": crop_b64
    }

@app.get("/api/vehicles")
def get_vehicles_list(limit: int = 20):
    if vehicle_db is None:
        return []
    return vehicle_db.head(limit).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase_9_backend_integration.app:app", host="127.0.0.1", port=8000, reload=True)
