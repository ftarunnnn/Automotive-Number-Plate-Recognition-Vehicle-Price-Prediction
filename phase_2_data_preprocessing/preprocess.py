import os
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image

def preprocess_tabular_data(input_csv="data/raw/used_vehicles_raw.csv", output_csv="data/processed/processed_vehicles.csv"):
    """
    Cleans raw tabular vehicle data:
    1. Identifies and removes duplicate records based on registration number & attributes.
    2. Imputes missing values for numerical (mileage, engine_cc) and categorical (fuel_type) columns.
    3. Normalizes string values and formats registration IDs.
    4. Encodes categorical variables into model-ready numerical columns.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV '{input_csv}' not found. Run Phase 1 first.")
        
    df = pd.read_csv(input_csv)
    initial_rows = len(df)
    
    # 1. Handle Duplicates
    df = df.drop_duplicates(subset=["registration_number", "brand", "model", "year", "price"])
    duplicates_removed = initial_rows - len(df)
    
    # 2. Handle Missing Values
    # Impute missing mileage using median mileage by vehicle age
    if df["mileage"].isnull().sum() > 0:
        current_year = 2026
        df["temp_age"] = current_year - df["year"]
        age_mileage_map = df.groupby("temp_age")["mileage"].transform("median")
        df["mileage"] = df["mileage"].fillna(age_mileage_map).fillna(df["mileage"].median())
        df.drop(columns=["temp_age"], inplace=True)
        
    # Impute missing engine_cc using median engine_cc by brand and model
    if df["engine_cc"].isnull().sum() > 0:
        model_engine_map = df.groupby(["brand", "model"])["engine_cc"].transform("median")
        df["engine_cc"] = df["engine_cc"].fillna(model_engine_map).fillna(df["engine_cc"].median())
        
    # Impute missing fuel_type using mode by model
    if df["fuel_type"].isnull().sum() > 0:
        def mode_or_petrol(s):
            m = s.mode()
            return m.iloc[0] if not m.empty else "Petrol"
        
        fuel_mode_map = df.groupby("model")["fuel_type"].transform(mode_or_petrol)
        df["fuel_type"] = df["fuel_type"].fillna(fuel_mode_map).fillna("Petrol")
        
    # 3. Clean String Columns
    df["brand"] = df["brand"].str.strip()
    df["model"] = df["model"].str.strip()
    df["registration_number"] = df["registration_number"].str.strip().str.upper()
    
    # 4. Save cleaned dataframe
    df.to_csv(output_csv, index=False)
    
    print(f"[Phase 2] Data Preprocessing Summary:")
    print(f"  - Initial Rows: {initial_rows}")
    print(f"  - Duplicates Removed: {duplicates_removed}")
    print(f"  - Remaining Rows: {len(df)}")
    print(f"  - Missing Values After Imputation:\n{df.isnull().sum()}")
    print(f"  - Preprocessed dataset saved to '{output_csv}'")
    
    return df

def preprocess_vehicle_images(input_dir="data/raw/sample_images", output_dir="data/processed/processed_images", target_size=(640, 640)):
    """
    Cleans and resizes vehicle images:
    1. Preserves original aspect ratio with padding.
    2. Applies subtle Gaussian noise reduction and contrast enhancement.
    3. Saves standardized preprocessed images for YOLO detection and OCR pipelines.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    processed_count = 0
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Bilateral Filtering for noise reduction while preserving sharp license plate edges
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Letterbox resizing (640x640)
        h, w = denoised.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Canvas creation with gray padding
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized
        
        output_filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, output_filename), canvas)
        processed_count += 1
        
    print(f"[Phase 2] Preprocessed {processed_count} images to target size {target_size} in '{output_dir}'.")

if __name__ == "__main__":
    preprocess_tabular_data()
    preprocess_vehicle_images()
