import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def build_feature_engineering_pipeline(input_csv="data/processed/processed_vehicles.csv", output_csv="data/processed/vehicle_features.csv", models_dir="data/models"):
    """
    Constructs feature engineering pipeline:
    1. Derives vehicle_age from year.
    2. Performs LabelEncoding on categorical variables (brand, model, fuel_type, transmission, owner_type).
    3. Fits and transforms StandardScaler on numerical variables (vehicle_age, mileage, engine_cc).
    4. Saves feature dataset and serialized scaler & encoders.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV '{input_csv}' not found. Run Phase 2 first.")
        
    df = pd.read_csv(input_csv)
    
    # 1. Feature Derivation: Vehicle Age & Mileage per Year
    current_year = 2026
    df["vehicle_age"] = current_year - df["year"]
    df["mileage_per_year"] = df["mileage"] / (df["vehicle_age"] + 1)
    
    # 2. Categorical Encoders Creation
    cat_cols = ["brand", "model", "fuel_type", "transmission", "owner_type"]
    encoders = {}
    
    encoded_df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        encoded_df[f"{col}_encoded"] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
        
    # Save encoders dictionary
    encoders_path = os.path.join(models_dir, "label_encoders.pkl")
    joblib.dump(encoders, encoders_path)
    print(f"[Phase 6] Saved LabelEncoders for {cat_cols} at '{encoders_path}'.")
    
    # 3. Feature Selection & Scaling
    num_cols = ["vehicle_age", "mileage", "engine_cc", "mileage_per_year"]
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(encoded_df[num_cols])
    
    for i, col in enumerate(num_cols):
        encoded_df[f"{col}_scaled"] = scaled_num[:, i]
        
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Phase 6] Saved StandardScaler at '{scaler_path}'.")
    
    # Select final ML feature matrix
    feature_cols = [f"{c}_encoded" for c in cat_cols] + [f"{c}_scaled" for c in num_cols]
    final_cols = ["registration_number", "price"] + feature_cols + cat_cols + num_cols
    
    output_df = encoded_df[final_cols]
    output_df.to_csv(output_csv, index=False)
    
    print(f"[Phase 6] Engineered feature matrix saved to '{output_csv}' with shape {output_df.shape}.")
    return output_df, encoders, scaler

if __name__ == "__main__":
    build_feature_engineering_pipeline()
