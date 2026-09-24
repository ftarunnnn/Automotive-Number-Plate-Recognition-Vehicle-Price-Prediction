import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

def train_and_evaluate_models(feature_csv="data/processed/vehicle_features.csv", models_dir="data/models", output_dir="phase_7_ml_price_prediction"):
    """
    Trains Random Forest, XGBoost, and LightGBM models for vehicle price prediction.
    Evaluates each model using MAE, RMSE, and R² scores.
    Saves trained models and evaluation summaries.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"Feature CSV '{feature_csv}' not found. Run Phase 6 first.")
        
    df = pd.read_csv(feature_csv)
    
    # Feature columns vs Target
    feature_cols = [c for c in df.columns if c.endswith("_encoded") or c.endswith("_scaled")]
    X = df[feature_cols]
    y = df["price"]
    
    # Train / Test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    metrics_report = {}
    
    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    
    rf_path = os.path.join(models_dir, "price_predictor_rf.pkl")
    joblib.dump(rf_model, rf_path)
    
    metrics_report["RandomForest"] = {
        "MAE": round(float(rf_mae), 2),
        "RMSE": round(float(rf_rmse), 2),
        "R2_Score": round(float(rf_r2), 4),
        "model_path": rf_path
    }
    print(f"[Phase 7] RandomForest -> MAE: ${rf_mae:,.2f}, RMSE: ${rf_rmse:,.2f}, R²: {rf_r2:.4f}")
    
    # 2. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_r2 = r2_score(y_test, xgb_preds)
    
    xgb_path = os.path.join(models_dir, "price_predictor_xgb.pkl")
    joblib.dump(xgb_model, xgb_path)
    
    metrics_report["XGBoost"] = {
        "MAE": round(float(xgb_mae), 2),
        "RMSE": round(float(xgb_rmse), 2),
        "R2_Score": round(float(xgb_r2), 4),
        "model_path": xgb_path
    }
    print(f"[Phase 7] XGBoost      -> MAE: ${xgb_mae:,.2f}, RMSE: ${xgb_rmse:,.2f}, R²: {xgb_r2:.4f}")
    
    # 3. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_r2 = r2_score(y_test, lgb_preds)
    
    lgb_path = os.path.join(models_dir, "price_predictor_lgb.pkl")
    joblib.dump(lgb_model, lgb_path)
    
    metrics_report["LightGBM"] = {
        "MAE": round(float(lgb_mae), 2),
        "RMSE": round(float(lgb_rmse), 2),
        "R2_Score": round(float(lgb_r2), 4),
        "model_path": lgb_path
    }
    print(f"[Phase 7] LightGBM     -> MAE: ${lgb_mae:,.2f}, RMSE: ${lgb_rmse:,.2f}, R²: {lgb_r2:.4f}")

    # Save metrics report
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    print(f"[Phase 7] Saved evaluation metrics to '{eval_path}'.")
    return metrics_report

if __name__ == "__main__":
    train_and_evaluate_models()
