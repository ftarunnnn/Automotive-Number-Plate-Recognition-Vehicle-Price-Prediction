# Phase 6 — Feature Engineering

## Overview
Phase 6 constructs model features by deriving domain-specific metrics, encoding categorical variables, and standardizing numerical scales.

## Engineered Features

### 1. Derived Features
- `vehicle_age`: Calculated as $2026 - \text{year}$.
- `mileage_per_year`: Calculated as $\frac{\text{mileage}}{\text{vehicle\_age} + 1}$ to capture usage intensity.

### 2. Categorical Encoding
- **Label Encoding**: Transform non-ordinal strings into integers (`brand`, `model`, `fuel_type`, `transmission`, `owner_type`).
- Serialized encoders exported to `data/models/label_encoders.pkl`.

### 3. Numerical Scaling
- **Standard Scaling**: $z = \frac{x - \mu}{\sigma}$ transformation on `vehicle_age`, `mileage`, `engine_cc`, and `mileage_per_year`.
- Serialized scaler exported to `data/models/feature_scaler.pkl`.

## Dataset Output
- `data/processed/vehicle_features.csv`
- Artifacts: `data/models/label_encoders.pkl`, `data/models/feature_scaler.pkl`

## Execution
```bash
python phase_6_feature_engineering/feature_engineering.py
```
