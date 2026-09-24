# Phase 2 — Data Preprocessing

## Overview
Phase 2 cleanses raw vehicle tabular data and standardizes vehicle images for downstream machine learning and computer vision processing.

## Key Operations

### Tabular Data Cleaning
1. **Duplicate Elimination**: Scans unique identifiers (`registration_number`, `brand`, `model`, `year`, `price`) to drop redundant records.
2. **Missing Value Imputation**:
   - `mileage`: Imputed using group-wise median corresponding to vehicle age.
   - `engine_cc`: Imputed using group-wise median based on `brand` and `model`.
   - `fuel_type`: Imputed using modal category by model.
3. **Data Formatting**: String trimming and upper-casing registration numbers.

### Image Standardization & Cleaning
1. **Edge-Preserving Noise Reduction**: Bilateral filtering to reduce sensor noise while maintaining sharp license plate character edges.
2. **Aspect-Ratio Preserving Resizing**: Resizes vehicle images to uniform 640x640 dimensions using letterbox padding with neutral background gray.

## Output Files
- Tabular: `data/processed/processed_vehicles.csv`
- Images: `data/processed/processed_images/`

## Execution
```bash
python phase_2_data_preprocessing/preprocess.py
```
