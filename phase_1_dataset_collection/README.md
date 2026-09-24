# Phase 1 — Dataset Collection

## Overview
Phase 1 focuses on collecting and acquiring the core datasets for both sub-systems:
1. **Used Vehicle Price & Metadata Dataset**: Tabular data containing vehicle specifications, usage history, registration details, and resale prices.
2. **Vehicle & License Plate Image Dataset**: High-resolution vehicle images featuring distinct license plates with bounding box ground truth annotations.

## Dataset Specifications

### Tabular Dataset (`data/raw/used_vehicles_raw.csv`)
- **Size**: 2,500+ records
- **Features**:
  - `registration_number`: Unique vehicle registration ID (e.g., `TN 01 AB 1234`)
  - `brand`: Manufacturer (Toyota, Honda, BMW, Mercedes, Tata, Mahindra, Hyundai, Maruti, Ford, Audi, VW, Tesla)
  - `model`: Specific model variant
  - `year`: Manufacturing year (2010 – 2025)
  - `mileage`: Total distance driven (km)
  - `fuel_type`: Fuel variant (Petrol, Diesel, Hybrid, Electric, CNG)
  - `transmission`: Gear box type (Manual, Automatic)
  - `engine_cc`: Engine displacement volume
  - `owner_type`: Ownership count (First, Second, Third, Fourth & Above)
  - `price`: Target variable - Resale price ($ / ₹)

### Image Dataset (`data/raw/sample_images/`)
- Synthetic and realistic vehicle images rendered at 640x480 resolution.
- Embedded high-contrast number plates with standard regional typography and IND badge.
- Annotated ground-truth bounding boxes stored in `annotations.json` for YOLO evaluation.

## Execution
Run the acquisition script to build raw datasets:
```bash
python phase_1_dataset_collection/dataset_generator.py
```
