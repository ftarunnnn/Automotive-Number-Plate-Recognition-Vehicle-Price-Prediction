# Phase 3 — Exploratory Data Analysis (EDA)

## Overview
Phase 3 performs statistical analysis and data visualization to analyze price patterns across vehicle brands, models, mileage, age, fuel types, and transmissions.

## Analysis Components
1. **Vehicle Price Distribution**: Evaluates mean, median, skewness, and kurtosis to understand price spread and skewness.
2. **Feature vs Price Relationships**:
   - **Mileage & Age vs Price**: Inverse exponential relationship demonstrating vehicle depreciation over time and usage.
   - **Brand & Fuel Type vs Price**: Evaluates luxury tier premiums (Mercedes, BMW, Audi, Tesla) versus economy brands (Maruti, Hyundai, Tata).
3. **Outlier Detection**: Employs Interquartile Range ($IQR = Q3 - Q1$) thresholding to flag extreme price anomalies.

## Generated Artefacts
- `phase_3_eda/eda_report.json`: JSON output containing summary statistics.
- Visual Plots (`phase_3_eda/plots/`):
  - `price_distribution.png`: Histogram & Kernel Density Estimation.
  - `mileage_age_vs_price.png`: Scatter plots for mileage and vehicle age against price.
  - `brand_fuel_vs_price.png`: Ordered box plots across luxury and budget brands and fuel variants.
  - `price_outliers_boxplot.png`: Outlier detection box plot.

## Execution
```bash
python phase_3_eda/run_eda.py
```
