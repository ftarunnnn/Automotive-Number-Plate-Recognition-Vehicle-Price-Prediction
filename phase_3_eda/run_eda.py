import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(input_csv="data/processed/processed_vehicles.csv", output_dir="phase_3_eda"):
    """
    Performs comprehensive Exploratory Data Analysis:
    1. Vehicle price distribution analysis (Mean, Median, Skewness, Kurtosis).
    2. Mileage, Age, Brand, and Fuel Type vs Price relationship analysis.
    3. Outlier detection using IQR (Interquartile Range) method.
    4. Exporting plots and JSON statistical summary report.
    """
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Processed CSV '{input_csv}' not found. Run Phase 2 first.")
        
    df = pd.read_csv(input_csv)
    current_year = 2026
    df["age"] = current_year - df["year"]
    
    # 1. Price Distribution Metrics
    price_stats = {
        "mean_price": round(float(df["price"].mean()), 2),
        "median_price": round(float(df["price"].median()), 2),
        "std_price": round(float(df["price"].std()), 2),
        "min_price": round(float(df["price"].min()), 2),
        "max_price": round(float(df["price"].max()), 2),
        "skewness": round(float(df["price"].skew()), 4),
        "kurtosis": round(float(df["price"].kurtosis()), 4)
    }
    
    # 2. Outlier Detection via IQR
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
    outlier_stats = {
        "Q1": round(float(Q1), 2),
        "Q3": round(float(Q3), 2),
        "IQR": round(float(IQR), 2),
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round(float(len(outliers) / len(df) * 100), 2)
    }
    
    # 3. Categorical Aggregations vs Price
    brand_price = df.groupby("brand")["price"].mean().round(2).to_dict()
    fuel_price = df.groupby("fuel_type")["price"].mean().round(2).to_dict()
    transmission_price = df.groupby("transmission")["price"].mean().round(2).to_dict()
    
    # Summary Report
    report = {
        "total_vehicles": len(df),
        "price_statistics": price_stats,
        "outlier_metrics": outlier_stats,
        "average_price_by_brand": brand_price,
        "average_price_by_fuel": fuel_price,
        "average_price_by_transmission": transmission_price
    }
    
    report_path = os.path.join(output_dir, "eda_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[Phase 3] Saved EDA Report to '{report_path}'.")
    
    # 4. Visualization Plots Generation
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Price Distribution Histogram & KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(df["price"], kde=True, color="#1f77b4", bins=30)
    plt.axvline(df["price"].mean(), color="red", linestyle="--", label=f"Mean: ${price_stats['mean_price']:,.0f}")
    plt.axvline(df["price"].median(), color="green", linestyle="-", label=f"Median: ${price_stats['median_price']:,.0f}")
    plt.title("Vehicle Price Distribution & Density", fontsize=14, fontweight="bold")
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "price_distribution.png"), dpi=300)
    plt.close()
    
    # Plot 2: Mileage & Vehicle Age vs Price (Scatter Plots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="mileage", y="price", hue="fuel_type", ax=axes[0], alpha=0.7)
    axes[0].set_title("Mileage vs Price", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mileage (km)")
    axes[0].set_ylabel("Price ($)")
    
    sns.scatterplot(data=df, x="age", y="price", hue="transmission", ax=axes[1], alpha=0.7)
    axes[1].set_title("Vehicle Age vs Price", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Vehicle Age (Years)")
    axes[1].set_ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "mileage_age_vs_price.png"), dpi=300)
    plt.close()
    
    # Plot 3: Brand & Fuel Type vs Price (Box Plots)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    brand_order = df.groupby("brand")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="brand", y="price", order=brand_order, ax=axes[0], palette="Spectral")
    axes[0].set_title("Brand vs Price (Ordered by Median Price)", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=30)
    
    sns.boxplot(data=df, x="fuel_type", y="price", ax=axes[1], palette="Set2")
    axes[1].set_title("Fuel Type vs Price", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "brand_fuel_vs_price.png"), dpi=300)
    plt.close()
    
    # Plot 4: Outlier Detection Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["price"], color="#ff7f0e")
    plt.title("Vehicle Price Outlier Boxplot (IQR Method)", fontsize=12, fontweight="bold")
    plt.xlabel("Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "price_outliers_boxplot.png"), dpi=300)
    plt.close()
    
    print(f"[Phase 3] Generated 4 EDA visualization plots in '{os.path.join(output_dir, 'plots')}'.")
    return report

if __name__ == "__main__":
    perform_eda()
