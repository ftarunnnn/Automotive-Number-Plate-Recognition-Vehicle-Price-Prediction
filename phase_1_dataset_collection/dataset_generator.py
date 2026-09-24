import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_vehicle_tabular_dataset(num_samples=2500, output_csv="data/raw/used_vehicles_raw.csv"):
    """
    Generates a realistic used-vehicle dataset with price, brand, model, year,
    mileage, fuel type, transmission, engine size, owner type, and registration number.
    Includes controlled missing values and duplicates for Phase 2 data preprocessing.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    random.seed(42)
    np.random.seed(42)
    
    brands_models = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Fortuner", "Innova"],
        "Honda": ["Civic", "Accord", "CR-V", "City", "Amaze"],
        "Ford": ["Mustang", "Explorer", "F-150", "EcoSport", "Endeavour"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
        "Hyundai": ["Elantra", "Sonata", "Tucson", "Creta", "i20", "Verna"],
        "Tata": ["Nexon", "Harrier", "Safari", "Altroz", "Tiago"],
        "Mahindra": ["Thar", "XUV700", "Scorpio", "Bolero"],
        "Maruti": ["Swift", "Baleno", "Brezza", "Dzire", "Ertiga"],
        "Audi": ["A4", "A6", "Q5", "Q7"],
        "Volkswagen": ["Polo", "Jetta", "Tiguan", "Virtus"],
        "Tesla": ["Model 3", "Model Y", "Model S"]
    }
    
    fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric", "CNG"]
    transmissions = ["Manual", "Automatic"]
    owner_types = ["First", "Second", "Third", "Fourth & Above"]
    state_codes = ["TN", "KA", "MH", "DL", "HR", "TS", "AP", "KL", "GJ", "UP", "WB"]
    
    data = []
    
    current_year = 2026
    
    for i in range(num_samples):
        brand = random.choice(list(brands_models.keys()))
        model = random.choice(brands_models[brand])
        year = random.randint(2010, 2025)
        age = current_year - year
        
        # Mileage correlated with age
        base_mileage = age * random.uniform(8000, 18000) + random.uniform(1000, 5000)
        mileage = int(max(1000, base_mileage))
        
        fuel = random.choice(fuel_types)
        if brand == "Tesla":
            fuel = "Electric"
            
        transmission = random.choice(transmissions)
        owner = random.choice(owner_types)
        
        # Engine capacity (cc)
        if fuel == "Electric":
            engine_cc = 0
        else:
            if brand in ["BMW", "Mercedes", "Audi"]:
                engine_cc = random.choice([1998, 2497, 2993, 3982])
            elif model in ["Mustang", "F-150", "Highlander", "Fortuner", "Endeavour", "XUV700"]:
                engine_cc = random.choice([2198, 2755, 3198, 4951])
            else:
                engine_cc = random.choice([999, 1197, 1497, 1598, 1799, 1999])
        
        # Base brand multiplier
        brand_tier = {
            "Tesla": 45000, "Mercedes": 48000, "BMW": 46000, "Audi": 44000,
            "Ford": 28000, "Toyota": 26000, "Mahindra": 22000, "Tata": 20000,
            "Volkswagen": 21000, "Hyundai": 19000, "Honda": 18500, "Maruti": 14000
        }
        
        base_price = brand_tier.get(brand, 20000)
        
        # Depreciate with age and mileage
        depreciation_factor = (0.88 ** age)
        mileage_penalty = max(0.4, 1.0 - (mileage / 300000))
        
        price = base_price * depreciation_factor * mileage_penalty * random.uniform(0.85, 1.15)
        price = round(max(2500, price), -2)  # Round to nearest 100
        
        # Registration number generation (e.g. TN 07 CA 4921)
        state = random.choice(state_codes)
        r_num = f"{state} {random.randint(1, 99):02d} {chr(random.randint(65, 90))}{chr(random.randint(65, 90))} {random.randint(1000, 9999)}"
        
        data.append({
            "registration_number": r_num,
            "brand": brand,
            "model": model,
            "year": year,
            "mileage": mileage,
            "fuel_type": fuel,
            "transmission": transmission,
            "engine_cc": engine_cc,
            "owner_type": owner,
            "price": price
        })
        
    df = pd.DataFrame(data)
    
    # Introduce controlled missing values (Phase 2 targets)
    df.loc[df.sample(frac=0.03, random_state=42).index, "mileage"] = np.nan
    df.loc[df.sample(frac=0.02, random_state=43).index, "fuel_type"] = np.nan
    df.loc[df.sample(frac=0.02, random_state=44).index, "engine_cc"] = np.nan
    
    # Introduce controlled duplicate rows
    duplicates = df.sample(n=35, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    df.to_csv(output_csv, index=False)
    print(f"[Phase 1] Vehicle dataset generated at '{output_csv}' with shape: {df.shape}")
    return df

def generate_sample_vehicle_images(output_dir="data/raw/sample_images", num_images=12):
    """
    Generates synthetic vehicle images with realistic number plates, bounding boxes,
    and annotations for YOLO detection and OCR testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_plates = [
        ("TN 01 AB 1234", "Toyota", "Fortuner", "White"),
        ("KA 05 CD 5678", "Honda", "Civic", "Blue"),
        ("MH 12 EF 9012", "BMW", "3 Series", "Black"),
        ("DL 03 GH 3456", "Hyundai", "Creta", "Red"),
        ("HR 26 JK 7890", "Mercedes", "C-Class", "Silver"),
        ("TS 09 LM 4321", "Tata", "Harrier", "Dark Grey"),
        ("AP 16 PQ 8765", "Mahindra", "Thar", "Yellow"),
        ("KL 07 RS 2109", "Maruti", "Swift", "White"),
        ("GJ 01 TU 6543", "Volkswagen", "Virtus", "Blue"),
        ("UP 32 VW 9876", "Audi", "A4", "Grey"),
        ("WB 02 XY 5432", "Ford", "Mustang", "Orange"),
        ("TN 09 ZA 1122", "Toyota", "Camry", "Silver")
    ]
    
    color_map = {
        "White": (240, 240, 240),
        "Blue": (30, 60, 150),
        "Black": (35, 35, 35),
        "Red": (180, 30, 30),
        "Silver": (192, 192, 192),
        "Dark Grey": (80, 80, 80),
        "Yellow": (230, 180, 20),
        "Grey": (120, 120, 120),
        "Orange": (230, 100, 20)
    }
    
    annotations = []
    
    for i, (plate_text, brand, model, color_name) in enumerate(sample_plates[:num_images]):
        img_w, img_h = 640, 480
        img = Image.new("RGB", (img_w, img_h), color=(220, 225, 230))
        draw = ImageDraw.Draw(img)
        
        # Background environment (road & scenery)
        draw.rectangle([0, 300, img_w, img_h], fill=(70, 70, 75)) # Road
        draw.line([0, 390, img_w, 390], fill=(240, 210, 50), width=4) # Lane marking
        
        # Vehicle body frame
        car_color = color_map.get(color_name, (100, 100, 100))
        v_left, v_top, v_right, v_bottom = 100, 160, 540, 380
        
        # Main chassis
        draw.rectangle([v_left, v_top + 60, v_right, v_bottom], fill=car_color, outline=(20, 20, 20), width=3)
        # Hood / Cabin top curve
        draw.polygon([(v_left + 70, v_top + 60), (v_left + 120, v_top), (v_right - 120, v_top), (v_right - 70, v_top + 60)], fill=(car_color[0]//2, car_color[1]//2, car_color[2]//2), outline=(20, 20, 20))
        # Windshield
        draw.polygon([(v_left + 80, v_top + 55), (v_left + 125, v_top + 10), (v_right - 125, v_top + 10), (v_right - 80, v_top + 55)], fill=(130, 180, 210, 200))
        
        # Headlights
        draw.ellipse([v_left + 15, v_top + 75, v_left + 65, v_top + 105], fill=(255, 255, 200), outline=(50, 50, 50))
        draw.ellipse([v_right - 65, v_top + 75, v_right - 15, v_top + 105], fill=(255, 255, 200), outline=(50, 50, 50))
        
        # Grille
        draw.rectangle([v_left + 100, v_top + 80, v_right - 100, v_top + 120], fill=(30, 30, 30))
        draw.text((v_left + 180, v_top + 90), f"{brand.upper()}", fill=(200, 200, 200))
        
        # Number plate region (Bumper area)
        plate_w, plate_h = 200, 45
        plate_left = (img_w - plate_w) // 2
        plate_top = v_bottom - 50
        plate_right = plate_left + plate_w
        plate_bottom = plate_top + plate_h
        
        # Outer plate border (Yellow or White plate)
        is_yellow = (i % 3 == 0) # Commercial / Taxi style or Standard white
        bg_plate_color = (255, 220, 0) if is_yellow else (255, 255, 255)
        text_plate_color = (0, 0, 0)
        
        draw.rectangle([plate_left - 3, plate_top - 3, plate_right + 3, plate_bottom + 3], fill=(20, 20, 20))
        draw.rectangle([plate_left, plate_top, plate_right, plate_bottom], fill=bg_plate_color)
        
        # Draw IND tag on left
        draw.rectangle([plate_left, plate_top, plate_left + 22, plate_bottom], fill=(0, 51, 153))
        draw.text((plate_left + 3, plate_top + 15), "IND", fill=(255, 255, 255))
        
        # Draw Plate text
        draw.text((plate_left + 30, plate_top + 12), plate_text, fill=text_plate_color)
        
        # Save image file
        img_filename = f"vehicle_{i+1:02d}_{brand}_{model.replace(' ', '_')}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)
        
        # Save bounding box metadata (x, y, w, h)
        annotations.append({
            "filename": img_filename,
            "vehicle_bbox": [v_left, v_top, v_right - v_left, v_bottom - v_top],
            "plate_bbox": [plate_left, plate_top, plate_w, plate_h],
            "registration_number": plate_text,
            "brand": brand,
            "model": model
        })
        
    import json
    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)
        
    print(f"[Phase 1] Generated {len(sample_plates[:num_images])} vehicle sample images in '{output_dir}'.")
    return annotations

if __name__ == "__main__":
    generate_vehicle_tabular_dataset()
    generate_sample_vehicle_images()
