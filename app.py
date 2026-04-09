from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__, static_folder='public', static_url_path='')

gps_log = []
obd2_data = {"error": "waiting_for_data"}

# ── Load pipeline objects ─────────────────────────────────────────────────────
try:
    co2_model  = joblib.load("model/co2_model.pkl")
    co2_pca    = joblib.load("model/co2_pca.pkl")
    co2_scaler = joblib.load("model/co2_scaler.pkl")
    print("[ML] All pipeline objects loaded")
except Exception as e:
    co2_model = co2_pca = co2_scaler = None
    print(f"[ML] Failed to load: {e}")

# ── Exact column definitions (must match training) ────────────────────────────

DF_COLUMNS = [
    'Engine_Size', 'Cylinders',
    'Fuel_Consumption_City', 'Fuel_Consumption_Hwy',
    'Fuel_Consumption_Comb', 'Fuel_Consumption_Comb1',
    'Fuel_Type_E', 'Fuel_Type_N', 'Fuel_Type_X', 'Fuel_Type_Z',
    'Transmission_A4', 'Transmission_A5', 'Transmission_A6', 'Transmission_A7',
    'Transmission_A8', 'Transmission_A9', 'Transmission_AM5', 'Transmission_AM6',
    'Transmission_AM7', 'Transmission_AM8', 'Transmission_AM9', 'Transmission_AS10',
    'Transmission_AS4', 'Transmission_AS5', 'Transmission_AS6', 'Transmission_AS7',
    'Transmission_AS8', 'Transmission_AS9', 'Transmission_AV', 'Transmission_AV10',
    'Transmission_AV6', 'Transmission_AV7', 'Transmission_AV8',
    'Transmission_M5', 'Transmission_M6', 'Transmission_M7',
    'Make_Type_Luxury', 'Make_Type_Premium', 'Make_Type_Sports',
    'Vehicle_Class_Type_SUV', 'Vehicle_Class_Type_Sedan', 'Vehicle_Class_Type_Truck'
]

DUMMY_COLUMNS = [
    'Fuel_Type_E', 'Fuel_Type_N', 'Fuel_Type_X', 'Fuel_Type_Z',
    'Transmission_A4', 'Transmission_A5', 'Transmission_A6', 'Transmission_A7',
    'Transmission_A8', 'Transmission_A9', 'Transmission_AM5', 'Transmission_AM6',
    'Transmission_AM7', 'Transmission_AM8', 'Transmission_AM9', 'Transmission_AS10',
    'Transmission_AS4', 'Transmission_AS5', 'Transmission_AS6', 'Transmission_AS7',
    'Transmission_AS8', 'Transmission_AS9', 'Transmission_AV', 'Transmission_AV10',
    'Transmission_AV6', 'Transmission_AV7', 'Transmission_AV8',
    'Transmission_M5', 'Transmission_M6', 'Transmission_M7',
    'Make_Type_Luxury', 'Make_Type_Premium', 'Make_Type_Sports',
    'Vehicle_Class_Type_SUV', 'Vehicle_Class_Type_Sedan', 'Vehicle_Class_Type_Truck'
]

SELECTED_PCA = [
    'pca0','pca1','pca2','pca3','pca4','pca5','pca6','pca7',
    'pca8','pca9','pca10','pca11','pca12','pca13','pca14','pca15',
    'pca16','pca17','pca18','pca19','pca20','pca21',
    'pca23','pca24','pca25','pca26','pca27','pca28','pca29',
    'pca30','pca31','pca32','pca33'
]

X_TRAIN_COLUMNS = ['const'] + SELECTED_PCA

# ── MAF → fuel consumption ────────────────────────────────────────────────────
def maf_to_fuel(maf_g_per_s, speed_kmh):
    """MAF (g/s) + speed (km/h) → L/100km"""
    if not maf_g_per_s or not speed_kmh or float(speed_kmh) == 0:
        return None
    
    maf = float(maf_g_per_s)
    speed = float(speed_kmh)
    
    # Minimum realistic MAF check (prevents negative CO2)
    # A car needs roughly 10-15 g/s per 100 km/h
    min_maf_for_speed = speed * 0.12
    if maf < min_maf_for_speed:
        print(f"[WARNING] MAF too low: {maf}g/s at {speed}km/h, using minimum {min_maf_for_speed:.1f}")
        maf = min_maf_for_speed
    
    # Stoichiometric ratio for gasoline (14.7:1)
    fuel_g_per_s = maf / 14.7
    fuel_L_per_s = fuel_g_per_s / 740  # 740g/L for gasoline
    speed_m_per_s = speed / 3.6
    fuel_L_per_100km = (fuel_L_per_s / speed_m_per_s) * 100000
    
    # Ensure realistic range (4-25 L/100km for normal driving)
    if fuel_L_per_100km < 4:
        fuel_L_per_100km = 4
    if fuel_L_per_100km > 25:
        fuel_L_per_100km = 25
    
    return round(fuel_L_per_100km, 2)

# ── Full prediction pipeline ──────────────────────────────────────────────────
def predict_co2(engine_size, cylinders, fuel_comb,
                fuel_type='X', transmission='A6',
                make_type='General', vehicle_class='Sedan'):
    """
    Replicates the exact Colab pipeline:
    raw features → scaler → combine with one-hot → PCA → select → add const → model
    """
    if co2_model is None:
        # Simple fallback formula if model not loaded
        return round(fuel_comb * 23.2, 1)

    try:
        # Estimate city/hwy from combined (typical split ratios)
        fuel_city  = round(fuel_comb * 1.18, 1)   # city ~18% worse than combined
        fuel_hwy   = round(fuel_comb * 0.85, 1)   # hwy ~15% better than combined
        fuel_comb1 = round(235.214 / fuel_comb, 1) # L/100km → mpg conversion

        # Step 1: scale the 6 numerical features
        numerical = np.array([[
            engine_size, cylinders,
            fuel_city, fuel_hwy, fuel_comb, fuel_comb1
        ]])
        scaled = co2_scaler.transform(numerical)
        scaled_df = pd.DataFrame(scaled, columns=[
            'Engine_Size', 'Cylinders',
            'Fuel_Consumption_City', 'Fuel_Consumption_Hwy',
            'Fuel_Consumption_Comb', 'Fuel_Consumption_Comb1'
        ])

        # Step 2: build one-hot dummy row
        dummy_df = pd.DataFrame(0, index=[0], columns=DUMMY_COLUMNS)
        if f'Fuel_Type_{fuel_type}'          in dummy_df.columns:
            dummy_df[f'Fuel_Type_{fuel_type}'] = 1
        if f'Transmission_{transmission}'    in dummy_df.columns:
            dummy_df[f'Transmission_{transmission}'] = 1
        if f'Make_Type_{make_type}'          in dummy_df.columns:
            dummy_df[f'Make_Type_{make_type}'] = 1
        if f'Vehicle_Class_Type_{vehicle_class}' in dummy_df.columns:
            dummy_df[f'Vehicle_Class_Type_{vehicle_class}'] = 1

        # Step 3: combine and reorder to match df columns exactly
        combined = pd.concat([scaled_df, dummy_df], axis=1)
        combined = combined.reindex(columns=DF_COLUMNS, fill_value=0)

        # Step 4: PCA transform (42 components)
        pca_array  = co2_pca.transform(combined)
        pca_df     = pd.DataFrame(
            pca_array,
            columns=[f'pca{i}' for i in range(pca_array.shape[1])]
        )

        # Step 5: select only the features used in training (pca22 was dropped)
        pca_selected = pca_df[SELECTED_PCA]

        # Step 6: add constant term (statsmodels requirement)
        final = sm.add_constant(pca_selected, has_constant='add')
        final = final[X_TRAIN_COLUMNS]

        # Step 7: predict
        co2 = co2_model.predict(final)
        return round(float(co2[0]), 1)

    except Exception as e:
        print(f"[ML] Pipeline error: {e}")
        return round(fuel_comb * 23.2, 1)  # fallback

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory('public', 'index.html')

@app.route("/gps", methods=["POST", "GET"])
def gps_handler():
    if request.method == "GET":
        return jsonify(gps_log)

    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    lat = data.get("lat")
    lng = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "missing lat/lng"}), 400

    entry = {
        "lat": float(lat),
        "lng": float(lng),
        "time": datetime.utcnow().isoformat()
    }
    gps_log.append(entry)
    if len(gps_log) > 100:
        gps_log.pop(0)

    print(f"[GPS] {entry}")
    return jsonify({"ok": True}), 200

@app.route("/obd2", methods=["POST", "GET"])
def obd2_handler():
    global obd2_data

    if request.method == "GET":
        return jsonify(obd2_data)

    try:
        # Handle both JSON and form data (FIXED)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        if not data:
            return jsonify({"error": "invalid data"}), 400

        # Convert form data to dictionary with proper types
        obd2_data = {
            "rpm": int(data.get("rpm", 0)),
            "speed": int(data.get("speed", 0)),
            "coolant": int(data.get("coolant", 0)),
            "iat": int(data.get("iat", 0)),
            "load": float(data.get("load", 0)),
            "throttle": float(data.get("throttle", 0)),
            "map": int(data.get("map", 0)),
            "maf": float(data.get("maf", 0)),
            "fuel": float(data.get("fuel", 0)),
            "batt": float(data.get("batt", 0)),
            "last_update": datetime.utcnow().isoformat()
        }
        
        # MAF → fuel consumption
        fuel_L_per_100km = maf_to_fuel(
            obd2_data.get("maf"),
            obd2_data.get("speed")
        )
        obd2_data["fuel_consumption"] = fuel_L_per_100km

        # CO2 prediction
        co2 = None
        if fuel_L_per_100km:
            co2 = predict_co2(
                engine_size   = float(data.get("engine_size", 2.0)),
                cylinders     = int(data.get("cylinders", 4)),
                fuel_comb     = fuel_L_per_100km,
                fuel_type     = data.get("fuel_type", "X"),
                transmission  = data.get("transmission", "A6"),
                make_type     = data.get("make_type", "General"),
                vehicle_class = data.get("vehicle_class", "Sedan")
            )
        obd2_data["co2_g_per_km"] = co2

        print(f"[OBD2] RPM={obd2_data['rpm']} SPEED={obd2_data['speed']} MAF={obd2_data['maf']}")
        print(f"[CO2] Fuel={fuel_L_per_100km} L/100km → {co2} g/km")

        return jsonify({"ok": True}), 200

    except Exception as e:
        print(f"[OBD2] Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/clear", methods=["POST"])
def clear_data():
    gps_log.clear()
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
