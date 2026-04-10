from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone
import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math

app = Flask(__name__, static_folder='public', static_url_path='')

# ── Per-car state stores ──────────────────────────────────────────────────────
# Keyed by car_id string (e.g. "CAR-A3F9").
# Each entry is created on first contact from the device.

def make_car_state():
    return {
        "gps_log":   [],
        "obd2_data": {"error": "waiting_for_data"},
        "trip": {
            "active":           False,
            "start_time":       None,
            "fuel_samples":     [],
            "last_obd2_time":   None,
            "last_gps":         None,
            "gps_stable_since": None,
            "ended_trips":      [],
            "car":              {},
        }
    }

cars: dict[str, dict] = {}   # car_id → state dict

def get_car(car_id: str) -> dict:
    """Return (and auto-create) state for a given car_id."""
    if car_id not in cars:
        cars[car_id] = make_car_state()
        print(f"[CAR] New device registered: {car_id}")
    return cars[car_id]

# ── Load ML pipeline ──────────────────────────────────────────────────────────
try:
    co2_model  = joblib.load("model/co2_model.pkl")
    co2_pca    = joblib.load("model/co2_pca.pkl")
    co2_scaler = joblib.load("model/co2_scaler.pkl")
    print("[ML] Pipeline loaded OK")
except Exception as e:
    co2_model = co2_pca = co2_scaler = None
    print(f"[ML] Load failed: {e}")

# ── Column definitions ────────────────────────────────────────────────────────
DF_COLUMNS = [
    'Engine_Size','Cylinders',
    'Fuel_Consumption_City','Fuel_Consumption_Hwy',
    'Fuel_Consumption_Comb','Fuel_Consumption_Comb1',
    'Fuel_Type_E','Fuel_Type_N','Fuel_Type_X','Fuel_Type_Z',
    'Transmission_A4','Transmission_A5','Transmission_A6','Transmission_A7',
    'Transmission_A8','Transmission_A9','Transmission_AM5','Transmission_AM6',
    'Transmission_AM7','Transmission_AM8','Transmission_AM9','Transmission_AS10',
    'Transmission_AS4','Transmission_AS5','Transmission_AS6','Transmission_AS7',
    'Transmission_AS8','Transmission_AS9','Transmission_AV','Transmission_AV10',
    'Transmission_AV6','Transmission_AV7','Transmission_AV8',
    'Transmission_M5','Transmission_M6','Transmission_M7',
    'Make_Type_Luxury','Make_Type_Premium','Make_Type_Sports',
    'Vehicle_Class_Type_SUV','Vehicle_Class_Type_Sedan','Vehicle_Class_Type_Truck'
]
DUMMY_COLUMNS = [c for c in DF_COLUMNS if c not in [
    'Engine_Size','Cylinders','Fuel_Consumption_City','Fuel_Consumption_Hwy',
    'Fuel_Consumption_Comb','Fuel_Consumption_Comb1'
]]
SELECTED_PCA = [f'pca{i}' for i in range(22)] + [f'pca{i}' for i in range(23,34)]
X_TRAIN_COLS = ['const'] + SELECTED_PCA

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000
    p = math.pi / 180
    a = (math.sin((lat2-lat1)*p/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin((lng2-lng1)*p/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def gps_distance_km(points):
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_m(points[i-1]['lat'], points[i-1]['lng'],
                             points[i]['lat'],   points[i]['lng'])
    return round(total / 1000, 3)

def now_utc():
    return datetime.now(timezone.utc)

def maf_to_fuel_L100(maf_g_per_s, speed_kmh):
    if not maf_g_per_s or not speed_kmh or float(speed_kmh) == 0:
        return None
    maf   = max(float(maf_g_per_s), float(speed_kmh) * 0.12)
    speed = float(speed_kmh)
    L_per_s     = (maf / 14.7) / 740
    m_per_s     = speed / 3.6
    L_per_100km = (L_per_s / m_per_s) * 100000
    return round(max(4.0, min(25.0, L_per_100km)), 2)

def maf_to_L_per_s(maf_g_per_s):
    if not maf_g_per_s or float(maf_g_per_s) <= 0:
        return 0.0
    return float(maf_g_per_s) / 14.7 / 740

def predict_co2(engine_size, cylinders, fuel_comb,
                fuel_type='X', transmission='A6',
                make_type='General', vehicle_class='Sedan'):
    if co2_model is None:
        return round(float(fuel_comb) * 23.2, 1)
    try:
        fc = float(fuel_comb)
        numerical = np.array([[
            engine_size, cylinders,
            round(fc*1.18,1), round(fc*0.85,1),
            fc, round(235.214/fc, 1)
        ]])
        scaled    = co2_scaler.transform(numerical)
        scaled_df = pd.DataFrame(scaled, columns=[
            'Engine_Size','Cylinders',
            'Fuel_Consumption_City','Fuel_Consumption_Hwy',
            'Fuel_Consumption_Comb','Fuel_Consumption_Comb1'
        ])
        dummy_df = pd.DataFrame(0, index=[0], columns=DUMMY_COLUMNS)
        for col in [f'Fuel_Type_{fuel_type}', f'Transmission_{transmission}',
                    f'Make_Type_{make_type}', f'Vehicle_Class_Type_{vehicle_class}']:
            if col in dummy_df.columns:
                dummy_df[col] = 1
        combined = pd.concat([scaled_df, dummy_df], axis=1)
        combined = combined.reindex(columns=DF_COLUMNS, fill_value=0)
        pca_arr  = co2_pca.transform(combined)
        pca_df   = pd.DataFrame(pca_arr, columns=[f'pca{i}' for i in range(pca_arr.shape[1])])
        final    = sm.add_constant(pca_df[SELECTED_PCA], has_constant='add')
        return round(float(co2_model.predict(final[X_TRAIN_COLS])[0]), 1)
    except Exception as e:
        print(f"[ML] {e}")
        return round(float(fuel_comb) * 23.2, 1)

def finalize_trip(state: dict, car_id: str):
    trip    = state["trip"]
    gps_log = state["gps_log"]

    if not trip["active"] or not trip["fuel_samples"]:
        trip["active"] = False
        return

    total_fuel_L = round(sum(r * i for r, i in trip["fuel_samples"]), 3)
    distance_km  = gps_distance_km(gps_log) if len(gps_log) >= 2 else 0.0
    avg_L100     = round(total_fuel_L / distance_km * 100, 2) if distance_km > 0 else None
    duration_s   = (now_utc() - datetime.fromisoformat(trip["start_time"])).total_seconds()

    car = trip["car"]
    co2_g_per_km = predict_co2(
        float(car.get("engine_size", 2.0)), int(car.get("cylinders", 4)),
        avg_L100 or 9.0,
        car.get("fuel_type", "X"), car.get("transmission", "A6"),
        car.get("make_type", "General"), car.get("vehicle_class", "Sedan"),
    ) if avg_L100 else None

    if co2_g_per_km and distance_km:
        co2_total_g = round(co2_g_per_km * distance_km * 1000, 1)
        method      = "ML model"
    else:
        co2_total_g = round(total_fuel_L * 2392, 1)
        method      = "combustion formula"

    summary = {
        "id":           int(now_utc().timestamp()),
        "car_id":       car_id,
        "start_time":   trip["start_time"],
        "end_time":     now_utc().isoformat(),
        "duration_min": round(duration_s / 60, 1),
        "distance_km":  distance_km,
        "total_fuel_L": total_fuel_L,
        "avg_L100km":   avg_L100,
        "co2_g_per_km": co2_g_per_km,
        "co2_total_g":  co2_total_g,
        "co2_total_kg": round(co2_total_g / 1000, 3),
        "method":       method,
        "car_make":     car.get("make", ""),
        "car_model":    car.get("model", ""),
        "car_nickname": car.get("nickname", ""),
        "gps_points":   len(gps_log),
    }
    trip["ended_trips"].insert(0, summary)
    if len(trip["ended_trips"]) > 20:
        trip["ended_trips"].pop()

    print(f"[TRIP] [{car_id}] Done — {distance_km}km | {total_fuel_L}L | "
          f"{summary['co2_total_kg']}kg CO2 ({method})")

    trip.update({
        "active": False, "start_time": None,
        "fuel_samples": [], "last_obd2_time": None,
        "last_gps": None, "gps_stable_since": None,
    })

def check_trip_end(state: dict, car_id: str):
    trip = state["trip"]
    if not trip["active"]:
        return
    now   = now_utc()
    LIMIT = 180

    obd2_silent = (trip["last_obd2_time"] and
        (now - datetime.fromisoformat(trip["last_obd2_time"])).total_seconds() > LIMIT)
    gps_stable  = (trip["gps_stable_since"] and
        (now - datetime.fromisoformat(trip["gps_stable_since"])).total_seconds() > LIMIT)

    if obd2_silent and gps_stable:
        print(f"[TRIP] [{car_id}] End conditions met")
        finalize_trip(state, car_id)

# ── Helper to extract car_id from request ────────────────────────────────────
def car_id_from_request(req) -> str:
    """Pull car_id from POST body or GET query param. Default to 'default'."""
    if req.method == "POST":
        data = req.get_json(silent=True) or req.form
        cid  = (data.get("car_id") or "").strip()
    else:
        cid = req.args.get("car_id", "").strip()
    return cid if cid else "default"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return send_from_directory('public', 'index.html')

@app.route("/garage")
def garage_page():
    return send_from_directory('public', 'garage.html')

@app.route("/cars_dataset.json")
def cars_dataset():
    return send_from_directory('public', 'cars_dataset.json')

# List all known car_ids (for the garage page to discover paired devices)
@app.route("/devices", methods=["GET"])
def list_devices():
    return jsonify(list(cars.keys()))

@app.route("/gps", methods=["GET","POST"])
def gps_handler():
    car_id = car_id_from_request(request)
    state  = get_car(car_id)

    if request.method == "GET":
        check_trip_end(state, car_id)
        return jsonify(state["gps_log"])

    data = request.get_json(silent=True) or request.form
    lat, lng = data.get("lat"), data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "missing lat/lng"}), 400

    lat, lng = float(lat), float(lng)
    now_s    = now_utc().isoformat()
    entry    = {"lat": lat, "lng": lng, "time": now_s}
    state["gps_log"].append(entry)
    if len(state["gps_log"]) > 500:
        state["gps_log"].pop(0)

    trip = state["trip"]
    if not trip["active"] and not state["obd2_data"].get("error"):
        trip.update({
            "active": True, "start_time": now_s,
            "last_gps": {"lat": lat, "lng": lng},
            "gps_stable_since": now_s,
        })
        print(f"[TRIP] [{car_id}] Started")

    if trip["active"]:
        last = trip.get("last_gps")
        if last and haversine_m(last["lat"], last["lng"], lat, lng) > 20:
            trip["gps_stable_since"] = None
            trip["last_gps"]         = {"lat": lat, "lng": lng}
        elif trip["gps_stable_since"] is None:
            trip["gps_stable_since"] = now_s

    return jsonify({"ok": True, "car_id": car_id}), 200


@app.route("/obd2", methods=["GET","POST"])
def obd2_handler():
    car_id = car_id_from_request(request)
    state  = get_car(car_id)

    if request.method == "GET":
        result = dict(state["obd2_data"])
        fuel   = result.get("fuel_consumption")
        if fuel and not result.get("error"):
            result["co2_g_per_km"] = predict_co2(
                float(request.args.get("engine_size", 2.0)),
                int(request.args.get("cylinders", 4)),
                fuel,
                request.args.get("fuel_type", "X"),
                request.args.get("transmission", "A6"),
                request.args.get("make_type", "General"),
                request.args.get("vehicle_class", "Sedan"),
            )
        result["car_id"] = car_id
        return jsonify(result)

    try:
        data = request.get_json(silent=True) or request.form
        if not data:
            return jsonify({"error": "invalid data"}), 400

        now_s    = now_utc().isoformat()
        trip     = state["trip"]
        obd2     = state["obd2_data"]

        interval_s = 0.0
        if trip["last_obd2_time"]:
            interval_s = min(60.0, (
                datetime.fromisoformat(now_s) -
                datetime.fromisoformat(trip["last_obd2_time"])
            ).total_seconds())
        trip["last_obd2_time"] = now_s

        state["obd2_data"] = {
            "rpm":      int(float(data.get("rpm",      0))),
            "speed":    int(float(data.get("speed",    0))),
            "coolant":  int(float(data.get("coolant",  0))),
            "iat":      int(float(data.get("iat",      0))),
            "load":     round(float(data.get("load",   0)), 1),
            "throttle": round(float(data.get("throttle", 0)), 1),
            "map":      int(float(data.get("map",      0))),
            "maf":      round(float(data.get("maf",    0)), 2),
            "fuel":     round(float(data.get("fuel",   0)), 1),
            "batt":     round(float(data.get("batt",   0)), 2),
            "last_update": now_s,
            "car_id":   car_id,
        }
        obd2 = state["obd2_data"]

        fuel_L100 = maf_to_fuel_L100(obd2["maf"], obd2["speed"])
        obd2["fuel_consumption"] = fuel_L100

        car      = trip["car"]
        co2_live = predict_co2(
            float(data.get("engine_size",  car.get("engine_size",  2.0))),
            int(data.get("cylinders",      car.get("cylinders",    4))),
            fuel_L100 or 9.0,
            data.get("fuel_type",          car.get("fuel_type",    "X")),
            data.get("transmission",       car.get("transmission", "A6")),
            data.get("make_type",          car.get("make_type",    "General")),
            data.get("vehicle_class",      car.get("vehicle_class","Sedan")),
        ) if fuel_L100 else None
        obd2["co2_g_per_km"] = co2_live

        if trip["active"] and interval_s > 0:
            trip["fuel_samples"].append((maf_to_L_per_s(obd2["maf"]), interval_s))

        print(f"[OBD2] [{car_id}] spd={obd2['speed']} maf={obd2['maf']} co2={co2_live}")
        return jsonify({"ok": True, "car_id": car_id,
                        "co2_g_per_km": co2_live, "fuel_L100": fuel_L100}), 200

    except Exception as e:
        print(f"[OBD2] [{car_id}] {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/trip/config", methods=["POST"])
def trip_config():
    data   = request.get_json(force=True)
    car_id = (data or {}).get("car_id", "default")
    state  = get_car(car_id)
    if data:
        state["trip"]["car"] = data
        print(f"[TRIP] [{car_id}] Car: {data.get('make')} {data.get('model')}")
    return jsonify({"ok": True, "car_id": car_id})


@app.route("/trip/status", methods=["GET"])
def trip_status():
    car_id = request.args.get("car_id", "default")
    state  = get_car(car_id)
    check_trip_end(state, car_id)

    trip        = state["trip"]
    gps_log     = state["gps_log"]
    dist        = gps_distance_km(gps_log) if len(gps_log) >= 2 else 0.0
    fuel_so_far = round(sum(r*i for r, i in trip["fuel_samples"]), 3)
    dur_s       = 0
    if trip["active"] and trip["start_time"]:
        dur_s = (now_utc() - datetime.fromisoformat(trip["start_time"])).total_seconds()

    return jsonify({
        "car_id":        car_id,
        "active":        trip["active"],
        "start_time":    trip["start_time"],
        "duration_min":  round(dur_s / 60, 1),
        "distance_km":   dist,
        "fuel_so_far_L": fuel_so_far,
        "samples":       len(trip["fuel_samples"]),
        "ended_trips":   trip["ended_trips"],
    })


@app.route("/clear", methods=["POST"])
def clear_data():
    car_id = car_id_from_request(request)
    if car_id in cars:
        cars[car_id]["gps_log"].clear()
    return jsonify({"ok": True, "car_id": car_id}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

