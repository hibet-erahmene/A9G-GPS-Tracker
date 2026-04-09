from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
import os

app = Flask(__name__, static_folder='public', static_url_path='')

# Store both GPS and OBD2 data
gps_log = []
obd2_data = {"error": "waiting_for_data"}

@app.route("/")
def index():
    return send_from_directory('public', 'index.html')

@app.route("/gps", methods=["POST", "GET"])
def gps_handler():
    if request.method == "GET":
        return jsonify(gps_log)
    
    # POST handling for GPS
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
    
    # Keep only last 100 points
    if len(gps_log) > 100:
        gps_log.pop(0)
    
    print(f"[GPS] {entry}")
    return jsonify({"ok": True}), 200

@app.route("/obd2", methods=["POST", "GET"])
def obd2_handler():
    global obd2_data
    
    if request.method == "GET":
        return jsonify(obd2_data)
    
    # POST handling for OBD2
    try:
        data = request.get_json()
        if data:
            obd2_data = data
            obd2_data["last_update"] = datetime.utcnow().isoformat()
            print(f"[OBD2] {data}")
            return jsonify({"ok": True}), 200
    except:
        pass
    
    return jsonify({"error": "invalid data"}), 400

@app.route("/clear", methods=["POST"])
def clear_data():
    gps_log.clear()
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
