from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
gps_log = []

@app.route("/gps", methods=["POST"])
def receive_gps():
    data = request.get_json(force=True, silent=True) or {}
    lat  = data.get("lat")
    lng  = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "missing lat/lng"}), 400
    entry = {"lat": lat, "lng": lng, "time": datetime.utcnow().isoformat()}
    gps_log.append(entry)
    print(f"[GPS] {entry}")
    return jsonify({"ok": True}), 200

@app.route("/gps", methods=["GET"])
def show_log():
    return jsonify(gps_log)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
