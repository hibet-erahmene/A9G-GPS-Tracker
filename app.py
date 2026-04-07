from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)
gps_log = []

# HTML template for the map
MAP_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Live GPS Tracker</title>
    <meta charset="utf-8" />
    <style>
        #map { height: 600px; width: 100%; }
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .info { margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
        .controls { margin-bottom: 10px; }
        button { padding: 5px 10px; margin-right: 10px; cursor: pointer; }
    </style>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
    <h1>Live GPS Tracker - A9G Device</h1>
    <div class="controls">
        <button onclick="location.reload()">Refresh Map</button>
        <button onclick="clearData()">Clear All Data</button>
    </div>
    <div id="map"></div>
    <div class="info" id="info">Loading data...</div>

    <script>
        var map = L.map('map').setView([36.123455, 7.456789], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        var marker = null;
        var path = [];
        var polyline = null;

        function fetchGPSData() {
            fetch('/gps')
                .then(response => response.json())
                .then(data => {
                    if (data.length === 0) {
                        document.getElementById('info').innerHTML = 'No GPS data yet. Waiting for A9G to send data...';
                        return;
                    }

                    var latest = data[data.length - 1];
                    var lat = latest.lat;
                    var lng = latest.lng;
                    var time = latest.time;

                    document.getElementById('info').innerHTML = `
                        <strong>Live Tracking:</strong><br>
                        Latitude: ${lat.toFixed(6)}<br>
                        Longitude: ${lng.toFixed(6)}<br>
                        Time: ${new Date(time).toLocaleString()}<br>
                        Total Points: ${data.length}
                    `;

                    if (marker === null) {
                        marker = L.marker([lat, lng]).addTo(map);
                        marker.bindPopup(`<b>Current Location</b><br>Lat: ${lat}<br>Lng: ${lng}`).openPopup();
                    } else {
                        marker.setLatLng([lat, lng]);
                        marker.getPopup().setContent(`<b>Current Location</b><br>Lat: ${lat}<br>Lng: ${lng}`);
                    }

                    path = data.map(p => [p.lat, p.lng]);
                    if (polyline !== null) {
                        map.removeLayer(polyline);
                    }
                    polyline = L.polyline(path, {color: '#3388ff', weight: 3, opacity: 0.7}).addTo(map);
                    map.setView([lat, lng], 15);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('info').innerHTML = 'Error loading data: ' + error;
                });
        }

        function clearData() {
            fetch('/clear', { method: 'POST' })
                .then(() => {
                    location.reload();
                });
        }

        fetchGPSData();
        setInterval(fetchGPSData, 3000);
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(MAP_TEMPLATE)

@app.route("/gps", methods=["POST", "GET"])
def gps_handler():
    if request.method == "GET":
        return jsonify(gps_log)
    
    # POST handling
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

@app.route("/clear", methods=["POST"])
def clear_data():
    gps_log.clear()
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
