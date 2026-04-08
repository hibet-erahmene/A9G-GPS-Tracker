from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import json

app = Flask(__name__)

# Store both GPS and OBD2 data
gps_log = []
obd2_data = {"error": "waiting_for_data"}

# HTML template with girly dashboard
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌸 GPS + OBD2 Tracker 🌸</title>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Dancing+Script:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #ffe9f4 0%, #ffe0f0 50%, #ffd6ea 100%);
            font-family: 'Quicksand', sans-serif;
            min-height: 100vh;
            padding: 20px;
        }

        /* Cute floating hearts animation */
        @keyframes floatHeart {
            0% { transform: translateY(100vh) rotate(0deg); opacity: 1; }
            100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
        }

        .heart {
            position: fixed;
            color: #ff9ec0;
            font-size: 20px;
            pointer-events: none;
            z-index: 999;
            animation: floatHeart linear forwards;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 30px;
            position: relative;
        }

        .header h1 {
            font-family: 'Dancing Script', cursive;
            font-size: 3rem;
            color: #ff6b9d;
            text-shadow: 3px 3px 0 #ffd4e8;
            margin-bottom: 5px;
        }

        .header p {
            color: #d47a9e;
            font-size: 0.9rem;
            letter-spacing: 2px;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 10px;
        }

        .status-online {
            background: #ff9ec0;
            color: white;
            box-shadow: 0 0 10px #ff9ec0;
        }

        .status-offline {
            background: #d4a5b8;
            color: #fff0f5;
        }

        /* Grid layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }

        /* Cards */
        .card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 25px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(255, 105, 180, 0.15);
            border: 1px solid rgba(255, 182, 193, 0.5);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h3 {
            color: #ff6b9d;
            font-size: 1.2rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 2px solid #ffc0d4;
            padding-bottom: 8px;
        }

        /* Map */
        #map {
            height: 400px;
            border-radius: 20px;
            overflow: hidden;
            border: 3px solid white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        /* OBD2 metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .metric {
            background: white;
            border-radius: 18px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(255, 105, 180, 0.1);
            transition: all 0.3s;
        }

        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #d47a9e;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ff6b9d;
            font-family: monospace;
        }

        .metric-unit {
            font-size: 0.7rem;
            color: #b8a0b0;
            margin-left: 3px;
        }

        .metric-bar {
            width: 100%;
            height: 4px;
            background: #ffe0f0;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }

        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff9ec0, #ff6b9d);
            border-radius: 2px;
            transition: width 0.5s ease;
            width: 0%;
        }

        /* GPS info panel */
        .gps-info {
            margin-top: 15px;
            background: white;
            border-radius: 18px;
            padding: 12px;
            font-size: 0.8rem;
        }

        .gps-info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #ffe0f0;
        }

        .gps-info-label {
            color: #d47a9e;
            font-weight: 600;
        }

        .gps-info-value {
            color: #ff6b9d;
            font-family: monospace;
        }

        /* Trajectory list */
        .trajectory-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 15px;
        }

        .trajectory-item {
            background: white;
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 8px;
            font-size: 0.7rem;
            font-family: monospace;
            color: #d47a9e;
            border-left: 3px solid #ff9ec0;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            color: #d47a9e;
            font-size: 0.7rem;
        }

        /* Scrollbar */
        .trajectory-list::-webkit-scrollbar {
            width: 6px;
        }

        .trajectory-list::-webkit-scrollbar-track {
            background: #ffe0f0;
            border-radius: 3px;
        }

        .trajectory-list::-webkit-scrollbar-thumb {
            background: #ff9ec0;
            border-radius: 3px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2rem;
            }
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌸 GPS + OBD2 Tracker 🌸</h1>
            <p>your cute little car companion</p>
            <div class="status-badge" id="statusBadge">LOADING...</div>
        </div>

        <div class="dashboard-grid">
            <!-- Left column: Map -->
            <div class="card">
                <h3>Live Location</h3>
                <div id="map"></div>
                <div class="gps-info" id="gpsInfo">
                    <div class="gps-info-row">
                        <span class="gps-info-label">Latest Position:</span>
                        <span class="gps-info-value" id="latestLat">--</span>
                    </div>
                    <div class="gps-info-row">
                        <span class="gps-info-label">Longitude:</span>
                        <span class="gps-info-value" id="latestLng">--</span>
                    </div>
                    <div class="gps-info-row">
                        <span class="gps-info-label">Last Update:</span>
                        <span class="gps-info-value" id="latestTime">--</span>
                    </div>
                    <div class="gps-info-row">
                        <span class="gps-info-label">Total Points:</span>
                        <span class="gps-info-value" id="totalPoints">0</span>
                    </div>
                </div>
            </div>

            <!-- Right column: OBD2 Data -->
            <div class="card">
                <h3>Engine Data</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">RPM</div>
                        <div class="metric-value" id="rpm">--</div>
                        <div class="metric-bar"><div class="metric-bar-fill" id="rpmBar"></div></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Speed</div>
                        <div class="metric-value" id="speed">-- <span class="metric-unit">km/h</span></div>
                        <div class="metric-bar"><div class="metric-bar-fill" id="speedBar"></div></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Coolant</div>
                        <div class="metric-value" id="coolant">-- <span class="metric-unit">°C</span></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Engine Load</div>
                        <div class="metric-value" id="load">-- <span class="metric-unit">%</span></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Throttle</div>
                        <div class="metric-value" id="throttle">-- <span class="metric-unit">%</span></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Battery</div>
                        <div class="metric-value" id="battery">-- <span class="metric-unit">V</span></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Fuel Level</div>
                        <div class="metric-value" id="fuel">-- <span class="metric-unit">%</span></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">MAF</div>
                        <div class="metric-value" id="maf">-- <span class="metric-unit">g/s</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Trajectory History -->
        <div class="card">
            <h3>Journey History</h3>
            <div class="trajectory-list" id="trajectoryList">
                <div style="text-align: center; color: #d47a9e;">Waiting for GPS data...</div>
            </div>
        </div>

        <div class="footer">
            💕 made with love • real-time tracking • your car is happy 💕
        </div>
    </div>

    <script>
        let map;
        let marker = null;
        let path = [];
        let polyline = null;

        // Initialize map
        function initMap() {
            map = L.map('map').setView([36.897028, 7.756056], 14);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> & CartoDB',
                subdomains: 'abcd',
                maxZoom: 19
            }).addTo(map);
        }

        // Fetch GPS data
        async function fetchGPS() {
            try {
                const response = await fetch('/gps');
                const data = await response.json();
                
                if (data.length > 0) {
                    const latest = data[data.length - 1];
                    document.getElementById('latestLat').textContent = latest.lat.toFixed(6);
                    document.getElementById('latestLng').textContent = latest.lng.toFixed(6);
                    document.getElementById('latestTime').textContent = new Date(latest.time).toLocaleString();
                    document.getElementById('totalPoints').textContent = data.length;
                    
                    // Update map
                    if (marker === null) {
                        marker = L.marker([latest.lat, latest.lng]).addTo(map);
                        marker.bindPopup('<b>Current Location</b><br>🌸 You are here!');
                    } else {
                        marker.setLatLng([latest.lat, latest.lng]);
                    }
                    
                    // Draw path
                    path = data.map(p => [p.lat, p.lng]);
                    if (polyline !== null) map.removeLayer(polyline);
                    polyline = L.polyline(path, {color: '#ff6b9d', weight: 3, opacity: 0.7}).addTo(map);
                    map.setView([latest.lat, latest.lng], 15);
                    
                    // Update trajectory list (show last 10)
                    const trajectoryList = document.getElementById('trajectoryList');
                    const last10 = data.slice(-10).reverse();
                    trajectoryList.innerHTML = last10.map(p => `
                        <div class="trajectory-item">
                             ${p.lat.toFixed(6)}, ${p.lng.toFixed(6)} 
                            <span style="float: right;"> ${new Date(p.time).toLocaleTimeString()}</span>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('GPS fetch error:', error);
            }
        }

        // Fetch OBD2 data
        async function fetchOBD2() {
            try {
                const response = await fetch('/obd2');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('statusBadge').textContent = 'WAITING FOR OBD2...';
                    document.getElementById('statusBadge').className = 'status-badge status-offline';
                    return;
                }
                
                document.getElementById('statusBadge').textContent = 'LIVE TRACKING';
                document.getElementById('statusBadge').className = 'status-badge status-online';
                
                // Update metrics
                document.getElementById('rpm').textContent = data.rpm || '--';
                document.getElementById('speed').innerHTML = (data.speed || '--') + ' <span class="metric-unit">km/h</span>';
                document.getElementById('coolant').innerHTML = (data.coolant || '--') + ' <span class="metric-unit">°C</span>';
                document.getElementById('load').innerHTML = (data.load || '--') + ' <span class="metric-unit">%</span>';
                document.getElementById('throttle').innerHTML = (data.throttle || '--') + ' <span class="metric-unit">%</span>';
                document.getElementById('battery').innerHTML = (data.batt || '--') + ' <span class="metric-unit">V</span>';
                document.getElementById('fuel').innerHTML = (data.fuel || '--') + ' <span class="metric-unit">%</span>';
                document.getElementById('maf').innerHTML = (data.maf || '--') + ' <span class="metric-unit">g/s</span>';
                
                // Update bars
                const rpmPct = Math.min(100, (data.rpm || 0) / 80);
                document.getElementById('rpmBar').style.width = rpmPct + '%';
                const speedPct = Math.min(100, (data.speed || 0) / 2);
                document.getElementById('speedBar').style.width = speedPct + '%';
                
            } catch (error) {
                console.error('OBD2 fetch error:', error);
            }
        }

        // Create floating hearts
        function createHeart() {
            const heart = document.createElement('div');
            heart.innerHTML = ['🌸', '💕', '💖', '💗', '💓', '🌸'][Math.floor(Math.random() * 6)];
            heart.className = 'heart';
            heart.style.left = Math.random() * 100 + '%';
            heart.style.fontSize = (Math.random() * 20 + 10) + 'px';
            heart.style.animationDuration = Math.random() * 8 + 5 + 's';
            heart.style.animationDelay = Math.random() * 5 + 's';
            document.body.appendChild(heart);
            setTimeout(() => heart.remove(), 13000);
        }

        // Start everything
        initMap();
        fetchGPS();
        fetchOBD2();
        setInterval(fetchGPS, 3000);
        setInterval(fetchOBD2, 2000);
        setInterval(createHeart, 3000);
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

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
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
