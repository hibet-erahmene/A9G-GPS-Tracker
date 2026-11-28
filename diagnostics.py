"""
Vehicle diagnostics: fault codes (DTC), telemetry storage, rule-based alerts
and a per-car Isolation Forest that learns what "normal" looks like.

Uses the same database as auth.py (Postgres on Render, SQLite locally).
Tables are created automatically by init_diagnostics().
"""
import os, re, threading, time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index, func
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from auth import Base, engine, DB

# ── Settings (override with environment variables on Render) ──────────────────
STORE_EVERY_S   = float(os.environ.get("STORE_EVERY_S", "5"))       # at most 1 stored sample per car per N seconds
RETENTION_DAYS  = int(os.environ.get("RETENTION_DAYS", "30"))       # old telemetry is deleted after this
MIN_TRAIN_ROWS  = int(os.environ.get("MIN_TRAIN_ROWS", "1500"))     # ~2 h of driving at 5 s per sample
MAX_TRAIN_ROWS  = int(os.environ.get("MAX_TRAIN_ROWS", "20000"))
RETRAIN_AFTER_H = float(os.environ.get("RETRAIN_AFTER_H", "24"))
ANOM_PCT        = float(os.environ.get("ANOM_PCT", "0.5"))          # training-score percentile used as the alarm threshold
ANOM_WINDOW     = 12                                                 # last N scored samples (~60 s)
ANOM_MIN_FLAGS  = 8                                                  # N flagged out of the window = "unusual behaviour"
ALERT_COOLDOWN_S = int(os.environ.get("ALERT_COOLDOWN_S", "1800"))  # same alert at most every 30 min

FEATURES = ["rpm", "speed", "load", "throttle", "coolant", "map", "maf", "batt"]


# ── Tables ────────────────────────────────────────────────────────────────────
class Telemetry(Base):
    __tablename__ = "telemetry"
    id = Column(Integer, primary_key=True)
    device_id = Column(String(40), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    rpm = Column(Float); speed = Column(Float); coolant = Column(Float); iat = Column(Float)
    load = Column(Float); throttle = Column(Float); map = Column(Float); maf = Column(Float)
    fuel = Column(Float); batt = Column(Float)
    __table_args__ = (Index("ix_telemetry_dev_ts", "device_id", "ts"),)


class DtcEvent(Base):
    __tablename__ = "dtc_events"
    id = Column(Integer, primary_key=True)
    device_id = Column(String(40), nullable=False, index=True)
    code = Column(String(8), nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)
    active = Column(Boolean, default=True, nullable=False)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    id = Column(Integer, primary_key=True)
    device_id = Column(String(40), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    kind = Column(String(40), nullable=False)
    severity = Column(String(10), nullable=False)      # info | warning | critical
    message = Column(String(300), nullable=False)
    __table_args__ = (Index("ix_alert_dev_ts", "device_id", "ts"),)


def init_diagnostics():
    Base.metadata.create_all(engine)


def _now():
    return datetime.now(timezone.utc)


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def _iso(dt):
    dt = _aware(dt)
    return dt.isoformat() if dt else None


# ── DTC decoding ──────────────────────────────────────────────────────────────
DTC_INFO = {
    "P0016": "Crankshaft / camshaft position correlation",
    "P0087": "Fuel rail pressure too low",
    "P0088": "Fuel rail pressure too high",
    "P0093": "Fuel system leak (large)",
    "P0100": "Mass air flow (MAF) circuit malfunction",
    "P0101": "Mass air flow (MAF) range / performance",
    "P0102": "Mass air flow (MAF) circuit low input",
    "P0103": "Mass air flow (MAF) circuit high input",
    "P0105": "Manifold pressure (MAP) circuit malfunction",
    "P0110": "Intake air temperature circuit malfunction",
    "P0113": "Intake air temperature circuit high input",
    "P0115": "Coolant temperature circuit malfunction",
    "P0117": "Coolant temperature circuit low input",
    "P0118": "Coolant temperature circuit high input",
    "P0120": "Throttle / pedal position sensor circuit malfunction",
    "P0128": "Coolant temperature below thermostat regulating temperature",
    "P0130": "Oxygen sensor circuit malfunction (bank 1, sensor 1)",
    "P0133": "Oxygen sensor slow response (bank 1, sensor 1)",
    "P0171": "System too lean (bank 1)",
    "P0172": "System too rich (bank 1)",
    "P0174": "System too lean (bank 2)",
    "P0175": "System too rich (bank 2)",
    "P0201": "Injector circuit malfunction, cylinder 1",
    "P0202": "Injector circuit malfunction, cylinder 2",
    "P0203": "Injector circuit malfunction, cylinder 3",
    "P0204": "Injector circuit malfunction, cylinder 4",
    "P0217": "Engine overtemperature condition",
    "P0234": "Turbocharger overboost",
    "P0299": "Turbocharger underboost",
    "P0300": "Random / multiple cylinder misfire detected",
    "P0301": "Cylinder 1 misfire detected",
    "P0302": "Cylinder 2 misfire detected",
    "P0303": "Cylinder 3 misfire detected",
    "P0304": "Cylinder 4 misfire detected",
    "P0325": "Knock sensor circuit malfunction",
    "P0335": "Crankshaft position sensor circuit malfunction",
    "P0340": "Camshaft position sensor circuit malfunction",
    "P0380": "Glow plug circuit malfunction",
    "P0400": "Exhaust gas recirculation (EGR) flow malfunction",
    "P0401": "EGR flow insufficient",
    "P0402": "EGR flow excessive",
    "P0403": "EGR circuit malfunction",
    "P0420": "Catalyst efficiency below threshold (bank 1)",
    "P0440": "Evaporative emission (EVAP) system malfunction",
    "P0442": "EVAP system small leak detected",
    "P0455": "EVAP system large leak detected",
    "P0500": "Vehicle speed sensor malfunction",
    "P0505": "Idle control system malfunction",
    "P0562": "System voltage low",
    "P0563": "System voltage high",
    "P0600": "Serial communication link malfunction",
    "P0670": "Glow plug control module malfunction",
    "P0700": "Transmission control system malfunction",
}
DTC_RE = re.compile(r"^[PCBU][0-3][0-9A-F]{3}$")


def describe_dtc(code):
    return DTC_INFO.get(code, "No description stored for this code. Look it up by the code and your car make.")


def _decode_pairs(payload):
    codes = []
    for i in range(0, len(payload) - 3, 4):
        w = payload[i:i + 4]
        if w == "0000":
            continue
        first = int(w[0], 16)
        letter = "PCBU"[first >> 2]
        code = f"{letter}{first & 3}{w[1:]}"
        if DTC_RE.match(code):
            codes.append(code)
    return codes


CAN_PROTOCOLS = set("6789ABC")      # ELM327 protocol numbers (ATDPN) that use CAN


def parse_dtc_response(raw, proto=None):
    """
    Decode the ELM327 reply to Mode 03. The ESP32 sends it as hex characters,
    ':' and ',' (line breaks became commas), plus the ELM protocol number (ATDPN).

    CAN replies: 43 <count> <code><code>...   (padding after the codes is ignored)
    K-line / J1850 replies: 43 <code><code><code>  (padded with 0000)

    Returns a list of codes (possibly empty) if a valid Mode 03 reply was found,
    or None if the reply was unusable (NO DATA, garbage, ...).
    """
    if not raw:
        return None
    p = str(proto).strip().upper()[-1:] if proto not in (None, "") else ""
    is_can = (p in CAN_PROTOCOLS) if p else None
    lines = [l for l in re.split(r"[,|\r\n ]+", str(raw).upper()) if l]
    framed = [l.split(":", 1)[1] for l in lines if re.match(r"^[0-9A-F]:", l)]   # multi-frame replies are always CAN
    streams = []                                                                  # (hex string, forced_can)
    if framed:
        streams.append(("".join(framed), True))
    streams += [(l, is_can) for l in lines if ":" not in l and l.startswith("43")]
    found, codes = False, []
    for s, can in streams:
        s = re.sub(r"[^0-9A-F]", "", s)
        if not s.startswith("43"):
            continue
        found = True
        payload = s[2:]
        if can is None:
            can = len(payload) % 4 == 2          # unknown protocol: guess from the length
        if can:
            if len(payload) < 2:
                continue
            n = int(payload[:2], 16)
            found_codes = _decode_pairs(payload[2:2 + 4 * n])
        else:
            found_codes = _decode_pairs(payload)
        for c in found_codes:
            if c not in codes:
                codes.append(c)
    return codes if found else None


# ── Per-car runtime state (in memory, rebuilt automatically) ──────────────────
_lock = threading.Lock()
_state = {}      # device_id -> dict
_models = {}     # device_id -> dict(model, scaler, thr, trained_at, n_train)
_training = set()


def _car(did):
    st = _state.get(did)
    if st is None:
        st = _state[did] = {
            "last_store": 0.0, "stored": 0, "since": {}, "cooldown": {},
            "run_start": None, "last_sample": None,
            "mil": None, "dtc_count": None, "dtc_updated": None,
            "flags": deque(maxlen=ANOM_WINDOW), "last_score": None,
        }
    return st


def _sustained(st, key, cond, secs, now_s):
    """True when `cond` has held continuously for `secs` seconds."""
    if not cond:
        st["since"].pop(key, None)
        return False
    t0 = st["since"].setdefault(key, now_s)
    return now_s - t0 >= secs


def _raise_alert(did, st, kind, severity, message, now):
    last = st["cooldown"].get(kind)
    if last is not None and now.timestamp() - last < ALERT_COOLDOWN_S:
        return
    st["cooldown"][kind] = now.timestamp()
    with DB() as db:
        db.add(AlertEvent(device_id=did, ts=now, kind=kind, severity=severity, message=message))
        db.commit()
    print(f"[ALERT] [{did}] {severity}: {message}")


# ── Rule-based alerts ─────────────────────────────────────────────────────────
def _v(sample, k):
    x = sample.get(k)
    return None if x is None or x < 0 else float(x)


def evaluate_rules(did, st, sample, now):
    ns = now.timestamp()
    rpm, speed, clt, batt = _v(sample, "rpm"), _v(sample, "speed"), _v(sample, "coolant"), _v(sample, "batt")
    running = rpm is not None and rpm > 300

    # engine "session": reset after a gap of >2 min without data or with the engine off
    prev = st["last_sample"]
    if not running or (prev is not None and ns - prev > 120):
        st["run_start"] = None
        for k in list(st["since"]):
            st["since"].pop(k, None)
    if running and st["run_start"] is None:
        st["run_start"] = ns
    st["last_sample"] = ns
    if not running:
        return
    run_s = ns - st["run_start"]

    if clt is not None:
        if _sustained(st, "overheat", clt >= 105, 10, ns):
            _raise_alert(did, st, "overheat", "critical",
                         f"Engine overheating: coolant at {clt:.0f} °C. Stop and let it cool down.", now)
        if run_s >= 900 and speed is not None and speed > 0 and \
                _sustained(st, "cold", clt < 70, 120, ns):
            _raise_alert(did, st, "no_warmup", "warning",
                         f"Coolant still only {clt:.0f} °C after 15+ min of running. "
                         "Possible stuck-open thermostat.", now)
    if batt is not None and rpm is not None and rpm > 1000:
        if _sustained(st, "battlow", batt < 12.0, 60, ns):
            _raise_alert(did, st, "battery_low", "warning",
                         f"Battery voltage {batt:.1f} V with the engine running. Charging system may be weak.", now)
        if _sustained(st, "batthigh", batt > 15.2, 30, ns):
            _raise_alert(did, st, "battery_high", "warning",
                         f"Battery voltage {batt:.1f} V is too high. Possible regulator fault.", now)
    if speed is not None and speed == 0 and rpm is not None and clt is not None and clt > 70:
        if _sustained(st, "idle", rpm < 500 or rpm > 1600, 30, ns):
            _raise_alert(did, st, "idle", "warning",
                         f"Abnormal idle speed ({rpm:.0f} RPM). Check for vacuum leaks, dirty throttle or sensor faults.", now)


# ── Fault codes ───────────────────────────────────────────────────────────────
def update_dtc(did, raw=None, mil=None, dtc_n=None, proto=None):
    now = _now()
    with _lock:
        st = _car(did)
        if mil is not None and mil >= 0:
            st["mil"] = int(mil) >= 1
        if dtc_n is not None and dtc_n >= 0:
            st["dtc_count"] = int(dtc_n)
        st["dtc_updated"] = now
        codes = parse_dtc_response(raw, proto)
        if codes is None:
            return
        with DB() as db:
            rows = db.query(DtcEvent).filter_by(device_id=did).all()
            known = {r.code: r for r in rows}
            for code in codes:
                r = known.get(code)
                if r is None:
                    db.add(DtcEvent(device_id=did, code=code, first_seen=now, last_seen=now, active=True))
                    new = True
                else:
                    new = not r.active
                    r.active, r.last_seen = True, now
                if new:
                    db.add(AlertEvent(device_id=did, ts=now, kind="dtc_" + code, severity="warning",
                                      message=f"Fault code {code}: {describe_dtc(code)}"))
                    print(f"[DTC] [{did}] new fault code {code}")
            for code, r in known.items():
                if code not in codes and r.active:
                    r.active = False          # cleared in the ECU
            db.commit()


# ── Isolation Forest (per car) ────────────────────────────────────────────────
def _features(sample):
    """
    Raw readings plus ratios between them. Faults usually break the *relationship*
    between sensors (e.g. airflow vs RPM) while every single value still looks normal,
    and a tree-based model only notices that if the ratio is a feature of its own.
    """
    vals = [_v(sample, k) for k in FEATURES]
    if any(x is None for x in vals) or vals[0] <= 0:
        return None
    rpm, speed, load, throttle, coolant, map_, maf, batt = vals
    return vals + [maf / rpm * 1000.0,            # airflow per engine revolution
                   speed / rpm * 1000.0,          # gear / clutch relationship
                   map_ / (load + 1.0)]           # manifold pressure vs engine load


def train_model(did):
    """Fit an Isolation Forest on this car's own stored telemetry. Returns True if a model was built."""
    with DB() as db:
        rows = db.query(Telemetry).filter(Telemetry.device_id == did) \
                 .order_by(Telemetry.ts.desc()).limit(MAX_TRAIN_ROWS).all()
    X = []
    for r in rows:
        f = _features({k: getattr(r, k) for k in FEATURES})
        if f:
            X.append(f)
    if len(X) < MIN_TRAIN_ROWS:
        return False
    X = np.array(X, dtype=float)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    model = IsolationForest(n_estimators=150, contamination="auto", random_state=0, n_jobs=1).fit(Z)
    thr = float(np.percentile(model.decision_function(Z), ANOM_PCT))
    with _lock:
        _models[did] = {"model": model, "scaler": scaler, "thr": thr,
                        "trained_at": _now(), "n_train": len(X)}
        _car(did)["flags"].clear()
    print(f"[IF] [{did}] trained on {len(X)} samples, threshold {thr:.4f}")
    return True


def _train_bg(did):
    try:
        train_model(did)
    except Exception as e:
        print(f"[IF] [{did}] training failed: {e}")
    finally:
        _training.discard(did)


def _maybe_train(did, st):
    if did in _training:
        return
    m = _models.get(did)
    if m is None:
        if st["stored"] % 100 != 1:           # check the row count only now and then
            return
    else:
        age_h = (_now() - m["trained_at"]).total_seconds() / 3600
        if age_h < RETRAIN_AFTER_H or sum(st["flags"]) > 0:   # never learn from a car that looks abnormal
            return
    _training.add(did)
    threading.Thread(target=_train_bg, args=(did,), daemon=True).start()


def _score(did, st, sample, now):
    m = _models.get(did)
    f = _features(sample)
    if m is None or f is None:
        return
    z = m["scaler"].transform([f])
    s = float(m["model"].decision_function(z)[0])
    st["last_score"] = s
    st["flags"].append(1 if s < m["thr"] else 0)
    if len(st["flags"]) == ANOM_WINDOW and sum(st["flags"]) >= ANOM_MIN_FLAGS:
        _raise_alert(did, st, "anomaly", "warning",
                     "Unusual behaviour detected: the engine readings differ from this car's normal pattern.", now)


# ── Main entry point, called for every OBD2 message ──────────────────────────
def ingest(did, sample):
    """sample: dict with rpm, speed, coolant, iat, load, throttle, map, maf, fuel, batt (-1 = missing)."""
    now = _now()
    with _lock:
        st = _car(did)
        evaluate_rules(did, st, sample, now)
        if now.timestamp() - st["last_store"] < STORE_EVERY_S:
            return
        rpm = _v(sample, "rpm")
        if rpm is None or rpm <= 0:
            return                                    # engine off: nothing worth storing
        st["last_store"] = now.timestamp()
        st["stored"] += 1
        row = Telemetry(device_id=did, ts=now, **{k: _v(sample, k) for k in
                        ("rpm", "speed", "coolant", "iat", "load", "throttle", "map", "maf", "fuel", "batt")})
        with DB() as db:
            db.add(row)
            if st["stored"] % 500 == 0:               # housekeeping
                db.query(Telemetry).filter(Telemetry.ts < now - timedelta(days=RETENTION_DAYS)) \
                  .delete(synchronize_session=False)
                db.query(AlertEvent).filter(AlertEvent.ts < now - timedelta(days=RETENTION_DAYS)) \
                  .delete(synchronize_session=False)
            db.commit()
        _score(did, st, sample, now)
        _maybe_train(did, st)


# ── Status for the dashboard ──────────────────────────────────────────────────
def status(did):
    now = _now()
    with _lock:
        st = _car(did)
        with DB() as db:
            dtcs = db.query(DtcEvent).filter_by(device_id=did, active=True).order_by(DtcEvent.first_seen).all()
            alerts = db.query(AlertEvent).filter(AlertEvent.device_id == did,
                                                 AlertEvent.ts > now - timedelta(hours=24)) \
                       .order_by(AlertEvent.ts.desc()).limit(20).all()
            n_rows = db.query(func.count(Telemetry.id)).filter(Telemetry.device_id == did).scalar() or 0
        m = _models.get(did)
        flagged = sum(st["flags"])
        if m is None:
            a_state = "learning"
        elif len(st["flags"]) == ANOM_WINDOW and flagged >= ANOM_MIN_FLAGS:
            a_state = "unusual"
        else:
            a_state = "normal"
        recent = [a for a in alerts if (now - _aware(a.ts)).total_seconds() < 600]
        if any(a.severity == "critical" for a in recent):
            level = "critical"
        elif dtcs or st["mil"] or a_state == "unusual" or any(a.severity == "warning" for a in recent):
            level = "warning"
        else:
            level = "ok"
        return {
            "car_id": did, "level": level,
            "mil": st["mil"], "dtc_count_reported": st["dtc_count"], "dtc_updated": _iso(st["dtc_updated"]),
            "dtcs": [{"code": d.code, "description": describe_dtc(d.code),
                      "first_seen": _iso(d.first_seen), "last_seen": _iso(d.last_seen)} for d in dtcs],
            "alerts": [{"time": _iso(a.ts), "kind": a.kind, "severity": a.severity, "message": a.message}
                       for a in alerts],
            "anomaly": {"state": a_state, "samples": n_rows, "needed": MIN_TRAIN_ROWS,
                        "trained_at": _iso(m["trained_at"]) if m else None,
                        "trained_on": m["n_train"] if m else 0,
                        "recent_flags": flagged, "window": ANOM_WINDOW},
        }
