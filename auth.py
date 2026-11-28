import os, json, secrets, hashlib, hmac
from datetime import datetime, timezone
from flask import request, session, jsonify, redirect, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

url = os.environ.get("DATABASE_URL", "sqlite:///dev.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(url, pool_pre_ping=True)
DB = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(200), unique=True, nullable=False)
    name = Column(String(120)); company = Column(String(120)); pw_hash = Column(String(300))


class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    data = Column(Text)                      # JSON: make, model, nickname, specs


class Device(Base):
    __tablename__ = "devices"
    device_id = Column(String(40), primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), index=True)
    secret_hash = Column(String(64))
    last_seen = Column(DateTime(timezone=True))


PUBLIC = ("/", "/health", "/login", "/landing.html", "/login.html", "/api/signup", "/api/login", "/cars_dataset.json")
GUARDED = ("/gps", "/obd2", "/trip/status", "/trip/config", "/clear", "/diagnostics")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def car_specs(did):
    """Specs (make, fuel_type, engine_size...) of the vehicle registered for this device id."""
    with DB() as db:
        v = db.query(Vehicle).join(Device, Device.vehicle_id == Vehicle.id).filter(Device.device_id == did).first()
        return json.loads(v.data) if v else {}


def init_auth(app):
    Base.metadata.create_all(engine)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")))
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    ph = PasswordHasher()
    body = lambda: request.get_json(silent=True) or {}

    @app.before_request
    def guard():
        p = request.path
        if p in PUBLIC or p.endswith((".css", ".js", ".ico")):
            return
        if p == "/devices":                                   # would leak other managers' units
            return jsonify(error="disabled"), 404
        if request.method == "POST" and p in ("/gps", "/obd2"):   # devices: token required
            cid, tok = request.values.get("car_id", ""), request.values.get("token", "")
            with DB() as db:
                d = db.get(Device, cid)
                if not d or not hmac.compare_digest(d.secret_hash, sha(tok)):
                    return jsonify(error="bad device token"), 401
                d.last_seen = datetime.now(timezone.utc)
                db.commit()
            return
        if not session.get("uid"):
            if p.startswith("/api/") or p in GUARDED:
                return jsonify(error="auth"), 401
            return redirect("/login")
        if p in GUARDED:                                      # ownership check
            cid = request.values.get("car_id") or body().get("car_id")
            with DB() as db:
                ok = db.query(Device).join(Vehicle, Vehicle.id == Device.vehicle_id).filter(
                    Device.device_id == cid, Vehicle.owner_id == session["uid"]).first()
            if not ok:
                return jsonify(error="forbidden"), 403

    @app.route("/login")
    def login_page():
        return send_from_directory("public", "login.html")

    @app.route("/dashboard")
    def dashboard_page():
        return send_from_directory("public", "index.html")

    @app.post("/api/signup")
    @limiter.limit("10 per hour")
    def signup():
        d = body()
        name, email, pw = (d.get("name") or "").strip(), (d.get("email") or "").strip().lower(), d.get("password") or ""
        if not name or "@" not in email or len(pw) < 8:
            return jsonify(error="Name, a valid email and a password of 8+ characters are required"), 400
        with DB() as db:
            if db.query(User).filter_by(email=email).first():
                return jsonify(error="This email is already registered"), 409
            u = User(email=email, name=name, company=(d.get("company") or "").strip(), pw_hash=ph.hash(pw))
            db.add(u); db.commit()
            session.clear(); session["uid"] = u.id; session["name"] = name
        return jsonify(ok=True)

    @app.post("/api/login")
    @limiter.limit("5 per minute")
    def login():
        d = body()
        with DB() as db:
            u = db.query(User).filter_by(email=(d.get("email") or "").strip().lower()).first()
            info = (u.pw_hash, u.id, u.name) if u else None
        try:
            if not info:
                raise VerifyMismatchError()
            ph.verify(info[0], d.get("password") or "")
        except VerifyMismatchError:
            return jsonify(error="Wrong email or password"), 401
        session.clear(); session["uid"] = info[1]; session["name"] = info[2]
        return jsonify(ok=True)

    @app.post("/api/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/api/me")
    def me():
        return jsonify(name=session.get("name"))

    @app.get("/api/fleet")
    def fleet_list():
        now, out = datetime.now(timezone.utc), []
        with DB() as db:
            rows = db.query(Vehicle, Device).join(Device, Device.vehicle_id == Vehicle.id).filter(
                Vehicle.owner_id == session["uid"]).order_by(Vehicle.id).all()
            for v, dv in rows:
                ls = dv.last_seen
                if ls is not None and ls.tzinfo is None:
                    ls = ls.replace(tzinfo=timezone.utc)
                c = json.loads(v.data)
                c.update(device_id=dv.device_id, id=v.id, online=bool(ls and (now - ls).total_seconds() < 60))
                out.append(c)
        return jsonify(out)

    @app.post("/api/fleet")
    def fleet_add():
        car = body()
        did, secret = "CAR-" + secrets.token_hex(8).upper(), secrets.token_urlsafe(32)
        with DB() as db:
            v = Vehicle(owner_id=session["uid"], data=json.dumps(car))
            db.add(v); db.flush()
            vid = v.id
            db.add(Device(device_id=did, vehicle_id=vid, secret_hash=sha(secret)))
            db.commit()
        car.update(device_id=did, id=vid, online=False)
        return jsonify(car=car, secret=secret)          # the secret is shown only this once

    @app.delete("/api/fleet/<did>")
    def fleet_del(did):
        with DB() as db:
            row = db.query(Vehicle, Device).join(Device, Device.vehicle_id == Vehicle.id).filter(
                Device.device_id == did, Vehicle.owner_id == session["uid"]).first()
            if row:
                db.delete(row[1]); db.delete(row[0]); db.commit()
        return jsonify(ok=True)
