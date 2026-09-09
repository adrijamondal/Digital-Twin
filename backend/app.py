import atexit
import json
import os
import threading
import time
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from ml_engine import compute_predictions
from telemetry_collector import RealtimeTelemetryCollector


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
PREDICTIONS_JSON = os.path.join(DATA_DIR, "predictions.json")

os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

collector = RealtimeTelemetryCollector(DATA_DIR, interval=1.0, max_rows=2000)
_prediction_lock = threading.RLock()
_latest_predictions: dict[str, Any] = compute_predictions(collector.get_history(200))
_ml_stop_event = threading.Event()
_ml_thread: threading.Thread | None = None


def _prediction_loop() -> None:
    global _latest_predictions
    while not _ml_stop_event.is_set():
        try:
            predictions = compute_predictions(collector.get_history(200))
            with _prediction_lock:
                _latest_predictions = predictions
            _write_predictions(predictions)
        except Exception as exc:
            with _prediction_lock:
                _latest_predictions = {
                    **_latest_predictions,
                    "timestamp": time.time(),
                    "collecting": True,
                    "anomaly_status": "UNKNOWN",
                    "anomaly_reason": f"ML engine error: {exc}",
                }
        _ml_stop_event.wait(1.0)


def _write_predictions(predictions: dict[str, Any]) -> None:
    tmp_path = PREDICTIONS_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
    os.replace(tmp_path, PREDICTIONS_JSON)


def start_services() -> None:
    global _ml_thread
    collector.start()
    if _ml_thread is None or not _ml_thread.is_alive():
        _ml_stop_event.clear()
        _ml_thread = threading.Thread(target=_prediction_loop, name="ml-engine", daemon=True)
        _ml_thread.start()


def stop_services() -> None:
    _ml_stop_event.set()
    collector.stop()


atexit.register(stop_services)


def _latest_ml() -> dict[str, Any]:
    with _prediction_lock:
        data = dict(_latest_predictions)
    if not data or time.time() - float(data.get("timestamp", 0) or 0) > 3:
        data = compute_predictions(collector.get_history(200))
    return data


def _twin_state() -> dict[str, Any]:
    latest = collector.get_latest()
    predictions = _latest_ml()
    now = time.time()
    last_updated = float(latest.get("timestamp", 0) or 0)
    telemetry_online = bool(last_updated and now - last_updated < 5)
    ml_online = bool(predictions.get("timestamp") and now - float(predictions["timestamp"]) < 5)

    anomaly_status = predictions.get("anomaly_status") or ("ANOMALY" if predictions.get("anomaly") else "NORMAL")
    return {
        **latest,
        "cpu": latest["cpu_usage"],
        "memory": latest["memory_usage"],
        "cpu_pred_10s": predictions.get("cpu_pred_10s", 0.0),
        "cpu_pred_30s": predictions.get("cpu_pred_30s", 0.0),
        "cpu_pred_60s": predictions.get("cpu_pred_60s", 0.0),
        "memory_pred": predictions.get("memory_pred", 0.0),
        "anomaly_status": anomaly_status,
        "anomaly": bool(predictions.get("anomaly", False)),
        "anomaly_reason": predictions.get("anomaly_reason", ""),
        "anomaly_score": predictions.get("anomaly_score", 0.0),
        "anomalous_metrics": predictions.get("anomalous_metrics", []),
        "confidence": predictions.get("confidence", 0.0),
        "failure_risk": predictions.get("failure_risk", 0.0),
        "failure_risk_percent": predictions.get("failure_risk_percent", 0.0),
        "health_score": predictions.get("health_score", 100.0),
        "health_status": predictions.get("health_status", "Healthy"),
        "last_updated": last_updated or now,
        "telemetry_agent_running": telemetry_online,
        "ml_engine_running": ml_online,
    }


@app.after_request
def add_no_cache_headers(response):
    if response.content_type and "application/json" in response.content_type:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/telemetry/live")
def api_live_telemetry():
    return jsonify(collector.get_latest())


@app.get("/api/telemetry/history")
def api_telemetry_history():
    limit = _query_int("limit", 60)
    data = collector.get_history(limit)
    return jsonify({"data": data, "count": len(data)})


@app.get("/api/processes")
def api_processes():
    return jsonify(collector.get_processes())


@app.get("/api/ml/predictions")
def api_ml_predictions():
    return jsonify(_latest_ml())


@app.get("/api/status")
def api_status():
    latest = collector.get_latest()
    last_updated = float(latest.get("timestamp", 0) or 0)
    predictions = _latest_ml()
    return jsonify(
        {
            "status": "running",
            "backend": "flask",
            "collector_running": collector.running,
            "ml_engine_running": _ml_thread is not None and _ml_thread.is_alive(),
            "last_updated": last_updated,
            "last_updated_iso": _iso(last_updated),
            "sample_count": len(collector.get_history(2000)),
            "telemetry_csv": os.path.relpath(collector.csv_path, ROOT_DIR),
            "predictions_timestamp": predictions.get("timestamp"),
            "last_error": collector.last_error,
        }
    )


@app.get("/api/twin/state")
def api_twin_state():
    return jsonify(_twin_state())


@app.get("/api/system/health")
def api_system_health():
    predictions = _latest_ml()
    risk = float(predictions.get("failure_risk_percent", 0.0) or 0.0)
    health = float(predictions.get("health_score", 100.0) or 100.0)
    status = predictions.get("health_status", "Healthy")
    suggestions = _suggestions(_twin_state(), predictions)
    forecast = "OPTIMAL" if health >= 85 and risk < 20 else "STABLE"
    if health < 50 or risk >= 70:
        forecast = "CRITICAL"
    elif health < 75 or risk >= 40:
        forecast = "DEGRADED"
    return jsonify(
        {
            "health_score": health,
            "failure_risk": round(risk / 100, 4),
            "failure_risk_percent": risk,
            "health_status": status,
            "suggestions": suggestions,
            "performance_forecast": forecast,
            "timestamp": time.time(),
        }
    )


# Compatibility routes for the copied Flask dashboard and older generated client.
@app.get("/metrics")
@app.get("/api/telemetry/latest")
def compat_latest():
    return jsonify(collector.get_latest())


@app.get("/history")
def compat_history():
    return jsonify(collector.get_history(60))


@app.get("/predict")
@app.get("/api/predictions/latest")
def compat_predict():
    return jsonify(_latest_ml())


@app.get("/processes")
@app.get("/api/system/processes")
def compat_processes():
    data = collector.get_processes()
    return jsonify(data if request.path.startswith("/api/") else data["processes"])


def _query_int(name: str, default: int) -> int:
    try:
        return max(1, min(2000, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


def _iso(timestamp: float) -> str | None:
    if not timestamp:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp))


def _suggestions(state: dict[str, Any], predictions: dict[str, Any]) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    cpu = float(state.get("cpu_usage", 0) or 0)
    mem = float(state.get("memory_usage", 0) or 0)
    risk = float(predictions.get("failure_risk_percent", 0) or 0)

    if cpu > 85:
        suggestions.append({"type": "resource", "title": "High CPU Load", "description": f"CPU is at {cpu:.1f}%. Review the process monitor for the highest CPU users.", "severity": "critical"})
    elif cpu > 70:
        suggestions.append({"type": "performance", "title": "Elevated CPU", "description": f"CPU is at {cpu:.1f}%. Watch for sustained load.", "severity": "medium"})

    if mem > 90:
        suggestions.append({"type": "resource", "title": "Critical Memory Pressure", "description": f"Memory is at {mem:.1f}%. Close unused apps if the value stays high.", "severity": "critical"})
    elif mem > 75:
        suggestions.append({"type": "resource", "title": "High Memory Usage", "description": f"Memory is at {mem:.1f}%. Monitor for leaks or heavy workloads.", "severity": "high"})

    if predictions.get("anomaly"):
        suggestions.append({"type": "warning", "title": "Anomaly Detected", "description": str(predictions.get("anomaly_reason") or "Live telemetry moved outside its rolling baseline."), "severity": "high"})

    if risk > 60:
        suggestions.append({"type": "performance", "title": "High Failure Risk", "description": f"Risk estimate is {risk:.0f}%. Check CPU, memory, disk, and network spikes.", "severity": "critical"})

    if not suggestions:
        suggestions.append({"type": "performance", "title": "System Healthy", "description": "Live telemetry is within the current rolling baseline.", "severity": "low"})
    return suggestions


def _is_port_in_use(port_num: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port_num)) == 0


if __name__ == "__main__":
    start_services()
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
    elif _is_port_in_use(5000):
        print("Port 5000 is already in use (common on macOS AirPlay). Using port 5001 instead.", flush=True)
        port = 5001
    else:
        port = 5000
    print(f"Digital Twin OS realtime backend running on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

