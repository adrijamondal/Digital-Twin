from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler


FEATURE_COLS = [
    "cpu_usage",
    "memory_usage",
    "disk_read",
    "disk_write",
    "network_in",
    "network_out",
]

FEATURE_LABELS = {
    "cpu_usage": "CPU Usage",
    "memory_usage": "Memory",
    "disk_read": "Disk Read",
    "disk_write": "Disk Write",
    "network_in": "Net In",
    "network_out": "Net Out",
}

MIN_ROWS_FOR_ML = 15
ZSCORE_THRESHOLD = 2.5


def compute_predictions(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _sanitize_history(history)
    now = time.time()

    if not rows:
        return _collecting_result(0, now)

    latest = rows[-1]
    cpu_series = _series(rows, "cpu_usage")
    mem_series = _series(rows, "memory_usage")

    zscores = _compute_zscores(rows)
    anomalous_metrics = [
        FEATURE_LABELS[key]
        for key, value in zscores.items()
        if value > ZSCORE_THRESHOLD
    ]

    isolation = _run_isolation_forest(rows, zscores, anomalous_metrics)
    cpu_pred_10s = _ewma_forecast(cpu_series, 10)
    cpu_pred_30s = _ewma_forecast(cpu_series, 30)
    cpu_pred_60s = _ewma_forecast(cpu_series, 60)
    memory_pred = _ewma_forecast(mem_series, 10)

    failure_risk_percent = _failure_risk(rows, isolation["anomaly"], zscores)
    health_score = _health_score(rows, failure_risk_percent, isolation["anomaly"])
    health_status = _health_status(health_score)

    return {
        "timestamp": now,
        "sample_count": len(rows),
        "collecting": len(rows) < MIN_ROWS_FOR_ML,
        "cpu": latest["cpu_usage"],
        "memory": latest["memory_usage"],
        "cpu_usage": latest["cpu_usage"],
        "memory_usage": latest["memory_usage"],
        "disk_read": latest["disk_read"],
        "disk_write": latest["disk_write"],
        "network_in": latest["network_in"],
        "network_out": latest["network_out"],
        "cpu_pred_10s": cpu_pred_10s,
        "cpu_pred_30s": cpu_pred_30s,
        "cpu_pred_60s": cpu_pred_60s,
        "memory_pred": memory_pred,
        "anomaly": isolation["anomaly"],
        "anomaly_status": isolation["status"],
        "anomaly_reason": isolation["reason"],
        "anomaly_score": isolation["score"],
        "confidence": isolation["confidence"],
        "anomalous_metrics": anomalous_metrics,
        "per_metric_scores": zscores,
        "per_metric_statuses": {
            FEATURE_LABELS[key]: "Anomalous" if value > ZSCORE_THRESHOLD else "OK"
            for key, value in zscores.items()
        },
        "failure_risk": round(failure_risk_percent / 100, 4),
        "failure_risk_percent": round(failure_risk_percent, 2),
        "health_score": round(health_score, 2),
        "health_status": health_status,
    }


def _sanitize_history(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in history:
        try:
            rows.append(
                {
                    "timestamp": float(row.get("timestamp", 0) or 0),
                    "cpu_usage": _finite(row.get("cpu_usage", 0)),
                    "memory_usage": _finite(row.get("memory_usage", 0)),
                    "disk_read": _finite(row.get("disk_read", 0)),
                    "disk_write": _finite(row.get("disk_write", 0)),
                    "network_in": _finite(row.get("network_in", 0)),
                    "network_out": _finite(row.get("network_out", 0)),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows[-200:]


def _collecting_result(count: int, timestamp: float) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "sample_count": count,
        "collecting": True,
        "cpu": 0.0,
        "memory": 0.0,
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "disk_read": 0.0,
        "disk_write": 0.0,
        "network_in": 0.0,
        "network_out": 0.0,
        "cpu_pred_10s": 0.0,
        "cpu_pred_30s": 0.0,
        "cpu_pred_60s": 0.0,
        "memory_pred": 0.0,
        "anomaly": False,
        "anomaly_status": "COLLECTING DATA",
        "anomaly_reason": f"Collecting data ({count}/{MIN_ROWS_FOR_ML} samples)",
        "anomaly_score": 0.0,
        "confidence": 0.0,
        "anomalous_metrics": [],
        "per_metric_scores": {key: 0.0 for key in FEATURE_COLS},
        "per_metric_statuses": {label: "OK" for label in FEATURE_LABELS.values()},
        "failure_risk": 0.0,
        "failure_risk_percent": 0.0,
        "health_score": 100.0,
        "health_status": "Healthy",
    }


def _run_isolation_forest(
    rows: list[dict[str, float]],
    zscores: dict[str, float],
    anomalous_metrics: list[str],
) -> dict[str, Any]:
    if len(rows) < MIN_ROWS_FOR_ML:
        return {
            "anomaly": False,
            "status": "COLLECTING DATA",
            "reason": f"Collecting data ({len(rows)}/{MIN_ROWS_FOR_ML} samples)",
            "score": 0.0,
            "confidence": 0.0,
        }

    matrix = np.array([[row[col] for col in FEATURE_COLS] for row in rows], dtype=float)
    try:
        scaled = RobustScaler().fit_transform(matrix)
        model = IsolationForest(
            n_estimators=120,
            contamination=0.08,
            random_state=42,
        )
        model.fit(scaled)
        prediction = int(model.predict(scaled[-1:])[0])
        raw_score = float(model.score_samples(scaled[-1:])[0])
        anomaly = prediction == -1
    except Exception:
        raw_score = 0.0
        anomaly = False

    z_anomaly = any(value > ZSCORE_THRESHOLD for value in zscores.values())
    anomaly = bool(anomaly or z_anomaly)
    top_metric = max(zscores, key=zscores.get) if zscores else "cpu_usage"
    top_score = zscores.get(top_metric, 0.0)

    if anomaly:
        reason = (
            ", ".join(anomalous_metrics)
            if anomalous_metrics
            else f"Combined metric pattern, strongest signal: {FEATURE_LABELS[top_metric]}"
        )
    else:
        reason = "All live metrics are within the rolling baseline"

    confidence = 0.82 if not anomaly else min(0.99, max(0.55, top_score / 4.0))

    return {
        "anomaly": anomaly,
        "status": "ANOMALY" if anomaly else "NORMAL",
        "reason": reason,
        "score": round(raw_score, 5),
        "confidence": round(confidence, 3),
    }


def _ewma_forecast(values: list[float], steps: int) -> float:
    if not values:
        return 0.0
    if len(values) < 3:
        return round(max(0.0, min(100.0, values[-1])), 2)

    span = min(30, len(values))
    alpha = 2 / (span + 1)
    ewma = values[0]
    previous = ewma
    for value in values[1:]:
        previous = ewma
        ewma = alpha * value + (1 - alpha) * ewma

    trend = ewma - previous
    predicted = ewma + trend * steps
    return round(max(0.0, min(100.0, predicted)), 2)


def _failure_risk(rows: list[dict[str, float]], anomaly: bool, zscores: dict[str, float]) -> float:
    latest = rows[-1]
    cpu = latest["cpu_usage"]
    mem = latest["memory_usage"]
    disk_peak = max(latest["disk_read"], latest["disk_write"])
    net_peak = max(latest["network_in"], latest["network_out"])
    z_peak = max(zscores.values()) if zscores else 0.0

    heuristic = 0.0
    heuristic += max(0.0, cpu - 65) * 1.05
    heuristic += max(0.0, mem - 70) * 1.15
    heuristic += min(18.0, disk_peak / 2048)
    heuristic += min(14.0, net_peak / 2048)
    heuristic += min(18.0, max(0.0, z_peak - 1.5) * 8)
    if anomaly:
        heuristic += 18.0

    rf_component = None
    if len(rows) >= 45:
        try:
            matrix = np.array([[row[col] for col in FEATURE_COLS] for row in rows], dtype=float)
            health_targets = np.array([_base_health(row, 0.0, False) for row in rows], dtype=float)
            y = (health_targets < 78).astype(float)
            scaled = MinMaxScaler().fit_transform(matrix)
            model = RandomForestRegressor(n_estimators=80, random_state=42)
            model.fit(scaled[:-1], y[:-1])
            rf_component = float(model.predict(scaled[-1:])[0]) * 100
        except Exception:
            rf_component = None

    risk = heuristic if rf_component is None else (heuristic * 0.55 + rf_component * 0.45)
    return max(0.0, min(100.0, risk))


def _health_score(rows: list[dict[str, float]], risk_percent: float, anomaly: bool) -> float:
    current = _base_health(rows[-1], risk_percent, anomaly)

    if len(rows) < 10:
        return current

    try:
        base_series = np.array([_base_health(row, risk_percent, anomaly) for row in rows[-30:]])
        x = np.arange(len(base_series)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, base_series)
        trend_estimate = float(model.predict([[len(base_series)]])[0])
        return max(0.0, min(100.0, (current * 0.7 + trend_estimate * 0.3)))
    except Exception:
        return current


def _base_health(row: dict[str, float], risk_percent: float, anomaly: bool) -> float:
    health = 100.0
    health -= max(0.0, row["cpu_usage"] - 50) * 0.55
    health -= max(0.0, row["memory_usage"] - 60) * 0.65
    health -= min(12.0, max(row["disk_read"], row["disk_write"]) / 4096)
    health -= min(8.0, max(row["network_in"], row["network_out"]) / 4096)
    health -= risk_percent * 0.2
    if anomaly:
        health -= 12.0
    return max(0.0, min(100.0, health))


def _health_status(score: float) -> str:
    if score >= 80:
        return "Healthy"
    if score >= 50:
        return "Warning"
    return "Critical"


def _compute_zscores(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) < 3:
        return {key: 0.0 for key in FEATURE_COLS}

    result: dict[str, float] = {}
    latest = rows[-1]
    baseline = rows[:-1]
    for key in FEATURE_COLS:
        values = np.array([row[key] for row in baseline], dtype=float)
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std(ddof=0)) if len(values) else 0.0
        if std < 1e-9:
            z = 0.0
        else:
            z = abs((latest[key] - mean) / std)
        result[key] = round(float(z), 3)
    return result


def _series(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row]


def _finite(value: Any) -> float:
    number = float(value or 0.0)
    if not math.isfinite(number):
        return 0.0
    return number
