# Digital Twin — Real-Time OS Monitor & Predictive Diagnostics

A real-time **Digital Twin of an Operating System** that mirrors and forecasts the host machine's telemetry in real time. Powered by a Python Flask backend, `psutil` system metrics collection, embedded machine learning models, and an interactive real-time dashboard with Chart.js.

---

## System Architecture

```
                                  HOST OPERATING SYSTEM
                      (macOS / Linux / Windows via psutil)
                                       │
                                       ▼ (Every 1s)
                       ┌────────────────────────────────┐
                       │  Realtime Telemetry Collector  │
                       │ (backend/telemetry_collector)  │
                       └───────────────┬────────────────┘
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
       ┌───────────────┐                             ┌───────────────────┐
       │ Rolling CSV   │                             │ Process Monitor   │
       │ (2,000 rows)  │                             │ (Top 15 by CPU)   │
       └───────┬───────┘                             └─────────┬─────────┘
               │                                               │
               ▼                                               │
       ┌───────────────────────────────┐                       │
       │       ML Engine Pipeline      │                       │
       │  • Isolation Forest (Anomaly) │                       │
       │  • EWMA Forecasting (CPU/Mem) │                       │
       │  • Random Forest (Risk Model) │                       │
       │  • Linear Regression (Health) │                       │
       │  • Per-metric Z-score filter  │                       │
       └───────────────┬───────────────┘                       │
                       │                                       │
                       ▼                                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │                Flask REST API Server                    │
       │                  (backend/app.py)                       │
       └───────────────────────────┬─────────────────────────────┘
                                   │ (JSON / 1s Polling)
                                   ▼
       ┌─────────────────────────────────────────────────────────┐
       │             Real-Time Web Dashboard (UI)                │
       │      HTML5 + Vanilla CSS + Chart.js (Dark Cyberpunk)    │
       │      • Live KPI Cards        • 60s Rolling Charts       │
       │      • Forecast Panels       • Process Monitor Table    │
       │      • ML Algorithms & Per-Metric Z-Scores View         │
       └─────────────────────────────────────────────────────────┘
```

---

## Features

- **Live Telemetry (1s Refresh)**: Continuously tracks CPU utilization, RAM usage, disk read/write throughput (KB/s), and network input/output rates (KB/s).
- **Process Activity Monitor**: Live listing of the top 15 resource-consuming processes sorted by CPU and memory usage with PID and status.
- **Embedded Machine Learning**:
  - **Isolation Forest**: Unsupervised anomaly detection fitted over rolling telemetry windows.
  - **EWMA (Exponentially Weighted Moving Average)**: Real-time forecasting of CPU demand at `+10s`, `+30s`, and `+60s` intervals.
  - **Random Forest**: Failure risk modeling based on sustained resource pressure and historical patterns.
  - **Linear Regression**: Health trend smoothing to compute a unified 0–100 system health score.
  - **Z-Score Outlier Analysis**: Statistical standard deviation scoring across each individual metric (`Z > 2.5` flagged anomalous).
- **Intelligent Port Fallback**: Automatically checks port availability and smoothly switches to port `5001` if macOS AirPlay Receiver occupies port `5000`.
- **Zero External Database**: Lightweight and self-contained using an in-memory buffer and a rolling 2,000-row CSV file (`data/telemetry.csv`).

---

## Project Structure

```text
dt-os-realtime/
├── backend/
│   ├── app.py                   # Flask server, routing, and lifecycle management
│   ├── telemetry_collector.py   # Multi-threaded psutil hardware & process collector
│   ├── ml_engine.py             # Isolation Forest, EWMA, RF risk & health analytics
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html               # Single-page application dashboard
│   └── static/
│       ├── script.js            # Chart.js instances, polling loop & DOM updates
│       └── style.css            # Responsive dark cyberpunk theme
├── data/
│   ├── telemetry.csv            # Rolling telemetry log (max 2,000 rows)
│   └── predictions.json         # Latest serialized ML inference state
├── run_realtime.sh              # One-click startup script for macOS & Linux
└── README.md                    # Project documentation
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** (Tested on Python 3.11, 3.12, 3.14 on macOS, Linux, and Windows)
- Modern web browser (Chrome, Safari, Firefox, Edge)

---

### macOS & Linux

1. **Clone or open the project folder**:
   ```bash
   cd "Digital Twin"
   ```

2. **Run using the helper script** (handles virtual environment and dependencies automatically if configured):
   ```bash
   chmod +x run_realtime.sh
   ./run_realtime.sh
   ```

   *Or run manually via terminal:*
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install dependencies
   pip install -r backend/requirements.txt

   # Start the application
   python backend/app.py
   ```

3. **Open the dashboard in your browser**:
   ```text
   http://localhost:5001
   ```
   *(or `http://localhost:5000` if port 5000 is not occupied by AirPlay)*

---

### Windows (PowerShell)

1. **Open PowerShell in the project root**:
   ```powershell
   cd "Digital Twin"
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r backend/requirements.txt
   ```

4. **Start the backend**:
   ```powershell
   $env:PORT="5000"
   python backend/app.py
   ```

5. **Open in browser**:
   ```text
   http://localhost:5000
   ```

---

## REST API Reference

The Flask backend exposes the following JSON endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the main HTML5/JavaScript dashboard. |
| `/api/status` | `GET` | Backend health, collector status, sample count, and timestamps. |
| `/api/telemetry/live` | `GET` | Most recent 1-second telemetry sample (CPU, RAM, Disk, Net). |
| `/api/telemetry/history?limit=60` | `GET` | Rolling telemetry window (default: last 60 samples for charts). |
| `/api/processes` | `GET` | Top 15 active OS processes sorted by CPU and memory consumption. |
| `/api/ml/predictions` | `GET` | Full ML outputs: forecasts (+10s/+30s/+60s), anomaly state, risk %, and health score. |
| `/api/twin/state` | `GET` | Consolidated state merging latest telemetry with ML inference. |
| `/api/system/health` | `GET` | High-level health score, risk percentage, and automated diagnostic recommendations. |

---

## Machine Learning Pipeline

### 1. Anomaly Detection (Isolation Forest)
- Fits an `IsolationForest` on a rolling window of recent telemetry points.
- Detects multi-dimensional structural deviations across CPU, memory, disk I/O, and network throughput.
- Emits anomaly flag, reason string, and confidence score.

### 2. Time-Series Forecasting (EWMA)
- Uses Exponentially Weighted Moving Averages with decaying weights ($\alpha$) to capture short-term momentum and trend.
- Generates forward-looking CPU usage projections at **+10 seconds**, **+30 seconds**, and **+60 seconds**.

### 3. Failure Risk Estimation (Random Forest)
- Calculates failure probability ($0.0 - 1.0$ / $0\% - 100\%$) based on sustained resource saturation, anomaly flags, and sudden spikes.
- Emits real-time risk alerts if the estimated probability exceeds safety baselines.

### 4. System Health Index (Linear Regression)
- Integrates resource headroom, anomaly severity, and failure risk into a smoothed **0 to 100 Health Score**.
- Categorizes current state into `Healthy`, `Degraded`, or `Critical`.

### 5. Statistical Z-Scores
- Calculates standard scores: $Z = \frac{x - \mu}{\sigma}$ for each metric.
- Any metric exceeding $|Z| > 2.5$ is flagged individually for root-cause visibility.

---

## Verification & Testing

Verify that your local deployment is working correctly:

```bash
# 1. Verify backend health
curl http://localhost:5001/api/status

# 2. Verify live telemetry updates
curl http://localhost:5001/api/telemetry/live

# 3. Verify ML engine predictions
curl http://localhost:5001/api/ml/predictions
```

Telemetry will continuously write to [data/telemetry.csv](file:///Users/adrijamondal/Desktop/Digital%20Twin/data/telemetry.csv) up to a maximum of 2,000 rolling rows.
