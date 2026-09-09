"use strict";

const API_BASE = "";
const HISTORY_LIMIT = 60;
const METRIC_LABELS = {
  cpu_usage: "CPU Usage",
  memory_usage: "Memory",
  disk_read: "Disk Read",
  disk_write: "Disk Write",
  network_in: "Net In",
  network_out: "Net Out",
};

let lastAnomalyState = false;
let wasConnected = false;

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

const CHART_OPTS = (label, color, fillColor) => ({
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label,
        data: [],
        borderColor: color,
        backgroundColor: fillColor,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.35,
        fill: true,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 8 },
        grid: { color: "#1a2340" },
      },
      y: {
        ticks: { color: "#64748b", font: { size: 9 } },
        grid: { color: "#1a2340" },
        beginAtZero: true,
      },
    },
  },
});

const TWO_SERIES_OPTS = (first, second, firstColor, secondColor) => ({
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: first,
        data: [],
        borderColor: firstColor,
        backgroundColor: hexToAlpha(firstColor, 0.06),
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
      {
        label: second,
        data: [],
        borderColor: secondColor,
        backgroundColor: hexToAlpha(secondColor, 0.06),
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#64748b", boxWidth: 10, font: { size: 10 } } },
    },
    scales: {
      x: { ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 8 }, grid: { color: "#1a2340" } },
      y: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "#1a2340" }, beginAtZero: true },
    },
  },
});

function makeChart(id, opts) {
  const node = document.getElementById(id);
  return new Chart(node.getContext("2d"), opts);
}

const cpuChart = makeChart("chart-cpu", CHART_OPTS("CPU %", "#00d4ff", "rgba(0,212,255,0.08)"));
const memChart = makeChart("chart-mem", CHART_OPTS("Memory %", "#7c3aed", "rgba(124,58,237,0.08)"));
const diskChart = makeChart("chart-disk", TWO_SERIES_OPTS("Read", "Write", "#00d4ff", "#7c3aed"));
const netChart = makeChart("chart-net", TWO_SERIES_OPTS("In", "Out", "#22c55e", "#f59e0b"));

function fmt(value, decimals = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return number.toFixed(decimals);
}

function fmtPercent(value, decimals = 1) {
  return fmt(value, decimals) + "%";
}

function fmtRate(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (number < 1024) return number.toFixed(1);
  return (number / 1024).toFixed(2) + " MB";
}

function hexToAlpha(hex, alpha) {
  const color = hex.replace("#", "");
  const bigint = parseInt(color, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function colorForValue(el, value, warnAt, badAt) {
  el.classList.remove("good", "warn", "bad");
  if (value >= badAt) el.classList.add("bad");
  else if (value >= warnAt) el.classList.add("warn");
  else el.classList.add("good");
}

function setStatus(connected, text, timestamp) {
  const dot = document.getElementById("status-dot");
  const label = document.getElementById("status-label");
  const updated = document.getElementById("last-update");

  dot.classList.toggle("anomaly", !connected);
  label.textContent = text;
  updated.textContent = timestamp ? "Updated: " + new Date(timestamp * 1000).toLocaleTimeString() : "Waiting for live telemetry";

  if (connected && !wasConnected) addLog("Connected to Flask realtime backend", "normal");
  if (!connected && wasConnected) addLog("Frontend cannot connect to backend", "anomaly");
  wasConnected = connected;
}

function addLog(message, cls) {
  const list = document.getElementById("log-list");
  const li = document.createElement("li");
  li.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  li.className = cls || "";
  list.prepend(li);
  while (list.children.length > 40) list.removeChild(list.lastChild);
}

function tsLabel(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function updateChart(chart, labels, ...dataSeries) {
  chart.data.labels = labels;
  dataSeries.forEach((series, index) => {
    chart.data.datasets[index].data = series;
  });
  chart.update("none");
}

async function getJson(path) {
  const response = await fetch(API_BASE + path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

async function fetchLive() {
  try {
    const [live, predictions, status] = await Promise.all([
      getJson("/api/telemetry/live"),
      getJson("/api/ml/predictions"),
      getJson("/api/status"),
    ]);

    updateDashboard(live, predictions);
    updateMLTab(live, predictions);
    setStatus(true, status.collector_running ? "LIVE" : "COLLECTING", live.timestamp);
  } catch (error) {
    console.warn("live fetch error:", error);
    setStatus(false, "DISCONNECTED", null);
  }
}

function updateDashboard(live, predictions) {
  const cpu = Number(live.cpu_usage || predictions.cpu || 0);
  const memory = Number(live.memory_usage || predictions.memory || 0);
  const diskRead = Number(live.disk_read || 0);
  const diskWrite = Number(live.disk_write || 0);
  const netIn = Number(live.network_in || 0);
  const netOut = Number(live.network_out || 0);

  const cpuEl = document.getElementById("kpi-cpu");
  cpuEl.textContent = fmtPercent(cpu);
  colorForValue(cpuEl, cpu, 70, 90);

  const memEl = document.getElementById("kpi-mem");
  memEl.textContent = fmtPercent(memory);
  colorForValue(memEl, memory, 75, 90);

  document.getElementById("kpi-dread").textContent = fmtRate(diskRead);
  document.getElementById("kpi-dwrite").textContent = fmtRate(diskWrite);
  document.getElementById("kpi-netin").textContent = fmtRate(netIn);
  document.getElementById("kpi-netout").textContent = fmtRate(netOut);

  document.getElementById("pred-10s").textContent = fmtPercent(predictions.cpu_pred_10s);
  document.getElementById("pred-30s").textContent = fmtPercent(predictions.cpu_pred_30s);
  document.getElementById("pred-60s").textContent = fmtPercent(predictions.cpu_pred_60s);

  const health = clamp(Number(predictions.health_score || 100), 0, 100);
  const risk = clamp(Number(predictions.failure_risk_percent ?? (predictions.failure_risk || 0) * 100), 0, 100);
  document.getElementById("bar-health").style.width = health + "%";
  document.getElementById("lbl-health").textContent = fmt(health, 0) + " - " + (predictions.health_status || "Healthy");
  document.getElementById("bar-risk").style.width = risk + "%";
  document.getElementById("lbl-risk").textContent = fmt(risk, 0) + "%";

  const banner = document.getElementById("anomaly-banner");
  const statusDot = document.getElementById("status-dot");
  const anomaly = Boolean(predictions.anomaly);
  if (anomaly) {
    banner.classList.add("visible");
    statusDot.classList.add("anomaly");
    document.getElementById("anomaly-reason").textContent = "ANOMALY: " + (predictions.anomaly_reason || "Unknown");
    if (!lastAnomalyState) addLog("ANOMALY: " + (predictions.anomaly_reason || "Unknown"), "anomaly");
  } else {
    banner.classList.remove("visible");
    if (wasConnected) statusDot.classList.remove("anomaly");
    if (lastAnomalyState) addLog("System returned to NORMAL", "normal");
  }
  lastAnomalyState = anomaly;
}

async function fetchHistory() {
  try {
    const response = await getJson(`/api/telemetry/history?limit=${HISTORY_LIMIT}`);
    const rows = Array.isArray(response) ? response : response.data || [];
    const labels = rows.map((row) => tsLabel(row.timestamp));
    updateChart(cpuChart, labels, rows.map((row) => row.cpu_usage));
    updateChart(memChart, labels, rows.map((row) => row.memory_usage));
    updateChart(diskChart, labels, rows.map((row) => row.disk_read), rows.map((row) => row.disk_write));
    updateChart(netChart, labels, rows.map((row) => row.network_in), rows.map((row) => row.network_out));
  } catch (error) {
    console.warn("history fetch error:", error);
  }
}

async function fetchProcesses() {
  try {
    const response = await getJson("/api/processes");
    const processes = Array.isArray(response) ? response : response.processes || [];
    const total = response.total_processes || processes.length;
    document.getElementById("proc-count").textContent = `TOP ${processes.length} / ${total}`;

    const tbody = document.getElementById("proc-body");
    tbody.innerHTML = "";
    if (!processes.length) {
      tbody.innerHTML = '<tr><td colspan="6">No process data yet</td></tr>';
      return;
    }

    for (const proc of processes) {
      const cpu = Number(proc.cpu_percent ?? proc.cpu ?? 0);
      const mem = Number(proc.memory_percent ?? proc.mem ?? 0);
      const cpuClass = cpu >= 20 ? "cpu-high" : cpu >= 5 ? "cpu-med" : "cpu-low";
      const status = proc.status || proc.state || "unknown";
      const statusClass = status === "running" ? "status-running" : status === "sleeping" ? "status-sleeping" : "status-other";
      const barWidth = Math.min(100, cpu * 2);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color:var(--muted)">${escapeHtml(proc.pid)}</td>
        <td title="${escapeHtml(proc.name)}">${escapeHtml(String(proc.name || "unknown").slice(0, 24))}</td>
        <td class="${cpuClass}"><span class="proc-cpu-bar" style="width:${barWidth}px"></span>${fmt(cpu)}%</td>
        <td style="color:var(--muted)">${fmt(mem, 2)}%</td>
        <td class="${statusClass}">${escapeHtml(status)}</td>
        <td style="color:var(--muted)">${escapeHtml(proc.username || proc.user || "unknown")}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (error) {
    console.warn("process fetch error:", error);
  }
}

function updateMLTab(live, predictions) {
  const anomaly = Boolean(predictions.anomaly);
  const collecting = Boolean(predictions.collecting);
  const statusText = predictions.anomaly_status || (anomaly ? "ANOMALY" : "NORMAL");
  const risk = clamp(Number(predictions.failure_risk_percent ?? (predictions.failure_risk || 0) * 100), 0, 100);
  const health = clamp(Number(predictions.health_score || 100), 0, 100);

  const anomalyEl = document.getElementById("ml-anomaly-status");
  anomalyEl.textContent = collecting ? "COLLECTING DATA" : statusText;
  anomalyEl.className = "ml-live-val " + (anomaly ? "warn" : collecting ? "collecting" : "ok");
  document.getElementById("ml-anomaly-reason").textContent = predictions.anomaly_reason || "--";
  document.getElementById("ml-pred-10").textContent = fmtPercent(predictions.cpu_pred_10s);

  const riskEl = document.getElementById("ml-risk");
  riskEl.textContent = fmt(risk, 0) + "%";
  riskEl.className = "ml-live-val " + (risk > 50 ? "warn" : "ok");

  const healthEl = document.getElementById("ml-health");
  healthEl.textContent = fmt(health, 0) + " - " + (predictions.health_status || "Healthy");
  healthEl.className = "ml-live-val " + (health < 60 ? "warn" : "ok");

  const ifBadge = document.getElementById("if-badge");
  ifBadge.textContent = collecting ? "COLLECTING" : statusText;
  ifBadge.className = "algo-badge anomaly-badge " + (anomaly ? "on" : "off");
  document.getElementById("rf-badge").textContent = fmt(risk, 0) + "% RISK";
  document.getElementById("lr-badge").textContent = "HEALTH " + fmt(health, 0);

  const anomalous = (predictions.anomalous_metrics || []).join(", ") || "none";
  document.getElementById("if-live").innerHTML =
    `Status: <strong>${escapeHtml(statusText)}</strong><br>` +
    `Reason: ${escapeHtml(predictions.anomaly_reason || "--")}<br>` +
    `Anomalous metrics: ${escapeHtml(anomalous)}<br>` +
    `Anomaly score: ${fmt(predictions.anomaly_score || 0, 4)}<br>` +
    `Samples: ${escapeHtml(predictions.sample_count || 0)}`;

  document.getElementById("ewma-live").innerHTML =
    `CPU now: <strong>${fmtPercent(live.cpu_usage || predictions.cpu || 0)}</strong><br>` +
    `+10s -> <strong>${fmtPercent(predictions.cpu_pred_10s)}</strong> &nbsp;` +
    `+30s -> <strong>${fmtPercent(predictions.cpu_pred_30s)}</strong> &nbsp;` +
    `+60s -> <strong>${fmtPercent(predictions.cpu_pred_60s)}</strong>`;

  document.getElementById("rf-live").innerHTML =
    `Failure risk: <strong>${fmt(risk, 0)}%</strong><br>` +
    `Mode: ${predictions.sample_count >= 45 ? "Random Forest + live heuristic" : "Live heuristic while samples accumulate"}`;

  document.getElementById("lr-live").innerHTML =
    `Health score: <strong>${fmt(health, 0)} / 100</strong><br>` +
    `Status: ${escapeHtml(predictions.health_status || "Healthy")}`;

  buildZscoreGrid(predictions.per_metric_scores || {});
}

function buildZscoreGrid(perMetric) {
  const grid = document.getElementById("zscore-grid");
  grid.innerHTML = "";
  const entries = Object.entries(perMetric);
  if (!entries.length) {
    grid.innerHTML = '<div class="zscore-item">Collecting baseline...</div>';
    return;
  }

  for (const [key, value] of entries) {
    const z = Number(value || 0);
    const label = METRIC_LABELS[key] || key;
    const anomalous = z > 2.5;
    const barWidth = Math.min(100, (z / 3) * 100).toFixed(1);
    const barColor = anomalous ? "var(--red)" : z > 1.5 ? "var(--yellow)" : "var(--green)";
    const item = document.createElement("div");
    item.className = "zscore-item";
    item.innerHTML = `
      <div class="zscore-name">${escapeHtml(label)}</div>
      <div class="zscore-bar-bg">
        <div class="zscore-bar" style="width:${barWidth}%;background:${barColor}"></div>
      </div>
      <div class="zscore-val">
        <span class="zscore-num">Z = ${fmt(z, 2)}</span>
        <span class="${anomalous ? "zscore-hi" : "zscore-ok"}">${anomalous ? "ANOMALOUS" : "OK"}</span>
      </div>
    `;
    grid.appendChild(item);
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

addLog("Dashboard started. Connecting to live psutil telemetry...", "normal");
fetchLive();
fetchHistory();
fetchProcesses();

setInterval(fetchLive, 1000);
setInterval(fetchHistory, 1000);
setInterval(fetchProcesses, 1000);
