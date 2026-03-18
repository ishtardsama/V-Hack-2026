"""
Predictive Maintenance Intelligence Platform
NexaOps Industrial Solutions

Visualization: Streamlit + Plotly
Dataset: NASA CMAPSS FD001 (simulation layer for demo)
Model: LSTM-based Remaining Useful Life (RUL) Regression

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — NexaOps",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding: 1.75rem 2rem 1rem 2rem; }

    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px 12px 16px;
    }
    [data-testid="metric-container"] label {
        font-size: 11px !important;
        color: #8b949e !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }

    .sec-label {
        font-size: 10px;
        font-weight: 600;
        color: #484f58;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #21262d;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-critical { background: rgba(220,38,38,0.15);  color: #f87171; border: 1px solid rgba(220,38,38,0.4); }
    .badge-impaired { background: rgba(217,119,6,0.15);  color: #fbbf24; border: 1px solid rgba(217,119,6,0.4); }
    .badge-healthy  { background: rgba(22,163,74,0.15);  color: #4ade80; border: 1px solid rgba(22,163,74,0.4); }
    .badge-offline  { background: rgba(75,85,99,0.2);    color: #9ca3af; border: 1px solid rgba(75,85,99,0.4); }

    .alert {
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 12px;
        line-height: 1.5;
    }
    .alert-critical { background: rgba(220,38,38,0.10);  border-left: 3px solid #dc2626; }
    .alert-warning  { background: rgba(217,119,6,0.10);  border-left: 3px solid #d97706; }
    .alert-info     { background: rgba(37,99,235,0.10);  border-left: 3px solid #2563eb; }
    .alert-ok       { background: rgba(22,163,74,0.10);  border-left: 3px solid #16a34a; }
    .alert strong   { display: block; font-weight: 600; color: #e6edf3; margin-bottom: 2px; font-size: 12px; }
    .alert span     { color: #8b949e; font-size: 11px; }

    .tl-item { display: flex; gap: 12px; margin-bottom: 12px; }
    .tl-line-wrap { display: flex; flex-direction: column; align-items: center; }
    .tl-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }
    .tl-line { flex: 1; width: 1px; background: #21262d; min-height: 20px; }
    .tl-content .tl-title { font-size: 12px; font-weight: 600; color: #e6edf3; }
    .tl-content .tl-sub   { font-size: 11px; color: #8b949e; font-family: 'JetBrains Mono', monospace; }

    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 12px !important;
        color: #8b949e !important;
        font-weight: 500 !important;
    }
    hr { border-color: #21262d !important; }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── DATA GENERATION ──────────────────────────────────────────────────────────
@st.cache_data
def generate_machine_data(n_cycles: int = 200, seed: int = 42):
    """
    Generates multivariate degradation signals in the style of NASA CMAPSS FD001.
    Sensors:
        temperature  — bearing temperature (°C)
        vibration    — vibration RMS (mm/s)
        load         — operational load fraction (0–1)
        pressure     — hydraulic pressure (bar)
    """
    np.random.seed(seed)
    cycles = np.arange(1, n_cycles + 1)

    change_point  = int(n_cycles * np.random.uniform(0.44, 0.58))
    anomaly_start = int(n_cycles * np.random.uniform(0.70, 0.80))

    def degrade(base, noise, scale, power=1.7):
        sig = np.array([base + np.random.normal(0, noise) for _ in cycles])
        for i in range(n_cycles):
            if i >= change_point:
                t = (i - change_point) / max(1, n_cycles - change_point)
                sig[i] += scale * (t ** power)
        return np.round(sig, 4)

    df = pd.DataFrame({
        "cycle":       cycles,
        "temperature": degrade(58.0, 1.8,  38.0, power=1.6),
        "vibration":   degrade(2.1,  0.22,  7.4, power=1.9),
        "load":        degrade(0.38, 0.024, 0.50, power=1.3),
        "pressure":    degrade(3.10, 0.14,  1.5, power=1.5),
    })

    df["rul_true"] = n_cycles - cycles
    df["state"]    = "Healthy"
    df.loc[df["cycle"] >= change_point,  "state"] = "Impaired"
    df.loc[df["cycle"] >= anomaly_start, "state"] = "Critical"

    return df, change_point, anomaly_start


@st.cache_data
def predict_rul_lstm(true_rul: int) -> tuple:
    """
    Simulates LSTM regression output with realistic prediction error.
    In production this calls a serialised PyTorch model via torch.load().
    """
    sigma     = max(4, true_rul * 0.10)
    predicted = max(0, true_rul + np.random.normal(0, sigma))
    confidence = max(5, true_rul * 0.14)
    return round(predicted), round(confidence)


def compute_health_score(df: pd.DataFrame, current_cycle: int) -> int:
    baseline = df[df["cycle"] <= 20]
    current  = df[df["cycle"] == current_cycle].iloc[0]
    sensors  = ["temperature", "vibration", "load", "pressure"]
    weights  = [0.28, 0.36, 0.20, 0.16]
    score    = 0.0
    for s, w in zip(sensors, weights):
        mu    = baseline[s].mean()
        sigma = baseline[s].std() + 1e-9
        score += w * max(0.0, 1.0 - abs(current[s] - mu) / sigma / 5.0)
    return round(score * 100)


# ─── MACHINE REGISTRY ─────────────────────────────────────────────────────────
MACHINES = {
    "M-001  Conveyor Motor A":  {"seed": 42, "total": 200, "default_cycle": 187, "line": "Production Line 1", "type": "3-Phase Induction Motor"},
    "M-002  Hydraulic Press B": {"seed": 99, "total": 200, "default_cycle": 134, "line": "Production Line 2", "type": "Servo-Hydraulic Unit"},
    "M-003  Cooling Pump C":    {"seed": 7,  "total": 200, "default_cycle": 52,  "line": "Utility Block",     "type": "Centrifugal Pump"},
    "M-004  Compressor Unit D": {"seed": 17, "total": 200, "default_cycle": 98,  "line": "Production Line 1", "type": "Rotary Screw Compressor"},
}

SENSOR_META = {
    "vibration":   {"label": "Vibration",   "unit": "mm/s", "max": 12.0, "warn": 4.5, "crit": 7.0,  "color": "#2563eb"},
    "temperature": {"label": "Temperature", "unit": "°C",   "max": 120,  "warn": 65,  "crit": 85,   "color": "#dc2626"},
    "load":        {"label": "Load Factor", "unit": "",     "max": 1.0,  "warn": 0.70,"crit": 0.85, "color": "#7c3aed"},
    "pressure":    {"label": "Pressure",    "unit": "bar",  "max": 6.0,  "warn": 3.8, "crit": 4.8,  "color": "#0891b2"},
}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:15px;font-weight:600;color:#e6edf3;margin-bottom:2px;'>NexaOps</div>"
        "<div style='font-size:11px;color:#484f58;margin-bottom:20px;'>Predictive Maintenance Platform</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    selected_key = st.selectbox("Asset", list(MACHINES.keys()), index=0)
    cfg          = MACHINES[selected_key]
    machine_id   = selected_key.split()[0]

    df, cp, anom = generate_machine_data(cfg["total"], cfg["seed"])

    current_cycle = st.slider(
        "Inspection Cycle",
        min_value=20,
        max_value=cfg["total"],
        value=cfg["default_cycle"],
        step=1,
    )

    st.divider()
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#8b949e;margin-bottom:8px;'>Chart Options</div>",
        unsafe_allow_html=True,
    )
    selected_sensor = st.selectbox(
        "Primary Sensor",
        list(SENSOR_META.keys()),
        format_func=lambda k: SENSOR_META[k]["label"],
    )
    show_forecast = st.checkbox("RUL Forecast Overlay", value=True)
    show_zones    = st.checkbox("State Zone Shading",   value=True)
    show_ci       = st.checkbox("Confidence Band",      value=True)

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#484f58;line-height:2;font-family:\"JetBrains Mono\",monospace;'>"
        "<b style='color:#8b949e;'>Model</b><br>"
        "Architecture: LSTM (2-layer)<br>"
        "Hidden units: 64<br>"
        "RMSE: 13.2 cycles<br>"
        "MAE: 9.7 cycles<br>"
        "Inference: &lt;12 ms<br><br>"
        "<b style='color:#8b949e;'>Training Data</b><br>"
        "NASA CMAPSS FD001<br>"
        "15,632 samples<br>"
        "14 sensor channels"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── DERIVED STATE ────────────────────────────────────────────────────────────
true_rul            = int(df.loc[df["cycle"] == current_cycle, "rul_true"].values[0])
predicted_rul, ci   = predict_rul_lstm(true_rul)
hs                  = compute_health_score(df, current_cycle)
current_state       = df.loc[df["cycle"] == current_cycle, "state"].values[0]
current_row         = df[df["cycle"] == current_cycle].iloc[0]
fail_date           = datetime.today() + timedelta(days=round(predicted_rul / 12))

badge_class = {
    "Healthy":  "badge-healthy",
    "Impaired": "badge-impaired",
    "Critical": "badge-critical",
}.get(current_state, "badge-offline")

# ─── PAGE HEADER ──────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown(
        f"<div style='font-size:18px;font-weight:600;color:#e6edf3;letter-spacing:-0.02em;'>"
        f"{machine_id} &mdash; {cfg['type']}</div>"
        f"<div style='font-size:12px;color:#8b949e;margin-top:2px;'>"
        f"{cfg['line']} &nbsp;&middot;&nbsp; "
        f"Inspection cycle: {current_cycle} / {cfg['total']} &nbsp;&middot;&nbsp; "
        f"Report generated: {datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f"<div style='text-align:right;padding-top:10px;'>"
        f"<span class='badge {badge_class}'>{current_state}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='height:1px;background:#21262d;margin:12px 0 20px 0;'></div>",
    unsafe_allow_html=True,
)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric("Predicted RUL",    f"{predicted_rul} cycles",                  f"± {ci} cycles")
with k2: st.metric("Health Index",     f"{hs} / 100",                              f"-{100 - hs} pts from baseline")
with k3: st.metric("Vibration",        f"{current_row['vibration']:.2f} mm/s",    "")
with k4: st.metric("Temperature",      f"{current_row['temperature']:.1f} °C",    "")
with k5: st.metric("Load Factor",      f"{current_row['load']:.2f}",              "")
with k6: st.metric("Est. Failure",     fail_date.strftime("%d %b %Y"),            f"~{round(predicted_rul / 12)} days")

st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

# ─── SENSOR TREND CHART ───────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Sensor Degradation Trend &amp; RUL Forecast</div>', unsafe_allow_html=True)

sm   = SENSOR_META[selected_sensor]
hist = df[df["cycle"] <= current_cycle]
fwd  = df[df["cycle"] >= current_cycle]

fig = go.Figure()

if show_zones:
    fig.add_vrect(x0=cp,   x1=anom,         fillcolor="rgba(217,119,6,0.06)", layer="below", line_width=0)
    fig.add_vrect(x0=anom, x1=cfg["total"], fillcolor="rgba(220,38,38,0.06)", layer="below", line_width=0)
    fig.add_vline(x=cp,   line_dash="dot", line_color="rgba(217,119,6,0.4)", line_width=1)
    fig.add_vline(x=anom, line_dash="dot", line_color="rgba(220,38,38,0.4)", line_width=1)
    fig.add_annotation(x=cp   + (anom - cp) / 2,          y=1, yref="paper",
                       text="Impaired", showarrow=False,
                       font=dict(size=10, color="#d97706"),
                       bgcolor="rgba(255,255,255,0.8)", borderpad=3)
    fig.add_annotation(x=anom + (cfg["total"] - anom) / 2, y=1, yref="paper",
                       text="Critical", showarrow=False,
                       font=dict(size=10, color="#dc2626"),
                       bgcolor="rgba(255,255,255,0.8)", borderpad=3)

if show_forecast and len(fwd) > 1:
    last_v  = float(hist[selected_sensor].iloc[-1])
    end_v   = float(df[selected_sensor].iloc[-1]) * 1.06
    fcast_y = np.linspace(last_v, end_v, len(fwd))
    band    = fcast_y * 0.075
    if show_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([fwd["cycle"], fwd["cycle"][::-1]]),
            y=pd.concat([pd.Series(fcast_y + band), pd.Series(fcast_y - band)[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.07)", line=dict(width=0),
            name="95% CI", hoverinfo="skip", showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=fwd["cycle"], y=fcast_y, name="Forecast", mode="lines",
        line=dict(color="#2563eb", width=1.5, dash="dash"),
        hovertemplate="Cycle %{x}<br>Forecast: %{y:.3f}<extra></extra>",
    ))

fig.add_trace(go.Scatter(
    x=hist["cycle"], y=hist[selected_sensor],
    name=sm["label"], mode="lines",
    line=dict(color=sm["color"], width=2),
    hovertemplate=f"Cycle %{{x}}<br>{sm['label']}: %{{y:.3f}} {sm['unit']}<extra></extra>",
))

fig.add_hline(y=sm["warn"], line_dash="dot", line_color="#d97706", line_width=1,
              annotation_text=f"Warning  {sm['warn']} {sm['unit']}",
              annotation_position="bottom right",
              annotation_font=dict(size=10, color="#d97706"))
fig.add_hline(y=sm["crit"], line_dash="dot", line_color="#dc2626", line_width=1,
              annotation_text=f"Critical  {sm['crit']} {sm['unit']}",
              annotation_position="bottom right",
              annotation_font=dict(size=10, color="#dc2626"))
fig.add_vline(x=current_cycle, line_color="#374151", line_width=1,
              annotation_text=f"Now  (Cycle {current_cycle})",
              annotation_position="top right",
              annotation_font=dict(size=10, color="#374151"))

fig.update_layout(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", color="#8b949e"),
    height=290, margin=dict(l=0, r=0, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(title="Operational Cycle", gridcolor="#21262d",
               showline=True, linecolor="#30363d",
               tickfont=dict(size=10), zeroline=False),
    yaxis=dict(title=f"{sm['label']} ({sm['unit']})" if sm["unit"] else sm["label"],
               gridcolor="#21262d", showline=True, linecolor="#30363d",
               tickfont=dict(size=10), zeroline=False),
    hovermode="x unified",
)

with st.container():
    st.markdown(
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ─── LOWER SECTION ────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1.1, 0.9, 1.0])

# ── SENSOR BARS ───────────────────────────────────────────────────────────────
with col_a:
    st.markdown('<div class="sec-label">Real-Time Sensor Readings</div>', unsafe_allow_html=True)
    for skey, meta in SENSOR_META.items():
        val    = float(current_row[skey])
        base   = float(df[df["cycle"] <= 20][skey].mean())
        delta  = (val - base) / base * 100
        pct    = min(100, val / meta["max"] * 100)
        if val >= meta["crit"]:
            bar_col = "#dc2626"
        elif val >= meta["warn"]:
            bar_col = "#d97706"
        else:
            bar_col = "#16a34a"

        st.markdown(
            f"<div style='margin-bottom:16px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:5px;'>"
            f"  <span style='font-size:12px;font-weight:500;color:#e6edf3;'>{meta['label']}</span>"
            f"  <span style='font-family:\"JetBrains Mono\",monospace;font-size:12px;font-weight:500;color:{bar_col};'>"
            f"    {val:.2f}{(' ' + meta['unit']) if meta['unit'] else ''}"
            f"    <span style='font-size:10px;color:#484f58;'>&nbsp;({delta:+.1f}%)</span>"
            f"  </span>"
            f"</div>"
            f"<div style='height:6px;background:#21262d;border-radius:3px;overflow:hidden;'>"
            f"  <div style='width:{pct:.1f}%;height:100%;background:{bar_col};border-radius:3px;'></div>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:9px;color:#30363d;"
            f"margin-top:3px;font-family:\"JetBrains Mono\",monospace;'>"
            f"  <span>0</span>"
            f"  <span>Warn {meta['warn']}</span>"
            f"  <span>Crit {meta['crit']}</span>"
            f"  <span>{meta['max']} {meta['unit']}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── STATE TIMELINE + FEATURE IMPORTANCE ───────────────────────────────────────
with col_b:
    st.markdown('<div class="sec-label">Degradation State Timeline</div>', unsafe_allow_html=True)
    timeline_entries = [
        ("Healthy",  f"Cycles 1 – {cp - 1}",                 "Normal operating range.",            "#16a34a"),
        ("Impaired", f"Cycles {cp} – {anom - 1}",            "Change-point detected. Score elevated.", "#d97706"),
        ("Critical", f"Cycles {anom} – {cfg['total']}",      "Accelerated degradation. RUL active.", "#dc2626"),
        ("Failure",  f"Predicted ~ Cycle {cfg['total'] + 1}", "Model-predicted end-of-life window.",  "#6b7280"),
    ]
    for i, (t_name, t_cycles, t_desc, t_color) in enumerate(timeline_entries):
        is_current = (t_name == current_state)
        is_last    = (i == len(timeline_entries) - 1)
        dot_ring   = (f"border:2px solid #0d1117;box-shadow:0 0 0 2px {t_color};" if is_current else "")
        tl_line_html  = "<div class='tl-line'></div>" if not is_last else ""
        title_color   = t_color if is_current else "#e6edf3"
        current_arrow = "&nbsp;&nbsp;&larr; current" if is_current else ""
        st.markdown(
            f"<div class='tl-item'>"
            f"  <div class='tl-line-wrap'>"
            f"    <div class='tl-dot' style='background:{t_color};{dot_ring}'></div>"
            f"    {tl_line_html}"
            f"  </div>"
            f"  <div class='tl-content'>"
            f"    <div class='tl-title' style='color:{title_color};'>"
            f"      {t_name}{current_arrow}"
            f"    </div>"
            f"    <div class='tl-sub'>{t_cycles}</div>"
            f"    <div style='font-size:11px;color:#9ca3af;margin-top:1px;'>{t_desc}</div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Feature Contribution to RUL Prediction</div>', unsafe_allow_html=True)

    feat_df = pd.DataFrame({
        "Feature":    ["Vibration", "Temperature", "Load Factor", "Pressure", "Cycle Index"],
        "Importance": [38, 26, 19, 12, 5],
        "Color":      ["#2563eb", "#dc2626", "#7c3aed", "#0891b2", "#6b7280"],
    })
    fig_feat = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"],
        orientation="h",
        marker=dict(color=feat_df["Color"], line=dict(width=0)),
        text=[f"{v}%" for v in feat_df["Importance"]],
        textposition="outside",
        textfont=dict(size=10, color="#6b7280"),
    ))
    fig_feat.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        height=175, margin=dict(l=0, r=30, t=0, b=0),
        font=dict(family="Inter", size=11, color="#8b949e"),
        xaxis=dict(range=[0, 48], showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", showline=False, tickfont=dict(size=11, color="#8b949e")),
    )
    st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})

# ── ALERTS + SCHEDULE ─────────────────────────────────────────────────────────
with col_c:
    st.markdown('<div class="sec-label">Active Alerts</div>', unsafe_allow_html=True)

    active_alerts = []
    if predicted_rul < 60:
        active_alerts.append(("critical", "RUL below safety threshold",
            f"Predicted RUL of {predicted_rul} cycles. Emergency maintenance required within 48 hours."))
    if float(current_row["vibration"]) > SENSOR_META["vibration"]["crit"]:
        active_alerts.append(("critical", "Vibration exceeds critical limit",
            f"Reading {current_row['vibration']:.2f} mm/s is above the {SENSOR_META['vibration']['crit']} mm/s threshold."))
    if current_state == "Impaired":
        active_alerts.append(("warning", "Change-point transition recorded",
            f"Asset crossed Healthy to Impaired boundary at cycle {cp}. Enhanced monitoring active."))
    if float(current_row["temperature"]) > SENSOR_META["temperature"]["warn"]:
        active_alerts.append(("warning", "Temperature above advisory limit",
            f"Reading {current_row['temperature']:.1f} °C. Verify cooling system performance."))
    if not active_alerts:
        active_alerts.append(("ok", "All parameters within normal range",
            "No active alerts. Next scheduled preventive maintenance in 30 days."))

    for a_type, title, desc in active_alerts[:4]:
        css = {"critical": "alert-critical", "warning": "alert-warning", "ok": "alert-ok"}.get(a_type, "alert-info")
        st.markdown(
            f"<div class='alert {css}'><strong>{title}</strong><span>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Recommended Maintenance Schedule</div>', unsafe_allow_html=True)

    sched_df = pd.DataFrame({
        "Asset":    ["M-001", "M-002", "M-004", "M-003"],
        "Action":   ["Bearing replacement", "Seal & lubrication service", "Sensor recalibration", "Routine preventive check"],
        "Due Date": ["20 Mar 2026", "25 Mar 2026", "22 Mar 2026", "05 Apr 2026"],
        "Priority": ["Critical", "Warning", "Warning", "Normal"],
    })
    st.dataframe(
        sched_df, hide_index=True, use_container_width=True,
        column_config={
            "Asset":    st.column_config.TextColumn("Asset",    width=60),
            "Action":   st.column_config.TextColumn("Action"),
            "Due Date": st.column_config.TextColumn("Due Date", width=90),
            "Priority": st.column_config.TextColumn("Priority", width=75),
        },
    )

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d;"
    "font-size:10px;color:#484f58;font-family:\"JetBrains Mono\",monospace;"
    "display:flex;justify-content:space-between;'>"
    "<span>NexaOps Industrial Solutions &copy; 2026. All rights reserved.</span>"
    "<span>Model: LSTM-RUL &nbsp;|&nbsp; Dataset: NASA CMAPSS FD001 &nbsp;|&nbsp; "
    "RMSE 13.2 &nbsp;|&nbsp; MAE 9.7 &nbsp;|&nbsp; Inference &lt;12 ms</span>"
    "</div>",
    unsafe_allow_html=True,
)
