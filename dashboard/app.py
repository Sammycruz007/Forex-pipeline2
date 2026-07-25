"""
Streamlit Trading Signal Dashboard
Visualises Forex chart + ML predictions from the FastAPI backend.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

API_BASE = "https://forex-signal-api-28dg.onrender.com"

st.set_page_config(
    page_title  = "Forex Signal Dashboard",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1c1c2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────

def wake_up_api():
    """
    Render free tier sleeps after 15 min inactivity.
    This pings the health endpoint first to wake it up.
    Shows a spinner while waiting.
    """
    import time
    try:
        r = requests.get(f"{API_BASE}/health", timeout=60)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=300)
def fetch_signal():
    try:
        r = requests.get(f"{API_BASE}/signal", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Cannot reach API: {e}. Is uvicorn running on port 8000?")
        return None


@st.cache_data(ttl=300)
def fetch_history(days: int = 120):
    try:
        r = requests.get(f"{API_BASE}/history?days={days}", timeout=60)
        r.raise_for_status()
        data = r.json()
        df   = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception as e:
        st.error(f"Cannot fetch history: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_importance():
    try:
        r = requests.get(f"{API_BASE}/features/importance", timeout=60)
        r.raise_for_status()
        return r.json()["features"]
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────

def _forecast_period_label(signal: dict) -> str:
    """
    Compute the actual business-day date range the signal covers,
    from as_of_date + horizon_days — replaces the old hardcoded
    'Week of May 26–30' placeholder text.
    """
    horizon_days = signal.get("horizon_days", 2)
    try:
        as_of = pd.to_datetime(signal["as_of_date"])
    except Exception:
        return f"{horizon_days}-Day Forecast"

    dates   = []
    current = as_of
    while len(dates) < horizon_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)

    start_label = dates[0].strftime("%b %d")
    end_label   = dates[-1].strftime("%b %d")
    if dates[0].month == dates[-1].month:
        return f"{dates[0].strftime('%b')} {dates[0].day}–{dates[-1].day}"
    return f"{start_label} – {end_label}"


def build_candlestick_chart(df: pd.DataFrame, signal: dict) -> go.Figure:
    """Candlestick chart with a single forecast zone spanning the model's horizon."""

    horizon_days = signal.get("horizon_days", 2)

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"EUR/USD — Daily Chart with {horizon_days}-Day Forecast"]
    )

    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["open"],
        high  = df["high"],
        low   = df["low"],
        close = df["close"],
        name  = "EURUSD",
        increasing_line_color = "#00ff88",
        decreasing_line_color = "#ff4444",
    ))

    last_date    = df.index[-1]
    last_price   = df["close"].iloc[-1]
    signal_dir   = signal["signal"]
    dn_prob      = signal["dn_probability"]
    up_prob      = signal["up_probability"]
    color        = "#00ff88" if signal_dir == "UP" else "#ff4444"
    fill_color   = "rgba(0,255,136,0.1)" if signal_dir == "UP" else "rgba(255,68,68,0.1)"

    # Generate forecast dates skipping weekends, spanning the model's
    # actual prediction horizon (previously hardcoded to 5 — the old
    # 5-day horizon)
    forecast_dates = []
    current        = last_date
    while len(forecast_dates) < horizon_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)

    end_date = forecast_dates[-1]

    # Single directional forecast line
    pip_move    = last_price * 0.008
    target_price = (
        last_price + pip_move if signal_dir == "UP"
        else last_price - pip_move
    )

    fig.add_trace(go.Scatter(
        x      = [last_date, end_date],
        y      = [last_price, target_price],
        mode   = "lines+markers",
        name   = f"{horizon_days}-Day Forecast ({signal_dir})",
        line   = dict(color=color, width=2, dash="dash"),
        marker = dict(size=8, color=color),
    ))

    # Forecast zone — shaded region for the week
    spread = last_price * 0.006
    fig.add_trace(go.Scatter(
        x          = [last_date, end_date, end_date, last_date],
        y          = [
            last_price + spread,
            target_price + spread,
            target_price - spread,
            last_price - spread
        ],
        fill       = "toself",
        fillcolor  = fill_color,
        line       = dict(color="rgba(255,255,255,0)"),
        name       = "Forecast Zone",
        showlegend = True,
    ))

    # Vertical line — scatter trace instead of add_vline
    fig.add_trace(go.Scatter(
        x    = [str(last_date.date()), str(last_date.date())],
        y    = [df["close"].min() * 0.999, df["close"].max() * 1.001],
        mode = "lines",
        name = "Forecast Start",
        line = dict(color="#888888", dash="dot", width=1),
        showlegend = False,
    ))

    # Annotation on the forecast
    confidence = signal["confidence"]
    prob       = max(up_prob, dn_prob)
    fig.add_annotation(
        x           = end_date,
        y           = target_price,
        text        = f"{'▲' if signal_dir == 'UP' else '▼'} {signal_dir} {prob*100:.1f}% — {confidence}",
        showarrow   = True,
        arrowhead   = 2,
        arrowcolor  = color,
        font        = dict(color=color, size=13),
        bgcolor     = "#1c1c2e",
        bordercolor = color,
        ax          = 40,
        ay          = -40
    )

    fig.update_layout(
        template                  = "plotly_dark",
        height                    = 540,
        showlegend                = True,
        xaxis_rangeslider_visible = False,
        paper_bgcolor             = "#0e1117",
        plot_bgcolor              = "#0e1117",
        font                      = dict(color="white"),
        legend                    = dict(
            bgcolor     = "#1c1c2e",
            bordercolor = "#2d2d44",
            borderwidth = 1
        ),
        margin = dict(l=10, r=10, t=40, b=10)
    )
    return fig


def build_probability_gauge(up_prob: float, dn_prob: float) -> go.Figure:
    """Gauge showing directional probability for the signal horizon."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = up_prob * 100,
        title = {"text": "UP Probability %", "font": {"color": "white"}},
        delta = {"reference": 50, "valueformat": ".1f"},
        gauge = {
            "axis"   : {"range": [0, 100], "tickcolor": "white"},
            "bar"    : {"color": "#00ff88" if up_prob > 0.5 else "#ff4444"},
            "bgcolor": "#1c1c2e",
            "steps"  : [
                {"range": [0,  40],  "color": "#ff4444"},
                {"range": [40, 60],  "color": "#ffd700"},
                {"range": [60, 100], "color": "#00ff88"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 3},
                "value": 50
            }
        },
        number = {"font": {"color": "white"}}
    ))
    fig.update_layout(
        template      = "plotly_dark",
        height        = 280,
        paper_bgcolor = "#0e1117",
        font          = dict(color="white"),
        margin        = dict(l=20, r=20, t=60, b=20)
    )
    return fig


def build_weekly_signal_card(signal: dict) -> go.Figure:
    """
    Single signal summary bar spanning the model's prediction horizon.
    Replaces the misleading 5-identical-bars chart.
    Shows one honest probability bar for the horizon window.
    """
    signal_dir   = signal["signal"]
    up_prob      = signal["up_probability"]
    dn_prob      = signal["dn_probability"]
    color_up     = "#00ff88"
    color_dn     = "#ff4444"
    period_label = _forecast_period_label(signal)

    fig = go.Figure()

    # UP probability bar
    fig.add_trace(go.Bar(
        x            = [period_label],
        y            = [up_prob * 100],
        name         = "UP %",
        marker_color = color_up,
        text         = [f"UP: {up_prob*100:.1f}%"],
        textposition = "inside",
        width        = 0.3,
    ))

    # DOWN probability bar
    fig.add_trace(go.Bar(
        x            = [period_label],
        y            = [dn_prob * 100],
        name         = "DOWN %",
        marker_color = color_dn,
        text         = [f"DOWN: {dn_prob*100:.1f}%"],
        textposition = "inside",
        width        = 0.3,
    ))

    fig.add_hline(
        y               = 50,
        line_dash       = "dash",
        line_color      = "white",
        annotation_text = "50% — no edge"
    )

    fig.update_layout(
        template      = "plotly_dark",
        title         = "Directional Probability — Single Signal",
        barmode       = "group",
        height        = 320,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
        yaxis         = dict(range=[0, 100], title="Probability %"),
        legend        = dict(bgcolor="#1c1c2e"),
        margin        = dict(l=10, r=10, t=40, b=10)
    )
    return fig


def build_importance_chart(features: list) -> go.Figure:
    names  = [f["feature"]    for f in features][::-1]
    values = [f["importance"] for f in features][::-1]

    fig = go.Figure(go.Bar(
        x            = values,
        y            = names,
        orientation  = "h",
        marker_color = "#7b68ee",
        text         = values,
        textposition = "outside"
    ))
    fig.update_layout(
        template      = "plotly_dark",
        title         = "Top 15 Feature Importances",
        height        = 450,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
        margin        = dict(l=10, r=10, t=40, b=10)
    )
    return fig


# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────

def main():
    st.markdown("""
        <h1 style='text-align:center; color:white;'>
            📈 Forex ML Signal Dashboard
        </h1>
        <p style='text-align:center; color:#888;'>
            EUR/USD · ML-Powered Macro Features · Directional Signal
        </p>
        <hr style='border-color:#2d2d44;'>
    """, unsafe_allow_html=True)

    # Wake up Render free tier if sleeping
    with st.spinner("Connecting to signal server... (may take 30-60s on first load)"):
        wake_up_api()

    # Fetch signal first — sidebar below needs it for live model/precision info
    signal = fetch_signal()

    if signal is None:
        st.error("Could not load data. Make sure the API is running.")
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        chart_days = st.slider("Chart history (days)", 30, 365, 120)
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        model_name       = signal.get("model_used", "Unknown")
        precision        = signal.get("precision_at_threshold")
        proba_threshold  = signal.get("proba_threshold", 0.65)
        precision_text   = f"{precision*100:.2f}%" if precision is not None else "N/A"
        horizon_days     = signal.get("horizon_days", 2)
        st.markdown(f"**Model:** {model_name}")
        st.markdown("**Features:** 52 (technical + macro)")
        st.markdown(f"**Precision (proba ≥ {proba_threshold:.2f}):** {precision_text}")
        st.markdown(f"**Signal frequency:** Every {horizon_days} trading day(s)")
        st.markdown("---")
        st.markdown("### 📖 How To Use")
        st.markdown(
            "1. Check signal every **Monday morning**\n"
            "2. **HIGH confidence** → trade with full size\n"
            "3. **MEDIUM confidence** → wait for 1hr confirmation\n"
            "4. **LOW confidence** → no trade\n"
            "5. Use 15min/1hr chart for entry timing"
        )
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown(
            "_ML research tool only. "
            "Not financial advice. "
            "Always use risk management._"
        )
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Fetch remaining data
    history_df = fetch_history(chart_days)
    importance = fetch_importance()

    if history_df is None:
        st.error("Could not load history data. Make sure the API is running.")
        st.stop()

    signal_dir   = signal["signal"]
    up_prob      = signal["up_probability"]
    dn_prob      = signal["dn_probability"]
    confidence   = signal["confidence"]
    price        = signal["current_price"]
    as_of        = signal["as_of_date"]
    horizon_days = signal.get("horizon_days", 2)
    period_label = _forecast_period_label(signal)
    color        = "#00ff88" if signal_dir == "UP" else "#ff4444"
    arrow        = "▲" if signal_dir == "UP" else "▼"

    # Signal banner
    st.markdown(f"""
    <div style='background:{color}22; border:2px solid {color};
                border-radius:12px; padding:20px; text-align:center; margin-bottom:20px;'>
        <span style='color:{color}; font-size:2.8em; font-weight:bold;'>
            {arrow} {signal_dir}
        </span>
        <span style='color:white; font-size:1.5em; margin-left:20px;'>
            {confidence} CONFIDENCE
        </span>
        <br><br>
        <span style='color:#aaa; font-size:1em;'>
            EUR/USD @ {price:.5f} · Signal as of {as_of} · Valid for {period_label}
        </span>
        <br>
        <span style='color:#666; font-size:0.85em;'>
            ⓘ Single {horizon_days}-day signal — not independent daily forecasts
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(f"Price ({as_of})",  f"{price:.5f}")
    with col2:
        st.metric(f"{horizon_days}-Day Signal", signal_dir)
    with col3:
        st.metric("UP Probability",  f"{up_prob*100:.1f}%")
    with col4:
        st.metric("DOWN Probability", f"{dn_prob*100:.1f}%")
    with col5:
        st.metric("Confidence",      confidence)

    st.markdown("---")

    # Main chart
    st.plotly_chart(
        build_candlestick_chart(history_df, signal),
        use_container_width=True
    )

    # Gauge + weekly signal card
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.plotly_chart(
            build_probability_gauge(up_prob, dn_prob),
            use_container_width=True
        )
    with col_right:
        st.plotly_chart(
            build_weekly_signal_card(signal),
            use_container_width=True
        )

    # Feature importance
    if importance:
        st.markdown("---")
        st.plotly_chart(
            build_importance_chart(importance),
            use_container_width=True
        )

    # Raw data expanders
    st.markdown("---")
    with st.expander("📋 Raw Signal JSON"):
        st.json(signal)
    with st.expander("📋 Recent Price Data"):
        st.dataframe(
            history_df.tail(10).sort_index(ascending=False),
            use_container_width=True
        )

    # Footer
    st.markdown(f"""
    <hr style='border-color:#2d2d44;'>
    <p style='text-align:center; color:#555; font-size:0.8em;'>
        Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ·
        Model: {signal['model_used']} ·
        Precision (≥{proba_threshold:.2f}): {precision_text} · Signal frequency: Every {horizon_days} trading day(s)
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
