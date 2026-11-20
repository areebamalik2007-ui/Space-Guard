"""SpaceGuard: Real-Time Space Weather Monitoring Dashboard."""

from datetime import datetime, timezone
import streamlit as st
import pandas as pd

from src.charts import cme_speed_chart, daily_counts, flare_timeline, kp_chart
from src.data import get_api_key, get_space_weather
from src.risk import assess_risk

# Streamlit Page Configuration
st.set_page_config(
    page_title="SpaceGuard | Space Weather Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Mission Control Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F1A;
        color: #E5E7EB;
    }
    .metric-card {
        background: #121A2B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #22D3EE;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9CA3AF;
        margin-top: 4px;
    }
    .risk-banner {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        border-left: 5px solid;
    }
    .risk-low {
        background: rgba(34, 211, 238, 0.1);
        border-color: #22D3EE;
    }
    .risk-moderate {
        background: rgba(245, 158, 11, 0.12);
        border-color: #F59E0B;
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        border-color: #EF4444;
    }
    .risk-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-low { background: #22D3EE; color: #0B0F1A; }
    .badge-moderate { background: #F59E0B; color: #0B0F1A; }
    .badge-high { background: #EF4444; color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def load_data(days: int) -> dict[str, pd.DataFrame]:
    """Fetch space weather datasets from NASA DONKI with caching."""
    return get_space_weather(days=days)


def main() -> None:
    # Sidebar
    st.sidebar.title("🛡️ SpaceGuard")
    st.sidebar.caption("Real-Time Space Weather Mission Control")
    st.sidebar.markdown("---")

    days = st.sidebar.slider(
        "Lookback Window (Days)",
        min_value=3,
        max_value=30,
        value=7,
        step=1,
        help="Number of days of NASA DONKI events to fetch and visualize.",
    )

    risk_hours = st.sidebar.selectbox(
        "Risk Assessment Window",
        options=[24, 48, 72],
        index=2,
        format_func=lambda h: f"Last {h} Hours",
        help="Time window for evaluating current operational risk.",
    )

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    api_key = get_api_key()
    is_live_key = bool(api_key and api_key != "DEMO_KEY")
    st.sidebar.markdown(
        f"**NASA API:** {'🟢 Active Key' if is_live_key else '🟡 Demo Key'}"
    )
    st.sidebar.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    # Header
    st.title("🛰️ SpaceGuard: Space Weather Dashboard")
    st.markdown(
        "Real-time monitoring of **Solar Flares (FLR)**, **Coronal Mass Ejections (CME)**, "
        "and **Geomagnetic Storms (GST)** via NASA DONKI."
    )

    # Fetch Data
    with st.spinner("Fetching NASA DONKI space weather data..."):
        try:
            data = load_data(days)
        except Exception as e:
            st.error(f"Error retrieving space weather data: {e}")
            return

    flares = data.get("flares", pd.DataFrame())
    cmes = data.get("cmes", pd.DataFrame())
    storms = data.get("storms", pd.DataFrame())

    # Risk Assessment
    risk = assess_risk(flares, cmes, storms, hours=risk_hours)
    level = risk.get("level", "Low")
    reasons = risk.get("reasons", [])

    # Risk Alert Banner
    level_lower = level.lower()
    badge_class = f"badge-{level_lower}"
    banner_class = f"risk-{level_lower}"

    st.markdown(
        f"""
        <div class="risk-banner {banner_class}">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 1.1rem; font-weight: 600;">Current Space Weather Threat Level (Last {risk_hours}h)</span>
                <span class="risk-badge {badge_class}">{level} Risk</span>
            </div>
            <ul style="margin-bottom: 0; padding-left: 20px; color: #E5E7EB;">
                {''.join(f'<li>{r}</li>' for r in reasons)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    # Flares metric
    flare_count = len(flares)
    strongest_flare = "None"
    if not flares.empty and "class_type" in flares.columns:
        m_x_flares = flares[flares["class_letter"].isin(["X", "M"])]
        if not m_x_flares.empty:
            strongest_flare = m_x_flares.iloc[0]["class_type"]
        else:
            strongest_flare = flares.iloc[0]["class_type"]

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Solar Flares ({days}d)</div>
                <div class="metric-value">{flare_count}</div>
                <div class="metric-sub">Max: {strongest_flare}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CME metric
    cme_count = len(cmes)
    earth_cme_count = len(cmes[cmes["is_earth_directed"]]) if not cmes.empty and "is_earth_directed" in cmes.columns else 0
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Coronal Ejections</div>
                <div class="metric-value">{cme_count}</div>
                <div class="metric-sub">Earth-Directed: {earth_cme_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Max Kp Metric
    max_kp_val = "Quiet (<5)"
    if not storms.empty and "max_kp" in storms.columns:
        highest_kp = storms["max_kp"].max()
        if pd.notna(highest_kp):
            max_kp_val = f"Kp {highest_kp:.1f}"

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Geomagnetic Storms</div>
                <div class="metric-value">{len(storms)}</div>
                <div class="metric-sub">Peak: {max_kp_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Threat Status
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Operational Status</div>
                <div class="metric-value" style="color: {'#EF4444' if level=='High' else ('#F59E0B' if level=='Moderate' else '#22D3EE')};">{level}</div>
                <div class="metric-sub">{len(reasons)} Active Indicator(s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Grid
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Activity Overview",
        "☀️ Solar Flares",
        "💥 Coronal Mass Ejections",
        "🌐 Geomagnetic Storms",
    ])

    with tab1:
        st.plotly_chart(daily_counts(flares, cmes, storms), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(flare_timeline(flares), use_container_width=True)
        with col_b:
            st.plotly_chart(cme_speed_chart(cmes), use_container_width=True)

    with tab2:
        st.plotly_chart(flare_timeline(flares), use_container_width=True)
        with st.expander("📄 View Solar Flares Raw Data"):
            if not flares.empty:
                st.dataframe(flares, use_container_width=True)
            else:
                st.info("No solar flare records in selected range.")

    with tab3:
        st.plotly_chart(cme_speed_chart(cmes), use_container_width=True)
        with st.expander("📄 View CME Raw Data"):
            if not cmes.empty:
                st.dataframe(cmes, use_container_width=True)
            else:
                st.info("No CME records in selected range.")

    with tab4:
        st.plotly_chart(kp_chart(storms), use_container_width=True)
        with st.expander("📄 View Geomagnetic Storms Raw Data"):
            if not storms.empty:
                st.dataframe(storms, use_container_width=True)
            else:
                st.info("No geomagnetic storm records in selected range.")


if __name__ == "__main__":
    main()
