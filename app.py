"""SpaceGuard: Real-Time Space Weather Monitoring Dashboard."""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    cme_speed_chart,
    daily_counts,
    earth_globe,
    flare_timeline,
    kp_chart,
    sun_disc,
)
from src.data import get_space_weather
from src.risk import assess_risk

# Streamlit Page Configuration
st.set_page_config(
    layout="wide",
    page_title="SpaceGuard",
    page_icon="🛰️",
)

# Custom Mission Control CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F1A;
        color: #E5E7EB;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 211, 238, 0.12);
        border: 1px solid #22D3EE;
        color: #22D3EE;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22D3EE;
        box-shadow: 0 0 8px #22D3EE;
    }
    .disclaimer-box {
        background: rgba(18, 26, 43, 0.6);
        border-left: 3px solid #22D3EE;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #9CA3AF;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #121A2B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.78rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #22D3EE;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #9CA3AF;
        margin-top: 3px;
    }
    .risk-container {
        background: #121A2B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 24px;
    }
    .badge-box {
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-align: center;
        min-width: 170px;
        text-transform: uppercase;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        color: #10B981;
    }
    .badge-moderate {
        background: rgba(245, 158, 11, 0.15);
        border: 2px solid #F59E0B;
        color: #F59E0B;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.18);
        border: 2px solid #EF4444;
        color: #EF4444;
    }
    .reasons-list {
        flex: 1;
        margin: 0;
        padding-left: 20px;
        color: #E5E7EB;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_cached_weather(days: int) -> dict[str, pd.DataFrame]:
    """Fetch space weather data with a 15-minute Streamlit cache."""
    return get_space_weather(days=days)


def _format_table(df: pd.DataFrame, col_map: dict[str, str], time_cols: list[str]) -> pd.DataFrame:
    """Format DataFrame for clean user-facing table display."""
    if df.empty:
        return pd.DataFrame(columns=list(col_map.values()))
    out = df.copy()
    for col in time_cols:
        if col in out.columns and pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M UTC")
    available_cols = [c for c in col_map.keys() if c in out.columns]
    out = out[available_cols].rename(columns=col_map)
    return out


def main() -> None:
    # 1. Sidebar
    st.sidebar.title("🛡️ SpaceGuard")
    st.sidebar.caption("NASA DONKI Telemetry Console")
    st.sidebar.markdown("---")

    days = st.sidebar.slider(
        "Days to look back",
        min_value=1,
        max_value=30,
        value=7,
        help="Historical range of events queried from the NASA DONKI API.",
    )

    event_types = st.sidebar.multiselect(
        "Event types",
        options=["Solar Flares", "Coronal Mass Ejections", "Geomagnetic Storms"],
        default=["Solar Flares", "Coronal Mass Ejections", "Geomagnetic Storms"],
        help="Choose event types to display in charts and tables.",
    )

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("ℹ️ How risk is calculated"):
        st.markdown(
            """
            | Level | Rule Threshold (Last 72 Hours) |
            | :--- | :--- |
            | **High** | Any **X-class flare**, OR max **Kp $\ge$ 7**, OR **Earth-directed CME $\ge$ 1000 km/s** |
            | **Moderate** | Any **M-class flare**, OR max **Kp $\ge$ 5** (below 7), OR any **CME $\ge$ 500 km/s** |
            | **Low** | All activity below Moderate thresholds |
            """
        )

    # 2. Header
    now_utc = datetime.now(timezone.utc)
    st.markdown(
        f"""
        <div class="header-container">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">
                    SpaceGuard <span style="font-weight: 400; font-size: 1.25rem; color: #9CA3AF;">· Space weather monitoring</span>
                </h1>
            </div>
            <div class="live-pill">
                <div class="live-dot"></div>
                LIVE · last updated {now_utc.strftime('%H:%M')} UTC
            </div>
        </div>
        <div class="disclaimer-box">
            ⚠️ <b>Notice:</b> Student prototype using NASA DONKI data. Not an official forecast.
            See NOAA SWPC for official alerts. DONKI entries can lag by hours.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Data Fetching
    with st.spinner("Connecting to NASA DONKI and retrieving space weather telemetry..."):
        try:
            data = fetch_cached_weather(days=days)
        except Exception as err:
            st.error(f"⚠️ Failed to connect to NASA space weather services: {err}")
            data = {"flares": pd.DataFrame(), "cmes": pd.DataFrame(), "storms": pd.DataFrame()}

    flares_raw = data.get("flares", pd.DataFrame())
    cmes_raw = data.get("cmes", pd.DataFrame())
    storms_raw = data.get("storms", pd.DataFrame())

    if flares_raw.empty and cmes_raw.empty and storms_raw.empty:
        st.warning("No space weather events were recorded in this period or the API returned empty telemetry.")

    # Apply sidebar filter for display
    flares = flares_raw if "Solar Flares" in event_types else pd.DataFrame(columns=flares_raw.columns)
    cmes = cmes_raw if "Coronal Mass Ejections" in event_types else pd.DataFrame(columns=cmes_raw.columns)
    storms = storms_raw if "Geomagnetic Storms" in event_types else pd.DataFrame(columns=storms_raw.columns)

    # 4. Risk Assessment (fixed 72-hour window on full telemetry)
    risk = assess_risk(flares_raw, cmes_raw, storms_raw, hours=72)
    level = risk.get("level", "Low")
    reasons = risk.get("reasons", [])

    badge_class = f"badge-{level.lower()}"
    reasons_html = "".join(f"<li>{r}</li>" for r in reasons)

    st.markdown(
        f"""
        <div class="risk-container">
            <div class="badge-box {badge_class}">
                {level} Risk
            </div>
            <ul class="reasons-list">
                {reasons_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Operational threat level evaluated over a strict 72-hour lookback window regardless of the sidebar display range.")

    # 5. Top Row Metrics Cards
    # Flares
    flare_count = len(flares)
    strongest_flare = "None"
    if not flares.empty and "class_type" in flares.columns:
        m_x = flares[flares["class_letter"].isin(["X", "M"])]
        strongest_flare = m_x.iloc[0]["class_type"] if not m_x.empty else flares.iloc[0]["class_type"]

    # CMEs
    cme_count = len(cmes)
    earth_cmes = len(cmes[cmes["is_earth_directed"]]) if (not cmes.empty and "is_earth_directed" in cmes.columns) else 0

    # Max Kp
    max_kp_str = "Quiet (<5)"
    highest_kp = None
    if not storms.empty and "max_kp" in storms.columns:
        valid_kp = storms["max_kp"].dropna()
        if not valid_kp.empty:
            highest_kp = float(valid_kp.max())
            max_kp_str = f"Kp {highest_kp:.1f}"

    # Fastest CME
    fastest_cme_str = "N/A"
    if not cmes.empty and "speed" in cmes.columns:
        valid_speeds = cmes["speed"].dropna()
        if not valid_speeds.empty:
            fastest_cme_str = f"{valid_speeds.max():.0f} km/s"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Solar Flares</div>
                <div class="metric-value">{flare_count}</div>
                <div class="metric-subtitle">Peak: {strongest_flare}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Coronal Ejections (CME)</div>
                <div class="metric-value">{cme_count}</div>
                <div class="metric-subtitle">Earth-Directed: {earth_cmes}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Geomagnetic Storm Peak</div>
                <div class="metric-value">{max_kp_str}</div>
                <div class="metric-subtitle">Events Cataloged: {len(storms)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Fastest CME Speed</div>
                <div class="metric-value">{fastest_cme_str}</div>
                <div class="metric-subtitle">Lookback: {days} Days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Middle Row (Two Columns: Earth Globe & Solar Disc)
    col_globe, col_sun = st.columns(2)
    with col_globe:
        st.plotly_chart(
            earth_globe(highest_kp),
            use_container_width=True,
            key="middle_earth_globe",
        )
        st.caption("🌐 **Earth Auroral Oval:** Boundary estimated from peak Kp index (illustrative approximation, not a physics model).")

    with col_sun:
        st.plotly_chart(
            sun_disc(flares),
            use_container_width=True,
            key="middle_sun_disc",
        )
        st.caption("☀️ **Solar Disc:** Active flare eruption sites mapped onto the visible Earth-facing solar hemisphere.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. Bottom Tabs
    tab_overview, tab_flares, tab_cmes, tab_storms = st.tabs([
        "📊 Overview",
        "☀️ Flares",
        "💥 CMEs",
        "🌐 Geomagnetic Storms",
    ])

    with tab_overview:
        st.plotly_chart(
            daily_counts(flares, cmes, storms),
            use_container_width=True,
            key="overview_daily_counts",
        )
        st.plotly_chart(
            flare_timeline(flares),
            use_container_width=True,
            key="overview_flare_timeline",
        )

    with tab_flares:
        st.plotly_chart(
            flare_timeline(flares),
            use_container_width=True,
            key="tab_flares_timeline",
        )
        with st.expander("📄 View Solar Flares Clean Data Table"):
            clean_flares = _format_table(
                flares,
                {
                    "begin_time": "Begin Time",
                    "peak_time": "Peak Time",
                    "class_type": "Class Type",
                    "source_location": "Source Location",
                },
                ["begin_time", "peak_time"],
            )
            if not clean_flares.empty:
                st.dataframe(clean_flares, use_container_width=True, hide_index=True)
            else:
                st.info("No solar flare records in the selected window.")

    with tab_cmes:
        st.plotly_chart(
            cme_speed_chart(cmes),
            use_container_width=True,
            key="tab_cmes_speed",
        )
        with st.expander("📄 View Coronal Mass Ejections Clean Data Table"):
            clean_cmes = _format_table(
                cmes,
                {
                    "start_time": "Start Time",
                    "speed": "Speed (km/s)",
                    "type": "CME Type",
                    "is_earth_directed": "Earth-Directed",
                },
                ["start_time"],
            )
            if not clean_cmes.empty:
                st.dataframe(clean_cmes, use_container_width=True, hide_index=True)
            else:
                st.info("No CME records in the selected window.")

    with tab_storms:
        st.plotly_chart(
            kp_chart(storms),
            use_container_width=True,
            key="tab_storms_kp",
        )
        with st.expander("📄 View Geomagnetic Storms Clean Data Table"):
            clean_storms = _format_table(
                storms,
                {
                    "start_time": "Start Time",
                    "max_kp": "Max Kp Index",
                },
                ["start_time"],
            )
            if not clean_storms.empty:
                st.dataframe(clean_storms, use_container_width=True, hide_index=True)
            else:
                st.info("No geomagnetic storm records in the selected window.")


if __name__ == "__main__":
    main()
