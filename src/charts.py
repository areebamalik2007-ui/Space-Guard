"""Plotly chart and timeline generation for space weather events."""

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Theme constants
CARD_BG = "#121A2B"
ACCENT_CYAN = "#22D3EE"
TEXT_COLOR = "#E5E7EB"
MUTED_TEXT = "#9CA3AF"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
LINE_COLOR = "rgba(255, 255, 255, 0.15)"

FLARE_COLORS = {
    "X": "#EF4444",  # Red
    "M": "#F97316",  # Orange
    "C": "#3B82F6",  # Blue
    "B": "#06B6D4",  # Cyan
    "A": "#94A3B8",  # Gray
}


def _apply_shared_layout(fig: go.Figure, title: str) -> None:
    """Apply a unified dark 'mission control' layout to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=TEXT_COLOR, size=15),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        margin=dict(l=45, r=30, t=55, b=45),
        hoverlabel=dict(
            bgcolor=CARD_BG,
            bordercolor=ACCENT_CYAN,
            font=dict(color=TEXT_COLOR, size=12),
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor="rgba(18, 26, 43, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        linecolor=LINE_COLOR,
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT_COLOR),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        linecolor=LINE_COLOR,
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT_COLOR),
    )


def _empty_figure(title: str) -> go.Figure:
    """Return an empty figure displaying a 'No data in this period' message."""
    fig = go.Figure()
    _apply_shared_layout(fig, title)
    fig.add_annotation(
        text="No data in this period",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=MUTED_TEXT),
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig


def _extract_magnitude(val: Any) -> float:
    """Extract numeric magnitude from flare class (e.g. 'M1.2' -> 1.2)."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(val))
    return float(match.group(1)) if match else 1.0


def flare_timeline(flares: pd.DataFrame) -> go.Figure:
    """Scatter timeline of solar flares over time with class letter and magnitude sizing."""
    title = "Solar Flare Activity Timeline"
    if flares.empty:
        return _empty_figure(title)

    time_col = "peak_time" if "peak_time" in flares.columns else "begin_time"
    valid = flares.dropna(subset=[time_col]).copy()
    if valid.empty:
        return _empty_figure(title)

    valid["class_letter"] = valid["class_type"].str[0].str.upper().fillna("C")
    valid["magnitude"] = valid["class_type"].apply(_extract_magnitude)
    valid["marker_size"] = np.clip(8 + valid["magnitude"] * 4, 8, 30)

    fig = go.Figure()

    # Add each class as a distinct trace for interactive legend toggle
    for letter in ["X", "M", "C", "B", "A"]:
        subset = valid[valid["class_letter"] == letter]
        if subset.empty:
            continue

        hover_text = [
            f"<b>{row['class_type']} Flare</b><br>"
            f"Peak: {row[time_col].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Location: {row.get('source_location') or 'N/A'}"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=subset[time_col],
                y=subset["class_letter"],
                mode="markers",
                name=f"{letter}-class",
                marker=dict(
                    size=subset["marker_size"],
                    color=FLARE_COLORS.get(letter, ACCENT_CYAN),
                    line=dict(width=1.5, color="white"),
                    opacity=0.9,
                ),
                text=hover_text,
                hoverinfo="text",
            )
        )

    _apply_shared_layout(fig, title)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["A", "B", "C", "M", "X"],
        title="Flare Class",
    )
    fig.update_xaxes(title="Time (UTC)")
    return fig


def cme_speed_chart(cmes: pd.DataFrame) -> go.Figure:
    """Scatter plot of CME speeds over time with Earth-directed highlights and reference lines."""
    title = "Coronal Mass Ejections (CME) Speed & Trajectory"
    if cmes.empty or "start_time" not in cmes.columns:
        return _empty_figure(title)

    valid = cmes.dropna(subset=["start_time", "speed"]).copy()
    if valid.empty:
        return _empty_figure(title)

    fig = go.Figure()

    # Normal CMEs
    normal = valid[~valid["is_earth_directed"]]
    if not normal.empty:
        normal_hover = [
            f"<b>CME ({row.get('type') or 'Unknown'})</b><br>"
            f"Start: {row['start_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Speed: {row['speed']:.0f} km/s<br>"
            f"Earth-directed: No"
            for _, row in normal.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=normal["start_time"],
                y=normal["speed"],
                mode="markers",
                name="CME (Other)",
                marker=dict(
                    size=9,
                    color=ACCENT_CYAN,
                    opacity=0.75,
                ),
                text=normal_hover,
                hoverinfo="text",
            )
        )

    # Earth-directed CMEs
    earth = valid[valid["is_earth_directed"]]
    if not earth.empty:
        earth_hover = [
            f"<b>Earth-Directed CME ({row.get('type') or 'Unknown'})</b><br>"
            f"Start: {row['start_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Speed: {row['speed']:.0f} km/s<br>"
            f"Earth-directed: <b>YES</b>"
            for _, row in earth.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=earth["start_time"],
                y=earth["speed"],
                mode="markers",
                name="Earth-Directed",
                marker=dict(
                    size=13,
                    symbol="diamond",
                    color="#EF4444",
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=1.0,
                ),
                text=earth_hover,
                hoverinfo="text",
            )
        )

    # Reference lines at 500 km/s and 1000 km/s
    fig.add_hline(
        y=500,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=1.5,
        annotation_text="500 km/s (Moderate)",
        annotation_position="top left",
        annotation_font_color="#F59E0B",
    )
    fig.add_hline(
        y=1000,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.5,
        annotation_text="1000 km/s (Severe)",
        annotation_position="top left",
        annotation_font_color="#EF4444",
    )

    _apply_shared_layout(fig, title)
    fig.update_xaxes(title="Time (UTC)")
    fig.update_yaxes(title="Speed (km/s)")
    return fig


def kp_chart(storms: pd.DataFrame) -> go.Figure:
    """Bar chart of max Kp index with colored bands at Kp 5 and Kp 7, maintaining 0-9 axis."""
    title = "Geomagnetic Storm Activity (Max Kp Index)"
    fig = go.Figure()

    # Reference bands for storm levels
    fig.add_hrect(
        y0=5,
        y1=7,
        fillcolor="#F59E0B",
        opacity=0.15,
        line_width=0,
        annotation_text="G1-G2 Storm (Kp ≥ 5)",
        annotation_position="top right",
        annotation_font_color="#F59E0B",
    )
    fig.add_hrect(
        y0=7,
        y1=9,
        fillcolor="#EF4444",
        opacity=0.15,
        line_width=0,
        annotation_text="G3-G5 Severe (Kp ≥ 7)",
        annotation_position="top right",
        annotation_font_color="#EF4444",
    )

    has_data = not storms.empty and "start_time" in storms.columns and not storms["max_kp"].dropna().empty

    if has_data:
        valid = storms.dropna(subset=["start_time", "max_kp"]).copy()
        colors = [
            "#EF4444" if kp >= 7 else ("#F59E0B" if kp >= 5 else ACCENT_CYAN)
            for kp in valid["max_kp"]
        ]
        hover_text = [
            f"<b>Geomagnetic Storm</b><br>"
            f"Start: {row['start_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Max Kp: {row['max_kp']:.2f}"
            for _, row in valid.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x=valid["start_time"],
                y=valid["max_kp"],
                name="Max Kp",
                marker_color=colors,
                text=hover_text,
                hoverinfo="text",
            )
        )
    else:
        fig.add_annotation(
            text="No data in this period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=15, color=MUTED_TEXT),
        )

    _apply_shared_layout(fig, title)
    fig.update_yaxes(range=[0, 9], dtick=1, title="Max Kp Index")
    fig.update_xaxes(title="Time (UTC)")
    return fig


def daily_counts(flares: pd.DataFrame, cmes: pd.DataFrame, storms: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of solar events per day."""
    title = "Daily Space Weather Events Count"

    # Extract dates
    flare_dates = (
        flares["peak_time"].dt.date
        if (not flares.empty and "peak_time" in flares.columns)
        else pd.Series(dtype="object")
    )
    cme_dates = (
        cmes["start_time"].dt.date
        if (not cmes.empty and "start_time" in cmes.columns)
        else pd.Series(dtype="object")
    )
    storm_dates = (
        storms["start_time"].dt.date
        if (not storms.empty and "start_time" in storms.columns)
        else pd.Series(dtype="object")
    )

    all_dates = sorted(set(flare_dates.dropna().unique()) | set(cme_dates.dropna().unique()) | set(storm_dates.dropna().unique()))

    if not all_dates:
        return _empty_figure(title)

    date_idx = pd.Index(all_dates)
    flare_counts = flare_dates.value_counts().reindex(date_idx, fill_value=0)
    cme_counts = cme_dates.value_counts().reindex(date_idx, fill_value=0)
    storm_counts = storm_dates.value_counts().reindex(date_idx, fill_value=0)

    x_vals = [d.strftime("%Y-%m-%d") for d in date_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals, y=flare_counts.values, name="Flares", marker_color="#F97316"))
    fig.add_trace(go.Bar(x=x_vals, y=cme_counts.values, name="CMEs", marker_color=ACCENT_CYAN))
    fig.add_trace(go.Bar(x=x_vals, y=storm_counts.values, name="Storms", marker_color="#A855F7"))

    _apply_shared_layout(fig, title)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Event Count", dtick=1)
    return fig
