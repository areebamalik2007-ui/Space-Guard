"""Space weather risk scoring and classification algorithms for SpaceGuard."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

# High risk thresholds
HIGH_KP_THRESHOLD = 7.0
HIGH_CME_SPEED_THRESHOLD = 1000.0  # km/s (must also be Earth-directed)
HIGH_FLARE_CLASSES = ("X",)

# Moderate risk thresholds
MOD_KP_MIN = 5.0
MOD_KP_MAX = 7.0
MOD_CME_SPEED_THRESHOLD = 500.0  # km/s
MOD_FLARE_CLASSES = ("M",)


def _to_utc_series(series: pd.Series) -> pd.Series:
    """Ensure datetime series is in UTC timezone."""
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def _format_time(dt: Any) -> str:
    """Format timestamp to UTC string format YYYY-MM-DD HH:MM UTC."""
    if pd.isna(dt):
        return "unknown time"
    ts = pd.to_datetime(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def assess_risk(
    flares: pd.DataFrame,
    cmes: pd.DataFrame,
    storms: pd.DataFrame,
    hours: int = 72,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess space weather risk level based on events within the last N hours.

    Rules:
    - High: any X-class flare, OR max Kp >= 7, OR Earth-directed CME with speed >= 1000 km/s.
    - Moderate: any M-class flare, OR max Kp >= 5 (below 7), OR any CME with speed >= 500 km/s.
    - Low: Otherwise.

    Args:
        flares: DataFrame with flares (peak_time/begin_time, class_type, class_letter).
        cmes: DataFrame with CMEs (start_time, speed, is_earth_directed).
        storms: DataFrame with geomagnetic storms (start_time, max_kp).
        hours: Evaluation lookback window in hours (default: 72).
        now: Optional reference UTC timestamp (defaults to current UTC time).

    Returns:
        dict with 'level' ("Low", "Moderate", "High") and 'reasons' (list of strings).
    """
    if now is None:
        ref_time = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        ref_time = now.replace(tzinfo=timezone.utc)
    else:
        ref_time = now.astimezone(timezone.utc)

    cutoff = ref_time - timedelta(hours=hours)

    high_reasons: list[str] = []
    mod_reasons: list[str] = []

    # 1. Evaluate Solar Flares
    if not flares.empty:
        time_series = flares["peak_time"] if "peak_time" in flares.columns else flares["begin_time"]
        times_utc = _to_utc_series(time_series)
        recent_flares = flares[times_utc >= cutoff]

        for _, row in recent_flares.iterrows():
            class_type = str(row.get("class_type") or "").strip()
            class_letter = str(row.get("class_letter") or (class_type[0] if class_type else "")).upper()
            flare_time = _format_time(row.get("peak_time") if pd.notna(row.get("peak_time")) else row.get("begin_time"))

            if class_letter in HIGH_FLARE_CLASSES:
                high_reasons.append(f"{class_type} flare at {flare_time} (rule: X-class flare)")
            elif class_letter in MOD_FLARE_CLASSES:
                mod_reasons.append(f"{class_type} flare at {flare_time} (rule: M-class flare)")

    # 2. Evaluate Geomagnetic Storms
    if not storms.empty and "start_time" in storms.columns and "max_kp" in storms.columns:
        times_utc = _to_utc_series(storms["start_time"])
        recent_storms = storms[times_utc >= cutoff]

        for _, row in recent_storms.iterrows():
            max_kp = row.get("max_kp")
            if pd.isna(max_kp):
                continue
            kp_val = float(max_kp)
            storm_time = _format_time(row.get("start_time"))

            if kp_val >= HIGH_KP_THRESHOLD:
                high_reasons.append(f"Geomagnetic storm with max Kp {kp_val:.1f} at {storm_time} (rule: max Kp >= 7)")
            elif kp_val >= MOD_KP_MIN:
                mod_reasons.append(f"Geomagnetic storm with max Kp {kp_val:.1f} at {storm_time} (rule: max Kp >= 5)")

    # 3. Evaluate CMEs
    if not cmes.empty and "start_time" in cmes.columns:
        times_utc = _to_utc_series(cmes["start_time"])
        recent_cmes = cmes[times_utc >= cutoff]

        for _, row in recent_cmes.iterrows():
            speed_raw = row.get("speed")
            if pd.isna(speed_raw):
                continue
            speed = float(speed_raw)
            is_earth = bool(row.get("is_earth_directed"))
            cme_time = _format_time(row.get("start_time"))

            if is_earth and speed >= HIGH_CME_SPEED_THRESHOLD:
                high_reasons.append(
                    f"Earth-directed CME with speed {speed:.0f} km/s at {cme_time} "
                    f"(rule: Earth-directed CME speed >= {HIGH_CME_SPEED_THRESHOLD:.0f} km/s)"
                )
            elif speed >= MOD_CME_SPEED_THRESHOLD:
                prefix = "Earth-directed CME" if is_earth else "CME"
                mod_reasons.append(
                    f"{prefix} with speed {speed:.0f} km/s at {cme_time} "
                    f"(rule: CME speed >= {MOD_CME_SPEED_THRESHOLD:.0f} km/s)"
                )

    # Determine overall risk level
    if high_reasons:
        level = "High"
        reasons = high_reasons + mod_reasons
    elif mod_reasons:
        level = "Moderate"
        reasons = mod_reasons
    else:
        level = "Low"
        reasons = [f"No flares M-class or above, no storm, no fast CMEs in the last {hours} hours"]

    return {"level": level, "reasons": reasons}
