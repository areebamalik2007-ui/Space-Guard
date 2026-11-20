"""NASA DONKI API client and data cleaning functions for SpaceGuard."""

from datetime import datetime, timedelta, timezone
import os
from typing import Any

import numpy as np
import pandas as pd
import requests

FLARE_CLASS_MAP = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}


def get_api_key(api_key: str | None = None) -> str:
    """Retrieve NASA API key from argument, environment, or default to DEMO_KEY."""
    if api_key:
        return api_key
    return os.getenv("NASA_API_KEY", "DEMO_KEY")


def fetch_donki_data(
    endpoint: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Fetch raw JSON records from NASA DONKI with fallback and error handling."""
    key = get_api_key(api_key)
    params = {"startDate": start_date, "endDate": end_date, "api_key": key}
    urls = [
        f"https://api.nasa.gov/DONKI/{endpoint}",
        f"https://ccmc.gsfc.nasa.gov/DONKI-API/get/{endpoint}",
    ]

    for url in urls:
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                return []
        except (requests.RequestException, ValueError):
            continue

    return []


def clean_flares(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Clean raw FLR records into a tidy DataFrame with UTC datetimes."""
    records = []
    for item in raw:
        class_type = str(item.get("classType") or "").strip()
        class_letter = class_type[0].upper() if class_type else ""
        class_letter_num = FLARE_CLASS_MAP.get(class_letter, 0)

        records.append({
            "begin_time": pd.to_datetime(item.get("beginTime"), utc=True),
            "peak_time": pd.to_datetime(item.get("peakTime"), utc=True),
            "class_type": class_type,
            "class_letter": class_letter,
            "class_letter_num": class_letter_num,
            "source_location": item.get("sourceLocation") or "",
        })

    df = pd.DataFrame(
        records,
        columns=[
            "begin_time",
            "peak_time",
            "class_type",
            "class_letter",
            "class_letter_num",
            "source_location",
        ],
    )
    if df.empty:
        df = df.astype({
            "begin_time": "datetime64[ns, UTC]",
            "peak_time": "datetime64[ns, UTC]",
            "class_type": "object",
            "class_letter": "object",
            "class_letter_num": "int64",
            "source_location": "object",
        })
    return df


def clean_cmes(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Clean raw CME records into a tidy DataFrame with UTC datetimes."""
    records = []
    for item in raw:
        start_time = item.get("startTime")
        analyses = item.get("cmeAnalyses") or []

        # Find the most accurate analysis, or fall back to the first available
        best_analysis = None
        for analysis in analyses:
            if analysis.get("isMostAccurate"):
                best_analysis = analysis
                break
        if best_analysis is None and analyses:
            best_analysis = analyses[0]

        speed = best_analysis.get("speed") if best_analysis else None
        cme_type = best_analysis.get("type") if best_analysis else None

        # Check all simulations and analyses for Earth-directed indicators
        is_earth_directed = False
        for analysis in analyses:
            for enlil in (analysis.get("enlilList") or []):
                if enlil.get("isEarthGB") or enlil.get("isEarthMinorImpact"):
                    is_earth_directed = True
                for imp in (enlil.get("impactList") or []):
                    if str(imp.get("location", "")).strip().lower() == "earth":
                        is_earth_directed = True

        records.append({
            "start_time": pd.to_datetime(start_time, utc=True),
            "speed": float(speed) if speed is not None else np.nan,
            "type": cme_type,
            "is_earth_directed": bool(is_earth_directed),
        })

    df = pd.DataFrame(records, columns=["start_time", "speed", "type", "is_earth_directed"])
    if df.empty:
        df = df.astype({
            "start_time": "datetime64[ns, UTC]",
            "speed": "float64",
            "type": "object",
            "is_earth_directed": "bool",
        })
    return df


def clean_storms(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Clean raw GST records into a tidy DataFrame with UTC datetimes."""
    records = []
    for item in raw:
        start_time = item.get("startTime")
        all_kp = item.get("allKpIndex") or []
        kp_values = [
            float(entry["kpIndex"])
            for entry in all_kp
            if entry.get("kpIndex") is not None
        ]
        max_kp = max(kp_values) if kp_values else np.nan

        records.append({
            "start_time": pd.to_datetime(start_time, utc=True),
            "max_kp": max_kp,
        })

    df = pd.DataFrame(records, columns=["start_time", "max_kp"])
    if df.empty:
        df = df.astype({
            "start_time": "datetime64[ns, UTC]",
            "max_kp": "float64",
        })
    return df


def get_space_weather(days: int = 7, api_key: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch and clean space weather data (FLR, CME, GST) for the last N days.

    Pure function design suitable for Streamlit caching (@st.cache_data).
    """
    now = datetime.now(timezone.utc)
    end_str = now.strftime("%Y-%m-%d")
    start_str = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    flr_raw = fetch_donki_data("FLR", start_str, end_str, api_key=api_key)
    cme_raw = fetch_donki_data("CME", start_str, end_str, api_key=api_key)
    gst_raw = fetch_donki_data("GST", start_str, end_str, api_key=api_key)

    return {
        "flares": clean_flares(flr_raw),
        "cmes": clean_cmes(cme_raw),
        "storms": clean_storms(gst_raw),
    }
