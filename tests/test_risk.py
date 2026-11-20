"""Unit tests for SpaceGuard risk assessment module."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.risk import assess_risk


@pytest.fixture
def empty_flares() -> pd.DataFrame:
    return pd.DataFrame(columns=["begin_time", "peak_time", "class_type", "class_letter", "class_letter_num", "source_location"])


@pytest.fixture
def empty_cmes() -> pd.DataFrame:
    return pd.DataFrame(columns=["start_time", "speed", "type", "is_earth_directed"])


@pytest.fixture
def empty_storms() -> pd.DataFrame:
    return pd.DataFrame(columns=["start_time", "max_kp"])


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def test_empty_data(empty_flares, empty_cmes, empty_storms, base_time):
    result = assess_risk(empty_flares, empty_cmes, empty_storms, hours=72, now=base_time)
    assert result["level"] == "Low"
    assert result["reasons"] == ["No flares M-class or above, no storm, no fast CMEs in the last 72 hours"]


def test_low_risk_events(empty_flares, empty_cmes, empty_storms, base_time):
    # C-class flare, slow CME, Kp below 5
    flares = pd.DataFrame([{
        "begin_time": base_time - timedelta(hours=2),
        "peak_time": base_time - timedelta(hours=1),
        "class_type": "C2.5",
        "class_letter": "C",
    }])
    cmes = pd.DataFrame([{
        "start_time": base_time - timedelta(hours=5),
        "speed": 350.0,
        "is_earth_directed": False,
    }])
    storms = pd.DataFrame([{
        "start_time": base_time - timedelta(hours=3),
        "max_kp": 3.67,
    }])

    result = assess_risk(flares, cmes, storms, hours=72, now=base_time)
    assert result["level"] == "Low"
    assert result["reasons"] == ["No flares M-class or above, no storm, no fast CMEs in the last 72 hours"]


def test_nan_speed_cme(empty_flares, empty_storms, base_time):
    # CME with NaN speed must not crash or trigger false moderate/high
    cmes = pd.DataFrame([
        {"start_time": base_time - timedelta(hours=4), "speed": np.nan, "is_earth_directed": True},
        {"start_time": base_time - timedelta(hours=2), "speed": None, "is_earth_directed": False},
    ])
    result = assess_risk(empty_flares, cmes, empty_storms, hours=72, now=base_time)
    assert result["level"] == "Low"
    assert "No flares M-class or above" in result["reasons"][0]


def test_moderate_flare_m1_0(empty_cmes, empty_storms, base_time):
    flares = pd.DataFrame([{
        "begin_time": base_time - timedelta(hours=10),
        "peak_time": base_time - timedelta(hours=9),
        "class_type": "M1.0",
        "class_letter": "M",
    }])
    result = assess_risk(flares, empty_cmes, empty_storms, hours=72, now=base_time)
    assert result["level"] == "Moderate"
    assert len(result["reasons"]) == 1
    assert "M1.0 flare at" in result["reasons"][0]
    assert "M-class flare" in result["reasons"][0]


def test_high_flare_x1_0(empty_cmes, empty_storms, base_time):
    flares = pd.DataFrame([{
        "begin_time": base_time - timedelta(hours=5),
        "peak_time": base_time - timedelta(hours=4),
        "class_type": "X1.0",
        "class_letter": "X",
    }])
    result = assess_risk(flares, empty_cmes, empty_storms, hours=72, now=base_time)
    assert result["level"] == "High"
    assert len(result["reasons"]) == 1
    assert "X1.0 flare at" in result["reasons"][0]
    assert "X-class flare" in result["reasons"][0]


def test_boundary_kp_moderate(empty_flares, empty_cmes, base_time):
    # Kp 4.9 is Low, Kp 5.0 is Moderate
    storms_low = pd.DataFrame([{"start_time": base_time - timedelta(hours=2), "max_kp": 4.9}])
    result_low = assess_risk(empty_flares, empty_cmes, storms_low, hours=72, now=base_time)
    assert result_low["level"] == "Low"

    storms_mod = pd.DataFrame([{"start_time": base_time - timedelta(hours=2), "max_kp": 5.0}])
    result_mod = assess_risk(empty_flares, empty_cmes, storms_mod, hours=72, now=base_time)
    assert result_mod["level"] == "Moderate"
    assert "max Kp >= 5" in result_mod["reasons"][0]


def test_boundary_kp_high(empty_flares, empty_cmes, base_time):
    # Kp 6.9 is Moderate, Kp 7.0 is High
    storms_69 = pd.DataFrame([{"start_time": base_time - timedelta(hours=2), "max_kp": 6.9}])
    result_69 = assess_risk(empty_flares, empty_cmes, storms_69, hours=72, now=base_time)
    assert result_69["level"] == "Moderate"
    assert "max Kp 6.9" in result_69["reasons"][0]

    storms_70 = pd.DataFrame([{"start_time": base_time - timedelta(hours=2), "max_kp": 7.0}])
    result_70 = assess_risk(empty_flares, empty_cmes, storms_70, hours=72, now=base_time)
    assert result_70["level"] == "High"
    assert "max Kp >= 7" in result_70["reasons"][0]


def test_boundary_cme_moderate(empty_flares, empty_storms, base_time):
    # CME 499 is Low, CME 500 is Moderate
    cmes_499 = pd.DataFrame([{"start_time": base_time - timedelta(hours=3), "speed": 499.0, "is_earth_directed": False}])
    result_499 = assess_risk(empty_flares, cmes_499, empty_storms, hours=72, now=base_time)
    assert result_499["level"] == "Low"

    cmes_500 = pd.DataFrame([{"start_time": base_time - timedelta(hours=3), "speed": 500.0, "is_earth_directed": False}])
    result_500 = assess_risk(empty_flares, cmes_500, empty_storms, hours=72, now=base_time)
    assert result_500["level"] == "Moderate"
    assert "speed >= 500 km/s" in result_500["reasons"][0]


def test_boundary_cme_high_earth_directed(empty_flares, empty_storms, base_time):
    # Earth-directed CME 999 is Moderate, Earth-directed CME 1000 is High
    cmes_999 = pd.DataFrame([{"start_time": base_time - timedelta(hours=3), "speed": 999.0, "is_earth_directed": True}])
    result_999 = assess_risk(empty_flares, cmes_999, empty_storms, hours=72, now=base_time)
    assert result_999["level"] == "Moderate"

    cmes_1000 = pd.DataFrame([{"start_time": base_time - timedelta(hours=3), "speed": 1000.0, "is_earth_directed": True}])
    result_1000 = assess_risk(empty_flares, cmes_1000, empty_storms, hours=72, now=base_time)
    assert result_1000["level"] == "High"
    assert "Earth-directed CME speed >= 1000 km/s" in result_1000["reasons"][0]

    # Non-Earth-directed CME 1200 is Moderate (not High)
    cmes_non_earth = pd.DataFrame([{"start_time": base_time - timedelta(hours=3), "speed": 1200.0, "is_earth_directed": False}])
    result_non_earth = assess_risk(empty_flares, cmes_non_earth, empty_storms, hours=72, now=base_time)
    assert result_non_earth["level"] == "Moderate"


def test_events_outside_window(empty_cmes, empty_storms, base_time):
    # X-class flare 80 hours ago is outside the 72-hour window -> Low
    flares = pd.DataFrame([{
        "begin_time": base_time - timedelta(hours=81),
        "peak_time": base_time - timedelta(hours=80),
        "class_type": "X2.0",
        "class_letter": "X",
    }])
    result = assess_risk(flares, empty_cmes, empty_storms, hours=72, now=base_time)
    assert result["level"] == "Low"


def test_multiple_triggers(empty_cmes, base_time):
    flares = pd.DataFrame([{
        "begin_time": base_time - timedelta(hours=10),
        "peak_time": base_time - timedelta(hours=9),
        "class_type": "X1.5",
        "class_letter": "X",
    }])
    storms = pd.DataFrame([{
        "start_time": base_time - timedelta(hours=5),
        "max_kp": 7.33,
    }])
    result = assess_risk(flares, empty_cmes, storms, hours=72, now=base_time)
    assert result["level"] == "High"
    assert len(result["reasons"]) == 2
