# tests/conftest.py — Shared fixtures for OrbitGuard AI test suite
"""
Provides reusable TLE data, Satrec objects, and helper utilities
used across multiple test modules.
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

# Ensure the Astraeus root is on sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sgp4.api import Satrec, jday


# ── Well-known TLE data ────────────────────────────────────────────
# ISS (ZARYA) — epoch ~2024, LEO ~420 km altitude, 51.6° inclination
ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002"
ISS_TLE_LINE2 = "2 25544  51.6400 208.0000 0006703 300.0000  60.0000 15.49560000    09"

# COSMOS 2251 debris — high inclination, ~800 km altitude
COSMOS_TLE_LINE1 = "1 33442U 08039A   24001.50000000  .00000040  00000-0  26780-4 0  9006"
COSMOS_TLE_LINE2 = "2 33442  74.0400  23.0000 0014500  90.0000 270.0000 14.34780000    05"

# Geostationary satellite — ~36000 km altitude, 0° inclination
GEO_TLE_LINE1 = "1 28884U 05041A   24001.50000000 -.00000080  00000-0  00000-0 0  9009"
GEO_TLE_LINE2 = "2 28884   0.0100 275.0000 0001500 135.0000 225.0000  1.00270000    03"


@pytest.fixture
def iss_satrec():
    """Return an SGP4 Satrec object for the ISS."""
    return Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)


@pytest.fixture
def cosmos_satrec():
    """Return an SGP4 Satrec object for a COSMOS 2251 debris object."""
    return Satrec.twoline2rv(COSMOS_TLE_LINE1, COSMOS_TLE_LINE2)


@pytest.fixture
def geo_satrec():
    """Return an SGP4 Satrec object for a geostationary satellite."""
    return Satrec.twoline2rv(GEO_TLE_LINE1, GEO_TLE_LINE2)


@pytest.fixture
def reference_epoch():
    """Return a fixed reference epoch for reproducible tests."""
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def jd_fr_at_epoch(reference_epoch):
    """Return (jd, fr) tuple for the reference epoch."""
    dt = reference_epoch
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    return jd, fr
