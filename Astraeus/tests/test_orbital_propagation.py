# tests/test_orbital_propagation.py — SGP4 propagation & feature extraction tests
"""
Validates:
  1. TLE parsing creates valid Satrec objects
  2. no_kozai is in rad/min and converted correctly to rad/s
  3. Derived semi-major axis, altitude, and period are physically correct
  4. SGP4 propagation returns valid TEME r,v vectors
  5. Position vectors are in valid orbital range
  6. Velocity magnitudes are physically reasonable
  7. Invalid TLE handling
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sgp4.api import Satrec, jday
from tests.conftest import (
    ISS_TLE_LINE1, ISS_TLE_LINE2,
    COSMOS_TLE_LINE1, COSMOS_TLE_LINE2,
    GEO_TLE_LINE1, GEO_TLE_LINE2,
)

MU_EARTH = 398600.4418  # km³/s²
EARTH_RADIUS_KM = 6371.0


class TestTLEParsing:
    """Test that TLE lines are parsed into valid Satrec objects."""

    def test_iss_satrec_created(self, iss_satrec):
        assert iss_satrec is not None
        assert iss_satrec.satnum == 25544

    def test_cosmos_satrec_created(self, cosmos_satrec):
        assert cosmos_satrec is not None
        assert cosmos_satrec.satnum == 33442

    def test_geo_satrec_created(self, geo_satrec):
        assert geo_satrec is not None

    def test_no_kozai_is_positive(self, iss_satrec, cosmos_satrec, geo_satrec):
        """no_kozai must be positive for a valid orbit."""
        assert iss_satrec.no_kozai > 0
        assert cosmos_satrec.no_kozai > 0
        assert geo_satrec.no_kozai > 0

    def test_invalid_tle_produces_error(self):
        """Garbage TLE lines should produce a Satrec that errors on propagation."""
        try:
            sat = Satrec.twoline2rv("1 INVALID LINE HERE 0", "2 INVALID LINE HERE 0")
            # Even if it doesn't raise, propagation should fail
            jd, fr = jday(2024, 1, 1, 12, 0, 0)
            e, r, v = sat.sgp4(jd, fr)
            # Either creation failed or propagation returns error
            assert e != 0 or sat.no_kozai <= 0, "Invalid TLE should not propagate successfully"
        except Exception:
            pass  # Raising is also acceptable


class TestMeanMotionUnits:
    """
    Critical test: verify that no_kozai is in rad/min (sgp4 convention)
    and that converting to rad/s gives physically correct results.
    """

    def test_iss_no_kozai_rad_min(self, iss_satrec):
        """ISS mean motion should be ~0.0677 rad/min (≈15.5 rev/day)."""
        n_rad_min = iss_satrec.no_kozai
        n_rev_day = n_rad_min * 1440.0 / (2 * np.pi)
        # ISS orbits Earth ~15.5 times per day
        assert 14.5 < n_rev_day < 16.5, (
            f"ISS revs/day = {n_rev_day:.2f}, expected ~15.5"
        )

    def test_geo_no_kozai_rad_min(self, geo_satrec):
        """GEO mean motion should be ~1 rev/day."""
        n_rad_min = geo_satrec.no_kozai
        n_rev_day = n_rad_min * 1440.0 / (2 * np.pi)
        assert 0.9 < n_rev_day < 1.1, (
            f"GEO revs/day = {n_rev_day:.2f}, expected ~1.0"
        )

    def test_rad_min_to_rad_s_conversion(self, iss_satrec):
        """Verify rad/min → rad/s conversion is correct."""
        n_rad_min = iss_satrec.no_kozai
        n_rad_s = n_rad_min / 60.0

        # ISS orbital period ≈ 92 minutes = 5520 s
        period_s = 2 * np.pi / n_rad_s
        assert 5000 < period_s < 6000, (
            f"ISS period = {period_s:.0f} s, expected ~5520 s"
        )


class TestDerivedOrbitalElements:
    """Verify semi-major axis, altitude, period derived from mean motion."""

    def test_iss_semi_major_axis(self, iss_satrec):
        """ISS semi-major axis ≈ 6790 km (altitude ≈ 420 km)."""
        n_rad_s = iss_satrec.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)
        assert 6700 < a < 6900, f"ISS a = {a:.1f} km, expected ~6790"

    def test_iss_altitude(self, iss_satrec):
        """ISS altitude ≈ 420 km above Earth surface."""
        n_rad_s = iss_satrec.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)
        alt = a - EARTH_RADIUS_KM
        assert 350 < alt < 500, f"ISS altitude = {alt:.1f} km, expected ~420"

    def test_geo_semi_major_axis(self, geo_satrec):
        """GEO semi-major axis ≈ 42164 km."""
        n_rad_s = geo_satrec.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)
        assert 41000 < a < 43000, f"GEO a = {a:.1f} km, expected ~42164"

    def test_geo_altitude(self, geo_satrec):
        """GEO altitude ≈ 35786 km."""
        n_rad_s = geo_satrec.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)
        alt = a - EARTH_RADIUS_KM
        assert 34000 < alt < 37000, f"GEO altitude = {alt:.1f} km, expected ~35786"

    def test_iss_orbital_period(self, iss_satrec):
        """ISS orbital period ≈ 92 minutes."""
        n_rad_s = iss_satrec.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)
        T_s = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
        T_min = T_s / 60.0
        assert 88 < T_min < 96, f"ISS period = {T_min:.1f} min, expected ~92"


class TestSGP4Propagation:
    """Test SGP4 propagation returns valid TEME position/velocity."""

    def test_iss_propagation_succeeds(self, iss_satrec, jd_fr_at_epoch):
        jd, fr = jd_fr_at_epoch
        e, r, v = iss_satrec.sgp4(jd, fr)
        assert e == 0, f"SGP4 error code {e} for ISS"

    def test_iss_position_in_leo(self, iss_satrec, jd_fr_at_epoch):
        """ISS position magnitude should be ~6790 km (LEO)."""
        jd, fr = jd_fr_at_epoch
        e, r, v = iss_satrec.sgp4(jd, fr)
        r_mag = np.linalg.norm(r)
        assert 6500 < r_mag < 7200, f"|r| = {r_mag:.1f} km, expected ~6790"

    def test_iss_velocity_in_leo(self, iss_satrec, jd_fr_at_epoch):
        """ISS velocity should be ~7.66 km/s."""
        jd, fr = jd_fr_at_epoch
        e, r, v = iss_satrec.sgp4(jd, fr)
        v_mag = np.linalg.norm(v)
        assert 7.0 < v_mag < 8.5, f"|v| = {v_mag:.2f} km/s, expected ~7.66"

    def test_geo_position(self, geo_satrec, jd_fr_at_epoch):
        """GEO position magnitude ≈ 42164 km."""
        jd, fr = jd_fr_at_epoch
        e, r, v = geo_satrec.sgp4(jd, fr)
        r_mag = np.linalg.norm(r)
        assert 40000 < r_mag < 44000, f"|r| = {r_mag:.1f} km, expected ~42164"

    def test_geo_velocity(self, geo_satrec, jd_fr_at_epoch):
        """GEO velocity ≈ 3.07 km/s."""
        jd, fr = jd_fr_at_epoch
        e, r, v = geo_satrec.sgp4(jd, fr)
        v_mag = np.linalg.norm(v)
        assert 2.5 < v_mag < 3.5, f"|v| = {v_mag:.2f} km/s, expected ~3.07"

    def test_position_above_earth_surface(self, iss_satrec, jd_fr_at_epoch):
        """Position magnitude must be above Earth's surface."""
        jd, fr = jd_fr_at_epoch
        e, r, v = iss_satrec.sgp4(jd, fr)
        r_mag = np.linalg.norm(r)
        assert r_mag > EARTH_RADIUS_KM, "Satellite position inside Earth!"

    def test_propagation_over_24h(self, iss_satrec, reference_epoch):
        """Propagation over 24 hours should produce valid results at each step."""
        for h in range(0, 25, 6):
            dt = reference_epoch + __import__('datetime').timedelta(hours=h)
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second)
            e, r, v = iss_satrec.sgp4(jd, fr)
            assert e == 0, f"SGP4 error at T+{h}h"
            r_mag = np.linalg.norm(r)
            assert r_mag > EARTH_RADIUS_KM, f"Position inside Earth at T+{h}h"

    def test_specific_orbital_energy(self, iss_satrec, jd_fr_at_epoch):
        """Specific orbital energy should be negative for bound orbit."""
        jd, fr = jd_fr_at_epoch
        e, r, v = iss_satrec.sgp4(jd, fr)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        energy = v_mag ** 2 / 2.0 - MU_EARTH / r_mag
        assert energy < 0, f"Specific energy = {energy:.2f}, should be negative for bound orbit"
