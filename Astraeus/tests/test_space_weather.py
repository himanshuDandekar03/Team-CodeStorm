# tests/test_space_weather.py — Space weather module tests
"""
Validates:
  1. X-ray flare classification boundaries are correct
  2. Atmospheric drag multiplier formula is bounded and monotonic
  3. Geomagnetic storm classification matches NOAA G-scale
  4. Drag multiplier is documented as empirical (NOT NRLMSISE-00)
  5. SGP4 isolation: space weather does NOT modify SGP4 propagation
"""
import sys
import os
import pytest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from space_weather import (
    classify_xray,
    atmospheric_drag_multiplier,
    get_geomagnetic_storm_level,
)


class TestXRayClassification:
    """Verify standard GOES X-ray flare classification boundaries."""

    def test_a_class(self):
        assert classify_xray(1e-9).startswith("A")

    def test_b_class(self):
        assert classify_xray(5e-7).startswith("B")

    def test_c_class(self):
        assert classify_xray(5e-6).startswith("C")

    def test_m_class(self):
        assert classify_xray(5e-5).startswith("M")

    def test_x_class(self):
        assert classify_xray(5e-4).startswith("X")

    def test_zero_flux(self):
        result = classify_xray(0)
        assert result == "A0.0"

    def test_negative_flux(self):
        result = classify_xray(-1e-6)
        assert result == "A0.0"


class TestAtmosphericDragMultiplier:
    """Test the empirical drag multiplier formula."""

    def test_quiet_sun_baseline(self):
        """At F10.7 = 70 (quiet Sun), multiplier should be ~1.0."""
        mult = atmospheric_drag_multiplier(f107=70.0, altitude_km=400)
        assert abs(mult - 1.0) < 0.05, f"Quiet Sun multiplier = {mult}, expected ~1.0"

    def test_increases_with_f107(self):
        """Higher solar flux should increase drag multiplier."""
        mult_low = atmospheric_drag_multiplier(f107=70.0, altitude_km=400)
        mult_high = atmospheric_drag_multiplier(f107=200.0, altitude_km=400)
        assert mult_high > mult_low

    def test_decreases_with_altitude(self):
        """At same F10.7, drag effect should decrease at higher altitude."""
        mult_low = atmospheric_drag_multiplier(f107=200.0, altitude_km=300)
        mult_high = atmospheric_drag_multiplier(f107=200.0, altitude_km=800)
        # The exponential decay term exp(-alt/500) makes low-alt more sensitive
        assert mult_low > mult_high

    def test_bounded_below(self):
        """Drag multiplier should never go below 0.2."""
        mult = atmospheric_drag_multiplier(f107=10.0, altitude_km=400)
        assert mult >= 0.2

    def test_is_empirical_not_nrlmsise(self):
        """
        The formula is 1 + 0.015*(F10.7 - 70)*exp(-alt/500).
        This is an empirical scaling approximation, NOT NRLMSISE-00.
        Verify the formula matches our implementation exactly.
        """
        f107 = 150.0
        alt = 400.0
        expected = 1.0 + 0.015 * (f107 - 70.0) * math.exp(-alt / 500.0)
        actual = atmospheric_drag_multiplier(f107=f107, altitude_km=alt)
        assert abs(actual - expected) < 1e-10


class TestGeomagneticStormScale:
    """Verify NOAA G0-G5 classification thresholds."""

    def test_quiet(self):
        level, desc = get_geomagnetic_storm_level(kp=2.0)
        assert level == "Quiet"

    def test_g0_active(self):
        level, _ = get_geomagnetic_storm_level(kp=4.0)
        assert level == "G0"

    def test_g1_minor(self):
        level, _ = get_geomagnetic_storm_level(kp=5.0)
        assert level == "G1"

    def test_g2_moderate(self):
        level, _ = get_geomagnetic_storm_level(kp=6.0)
        assert level == "G2"

    def test_g3_strong(self):
        level, _ = get_geomagnetic_storm_level(kp=7.0)
        assert level == "G3"

    def test_g4_severe(self):
        level, _ = get_geomagnetic_storm_level(kp=8.0)
        assert level == "G4"

    def test_g5_extreme(self):
        level, _ = get_geomagnetic_storm_level(kp=9.0)
        assert level == "G5"

    def test_boundary_kp5(self):
        """Kp=5 should be G1, Kp=4.9 should be G0."""
        assert get_geomagnetic_storm_level(kp=5.0)[0] == "G1"
        assert get_geomagnetic_storm_level(kp=4.9)[0] == "G0"


class TestSGP4Isolation:
    """
    Verify that space weather data does NOT modify SGP4 propagation.
    SGP4 uses the TLE's B* parameter internally; it does not ingest
    real-time F10.7 or Kp.
    """

    def test_drag_multiplier_does_not_affect_sgp4(self):
        """
        The drag multiplier is independent of the SGP4 propagator.
        SGP4 reads B* from the TLE; changing F10.7/Kp externally
        has zero effect on propagated positions.
        """
        from sgp4.api import Satrec, jday
        from tests.conftest import ISS_TLE_LINE1, ISS_TLE_LINE2
        import numpy as np

        sat = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        jd, fr = jday(2024, 1, 1, 12, 0, 0)

        # Propagate once
        e1, r1, v1 = sat.sgp4(jd, fr)

        # Call drag multiplier with extreme values (should NOT change SGP4)
        _ = atmospheric_drag_multiplier(f107=400.0, altitude_km=400)

        # Propagate again — result must be identical
        e2, r2, v2 = sat.sgp4(jd, fr)

        assert e1 == e2 == 0
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(v1, v2)
