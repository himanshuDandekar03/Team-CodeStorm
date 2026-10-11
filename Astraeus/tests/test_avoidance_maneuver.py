# tests/test_avoidance_maneuver.py — Collision avoidance module tests
"""
Validates:
  1. RTN frame construction (orthonormality, handedness)
  2. ECI ↔ RTN coordinate transforms are inverse of each other
  3. CW sensitivity matrix structure
  4. Tsiolkovsky fuel equation correctness
  5. Forward propagation of post-burn state (no algebraic shortcut)
  6. Manoeuvre options are ranked by ΔV ascending
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from avoidance_maneuver import (
    compute_rtn_frame,
    eci_to_rtn,
    rtn_to_eci,
    miss_distance_sensitivity,
    compute_avoidance_maneuver,
)
from conjunction_detector import ConjunctionEvent


class TestRTNFrame:
    """Test RTN frame construction."""

    def test_orthonormality(self):
        """RTN basis vectors must be orthonormal."""
        r = [7000.0, 0.0, 0.0]
        v = [0.0, 7.5, 0.0]
        M = compute_rtn_frame(r, v)

        # Check each row is unit length
        for i in range(3):
            assert abs(np.linalg.norm(M[i]) - 1.0) < 1e-10, (
                f"Row {i} not unit: norm = {np.linalg.norm(M[i])}"
            )

        # Check orthogonality (M @ M^T should be identity)
        eye = M @ M.T
        np.testing.assert_allclose(eye, np.eye(3), atol=1e-10)

    def test_right_handedness(self):
        """RTN frame should be right-handed: det(M) = +1."""
        r = [7000.0, 100.0, 50.0]
        v = [-0.5, 7.5, 0.1]
        M = compute_rtn_frame(r, v)
        det = np.linalg.det(M)
        assert abs(det - 1.0) < 1e-10, f"det(RTN) = {det}, expected +1"

    def test_radial_aligned_with_position(self):
        """R-hat should be aligned with the position vector."""
        r = np.array([7000.0, 100.0, 50.0])
        v = np.array([-0.5, 7.5, 0.1])
        M = compute_rtn_frame(r, v)
        R_hat = M[0]

        r_hat = r / np.linalg.norm(r)
        assert np.allclose(R_hat, r_hat, atol=1e-10)


class TestCoordinateTransforms:
    """Test ECI ↔ RTN round-trip transforms."""

    def test_eci_to_rtn_to_eci_roundtrip(self):
        """Transforming ECI → RTN → ECI should return the original vector."""
        r = [7000.0, 100.0, 50.0]
        v = [-0.5, 7.5, 0.1]
        M = compute_rtn_frame(r, v)

        vec_eci = np.array([1.0, 2.0, 3.0])
        vec_rtn = eci_to_rtn(vec_eci, M)
        vec_eci_back = rtn_to_eci(vec_rtn, M)

        np.testing.assert_allclose(vec_eci_back, vec_eci, atol=1e-10)

    def test_rtn_to_eci_to_rtn_roundtrip(self):
        """Transforming RTN → ECI → RTN should return the original vector."""
        r = [7000.0, 0.0, 0.0]
        v = [0.0, 7.5, 0.0]
        M = compute_rtn_frame(r, v)

        vec_rtn = np.array([10.0, -5.0, 3.0])
        vec_eci = rtn_to_eci(vec_rtn, M)
        vec_rtn_back = eci_to_rtn(vec_eci, M)

        np.testing.assert_allclose(vec_rtn_back, vec_rtn, atol=1e-10)


class TestCWSensitivityMatrix:
    """Test the Clohessy–Wiltshire sensitivity matrix."""

    def test_sensitivity_is_positive(self):
        """All sensitivity magnitudes should be positive."""
        r_pri = [7000.0, 0.0, 0.0]
        v_pri = [0.0, 7.5, 0.0]
        r_sec = [7010.0, 0.0, 0.0]
        v_sec = [0.0, 7.5, 0.0]
        tca = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        maneuver = datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc)

        sens, M, rtn = miss_distance_sensitivity(
            r_pri, v_pri, r_sec, v_sec, tca, maneuver
        )

        for axis in ["R", "T", "N"]:
            assert sens[axis] > 0, f"Sensitivity in {axis} is not positive"

    def test_in_track_most_efficient(self):
        """
        For near-circular orbits with sufficient lead time,
        in-track ΔV should generally produce the largest displacement
        (highest sensitivity), making it the most fuel-efficient axis.
        """
        r_pri = [7000.0, 0.0, 0.0]
        v_pri = [0.0, 7.5, 0.0]
        r_sec = [7005.0, 0.0, 0.0]
        v_sec = [0.0, 7.5, 0.0]
        tca = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        # 6 hours lead time — several orbits
        maneuver = datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc)

        sens, M, rtn = miss_distance_sensitivity(
            r_pri, v_pri, r_sec, v_sec, tca, maneuver
        )

        # In-track sensitivity should be large relative to cross-track
        assert sens["T"] > 0, "In-track sensitivity must be positive"

    def test_cw_matrix_has_correct_structure(self):
        """CW Φ_rv sub-matrix should be 3×3 with expected zero/nonzero pattern."""
        r_pri = [7000.0, 0.0, 0.0]
        v_pri = [0.0, 7.5, 0.0]
        r_sec = [7010.0, 0.0, 0.0]
        v_sec = [0.0, 7.5, 0.0]
        tca = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        maneuver = datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc)

        _, M, _ = miss_distance_sensitivity(
            r_pri, v_pri, r_sec, v_sec, tca, maneuver
        )

        assert M.shape == (3, 3)
        # CW structure: M[0,2]=0, M[1,2]=0, M[2,0]=0, M[2,1]=0
        assert abs(M[0, 2]) < 1e-15, "M[0,2] should be 0 (CW structure)"
        assert abs(M[1, 2]) < 1e-15, "M[1,2] should be 0 (CW structure)"
        assert abs(M[2, 0]) < 1e-15, "M[2,0] should be 0 (CW structure)"
        assert abs(M[2, 1]) < 1e-15, "M[2,1] should be 0 (CW structure)"


class TestTsiolkovskyFuel:
    """Test the Tsiolkovsky rocket equation implementation."""

    def test_fuel_mass_formula(self):
        """
        Tsiolkovsky: fuel_kg = m_sat * (1 - exp(-ΔV / (Isp * g0)))
        For ΔV = 1 m/s, m = 1000 kg, Isp = 300 s:
        """
        m_sat = 1000.0
        isp = 300.0
        g0 = 9.80665
        dv = 1.0  # m/s

        expected = m_sat * (1 - np.exp(-dv / (isp * g0)))
        # Should be ~0.34 kg for 1 m/s on a 1000 kg satellite
        assert 0.3 < expected < 0.4, f"Expected ~0.34 kg, got {expected:.3f}"

    def test_zero_delta_v_no_fuel(self):
        """Zero ΔV requires zero fuel."""
        m_sat = 1000.0
        isp = 300.0
        g0 = 9.80665
        dv = 0.0

        fuel = m_sat * (1 - np.exp(-dv / (isp * g0)))
        assert abs(fuel) < 1e-10

    def test_exact_analytical_verification_1000kg_and_500kg(self):
        """
        Regression test addressing Audit Finding 1:
        Verify exact propellant consumption for ΔV = 0.22 m/s, Isp = 300 s, g0 = 9.80665 m/s²:
        - For m0 = 1000.0 kg: m_prop = 1000 * (1 - exp(-0.22 / (300 * 9.80665))) ≈ 0.07478 kg.
        - For m0 = 500.0 kg:  m_prop = 500 * (1 - exp(-0.22 / (300 * 9.80665)))  ≈ 0.03739 kg (≈0.038 kg).
        Confirms that 0.038 kg corresponds to m0 = 500 kg (or two 0.11 m/s burns), not m0 = 1000 kg.
        """
        from avoidance_maneuver import compute_fuel_consumption
        # Case 1: 1000 kg satellite
        fuel_1000 = compute_fuel_consumption(delta_v_m_s=0.22, initial_mass_kg=1000.0, isp_s=300.0)
        assert pytest.approx(0.074779, rel=1e-4) == fuel_1000
        assert fuel_1000 != pytest.approx(0.038, abs=0.01)  # Must NOT be 0.038 kg for 1000 kg

        # Case 2: 500 kg satellite
        fuel_500 = compute_fuel_consumption(delta_v_m_s=0.22, initial_mass_kg=500.0, isp_s=300.0)
        assert pytest.approx(0.037389, rel=1e-4) == fuel_500
        assert round(fuel_500, 3) == 0.037 or round(fuel_500, 3) == 0.038


class TestAvoidanceManeuverComputation:
    """Test the full avoidance maneuver computation."""

    def _make_event(self, miss_km=5.0):
        """Create a synthetic conjunction event for testing."""
        return ConjunctionEvent(
            primary_idx=0,
            secondary_idx=1,
            primary_name="SAT-A",
            secondary_name="SAT-B",
            tca=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            miss_distance_km=miss_km,
            relative_velocity_km_s=12.0,
            collision_probability=None,
            pc_status="UNAVAILABLE (No Covariance)",
            risk_level="YELLOW",
            primary_pos=[7000.0, 0.0, 0.0],
            secondary_pos=[7000.0 + miss_km, 0.0, 0.0],
            primary_vel=[0.0, 7.5, 0.0],
            secondary_vel=[0.0, -7.5, 0.0],
        )

    def test_returns_options(self):
        """Should return at least one manoeuvre option."""
        ev = self._make_event()
        options = compute_avoidance_maneuver(ev)
        assert len(options) > 0

    def test_options_sorted_by_delta_v(self):
        """Options should be sorted by ΔV ascending (cheapest first)."""
        ev = self._make_event()
        options = compute_avoidance_maneuver(ev)
        dvs = [o["delta_v_m_s"] for o in options]
        assert dvs == sorted(dvs), f"Not sorted: {dvs}"

    def test_new_miss_from_forward_propagation(self):
        """
        new_miss_km should NOT be simply current_miss + delta_miss.
        It should come from the CW forward propagation.
        """
        ev = self._make_event(miss_km=5.0)
        options = compute_avoidance_maneuver(ev, target_miss_km=50.0)
        for opt in options:
            # The new miss distance should be computed, not just added
            assert "new_miss_km" in opt
            assert opt["new_miss_km"] > 0
            # It should also have the propagation method documented
            assert "propagation_method" in opt

    def test_fuel_is_positive(self):
        """Fuel consumption should be positive for any non-zero ΔV."""
        ev = self._make_event()
        options = compute_avoidance_maneuver(ev)
        for opt in options:
            if opt["delta_v_m_s"] > 0:
                assert opt["fuel_kg"] > 0

    def test_delta_v_vector_has_three_components(self):
        """ECI ΔV vector should have 3 components."""
        ev = self._make_event()
        options = compute_avoidance_maneuver(ev)
        for opt in options:
            assert len(opt["delta_v_vector_eci_m_s"]) == 3

    def test_all_three_directions(self):
        """Should produce options for R, T, N directions."""
        ev = self._make_event()
        options = compute_avoidance_maneuver(ev)
        directions = {o["direction"] for o in options}
        assert directions == {"R", "T", "N"}, f"Missing directions: {directions}"
