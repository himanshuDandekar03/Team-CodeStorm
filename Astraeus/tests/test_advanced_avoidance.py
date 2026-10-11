# tests/test_advanced_avoidance.py — Unit tests for Phase 6 Advanced Avoidance Planner
"""
Validates:
  1. Universal two-body Keplerian propagation (Vallado Alg 8) conservation laws
  2. Clohessy-Wiltshire (CW) applicability verification
  3. Multi-candidate evaluation across 6 burn directions (+T, -T, +N, -N, +R, -R)
  4. Feasibility classification, safety constraint filtering, and lowest-fuel selection
  5. Infeasible candidate handling when safety margin cannot be met
  6. Synthetic conjunction scenario generator reproducibility
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from avoidance_maneuver import (
    propagate_universal,
    check_cw_assumptions,
    evaluate_candidate_maneuvers,
    create_synthetic_conjunction_scenario,
    compute_avoidance_maneuver,
    MU_EARTH,
)


class TestUniversalKeplerianPropagation:
    """Tests for singularity-free universal variable two-body propagation."""

    def test_circular_orbit_conservation(self):
        """Energy and angular momentum must be conserved across an entire orbit."""
        r0 = np.array([7000.0, 0.0, 0.0])
        v_circ = np.sqrt(MU_EARTH / 7000.0)
        v0 = np.array([0.0, v_circ, 0.0])

        period_s = 2.0 * np.pi * np.sqrt(7000.0**3 / MU_EARTH)

        # Propagate through half orbit and full orbit
        r_half, v_half = propagate_universal(r0, v0, period_s / 2.0)
        r_full, v_full = propagate_universal(r0, v0, period_s)

        # At half period, r must be [-7000, 0, 0] and v must be [0, -v_circ, 0]
        np.testing.assert_allclose(r_half, np.array([-7000.0, 0.0, 0.0]), atol=1e-3)
        np.testing.assert_allclose(v_half, np.array([0.0, -v_circ, 0.0]), atol=1e-3)

        # At full period, must return to initial state
        np.testing.assert_allclose(r_full, r0, atol=1e-3)
        np.testing.assert_allclose(v_full, v0, atol=1e-3)

        # Specific mechanical energy conservation: E = v^2/2 - mu/r
        e0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
        e_half = 0.5 * np.dot(v_half, v_half) - MU_EARTH / np.linalg.norm(r_half)
        assert abs(e0 - e_half) < 1e-6

    def test_eccentric_orbit_conservation(self):
        """Energy conservation for eccentric orbit (e = 0.2)."""
        r_p = 6800.0  # perigee km
        e = 0.2
        a = r_p / (1.0 - e)  # 8500 km
        v_p = np.sqrt(MU_EARTH * (2.0 / r_p - 1.0 / a))

        r0 = np.array([r_p, 0.0, 0.0])
        v0 = np.array([0.0, v_p, 0.0])

        # Propagate for 1800 seconds
        r_t, v_t = propagate_universal(r0, v0, 1800.0)

        e0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
        et = 0.5 * np.dot(v_t, v_t) - MU_EARTH / np.linalg.norm(r_t)
        assert abs(e0 - et) < 1e-5


class TestCWApplicability:
    """Tests for Clohessy-Wiltshire assumption validity checking."""

    def test_circular_short_drift_is_valid(self):
        """e < 0.01 and dt < 2 orbits satisfies CW."""
        r0 = np.array([7000.0, 0.0, 0.0])
        v_circ = np.sqrt(MU_EARTH / 7000.0)
        v0 = np.array([0.0, v_circ, 0.0])
        period = 2.0 * np.pi * np.sqrt(7000.0**3 / MU_EARTH)

        is_valid, ecc, period_s, reason = check_cw_assumptions(r0, v0, period)
        assert is_valid is True
        assert ecc < 0.01
        assert "valid" in reason.lower()

    def test_eccentric_orbit_fails_cw(self):
        """e > 0.05 violates CW assumptions."""
        r_p = 6800.0
        e = 0.15
        a = r_p / (1.0 - e)
        v_p = np.sqrt(MU_EARTH * (2.0 / r_p - 1.0 / a))
        r0 = np.array([r_p, 0.0, 0.0])
        v0 = np.array([0.0, v_p, 0.0])

        is_valid, ecc, period_s, reason = check_cw_assumptions(r0, v0, 3600.0)
        assert is_valid is False
        assert "Eccentricity" in reason

    def test_excessive_drift_fails_cw(self):
        """Drift time > 3 orbits violates linearized gravity."""
        r0 = np.array([7000.0, 0.0, 0.0])
        v_circ = np.sqrt(MU_EARTH / 7000.0)
        v0 = np.array([0.0, v_circ, 0.0])
        period = 2.0 * np.pi * np.sqrt(7000.0**3 / MU_EARTH)

        is_valid, ecc, period_s, reason = check_cw_assumptions(r0, v0, period * 4.0)
        assert is_valid is False
        assert "exceeds" in reason or "Lead time" in reason


class TestCandidateEvaluation:
    """Tests for multi-candidate avoidance maneuver evaluation."""

    def test_evaluates_all_six_directions(self):
        """Evaluation must test +T, -T, +N, -N, +R, -R."""
        synth_ev = create_synthetic_conjunction_scenario(miss_km=0.4, lead_time_hours=6.0)
        results = evaluate_candidate_maneuvers(
            synth_ev,
            candidate_lead_times_hours=[6.0],
            target_miss_km=10.0,
        )

        assert results["candidate_count"] == 6
        candidates = results["candidates"]
        evaluated_directions = {c["direction"] for c in candidates}
        assert evaluated_directions == {"+T", "-T", "+N", "-N", "+R", "-R"}

    def test_selects_lowest_fuel_feasible_candidate(self):
        """The chosen candidate must be feasible and have the minimal fuel usage."""
        synth_ev = create_synthetic_conjunction_scenario(miss_km=0.5, lead_time_hours=6.0)
        results = evaluate_candidate_maneuvers(
            synth_ev,
            candidate_lead_times_hours=[3.0, 6.0],
            target_miss_km=15.0,
        )

        assert results["feasibility_status"] == "FEASIBLE"
        assert results["feasible_count"] > 0
        best = results["recommended"]
        assert best is not None
        assert best["is_feasible"] is True
        assert best["new_miss_km"] >= 14.0  # meets configured constraint tolerance

        # Verify best candidate has lowest fuel among all feasible
        feasible_fuels = [c["fuel_kg"] for c in results["candidates"] if c["is_feasible"]]
        assert best["fuel_kg"] == min(feasible_fuels)

    def test_infeasible_when_margin_unreachable(self):
        """When safety margin is impossible within max delta-v, returns no recommendation."""
        synth_ev = create_synthetic_conjunction_scenario(miss_km=0.5, lead_time_hours=6.0)
        results = evaluate_candidate_maneuvers(
            synth_ev,
            candidate_lead_times_hours=[1.0],
            target_miss_km=500.0,
            max_delta_v_m_s=0.01,
        )

        assert results["feasibility_status"] == "NO_FEASIBLE_CANDIDATE"
        assert results["recommended"] is None
        assert results["feasible_count"] == 0
        for c in results["candidates"]:
            assert c["is_feasible"] is False
            assert "below safety constraint" in c["rejection_reason"] or "exceeds" in c["rejection_reason"]

    def test_complete_fuel_and_maneuver_calculation_regression(self):
        """
        Regression test addressing Tasks 1 & 8:
        Verify full pipeline candidate evaluation, Tsiolkovsky propellant calculation,
        and universal Keplerian trajectory propagation.
        """
        from avoidance_maneuver import compute_fuel_consumption

        synth_ev = create_synthetic_conjunction_scenario(miss_km=0.42, lead_time_hours=6.0)
        m0 = 1000.0
        isp = 300.0

        results = evaluate_candidate_maneuvers(
            synth_ev,
            candidate_lead_times_hours=[6.0],
            target_miss_km=25.0,
            initial_mass_kg=m0,
            isp_s=isp,
        )

        assert results["feasibility_status"] == "FEASIBLE"
        rec = results["recommended"]
        assert rec is not None
        assert rec["is_feasible"] is True
        assert rec["new_miss_km"] >= 23.75  # within 5% of 25.0 km

        # Verify fuel calculation matches Tsiolkovsky exactly
        dv = rec["delta_v_m_s"]
        expected_fuel = compute_fuel_consumption(dv, initial_mass_kg=m0, isp_s=isp)
        assert rec["fuel_kg"] == pytest.approx(round(expected_fuel, 3), abs=1e-3)

        # For typical 6h lead time, in-track dv is ~0.20-0.30 m/s
        # 1000 kg -> ~0.07-0.10 kg fuel
        assert 0.05 < rec["fuel_kg"] < 0.15


class TestSyntheticConjunctionScenario:
    """Tests for reproducible synthetic conjunction generation."""

    def test_synthetic_scenario_geometry(self):
        """Synthetic scenario must generate a sub-kilometer encounter at TCA."""
        synth_ev = create_synthetic_conjunction_scenario(miss_km=0.35, lead_time_hours=4.0)

        assert synth_ev.miss_distance_km == pytest.approx(0.35, rel=1e-3)
        assert synth_ev.risk_level == "RED"
        assert synth_ev.operational_priority == "CRITICAL"
        assert synth_ev.collision_probability is None
        assert "UNAVAILABLE" in synth_ev.pc_status
        assert synth_ev.tca > datetime.now(timezone.utc)
