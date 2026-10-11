# tests/test_conjunction_detector.py — Conjunction detection & collision probability tests
"""
Validates:
  1. Spatial filtering (kd-tree) finds known close pairs
  2. TCA refinement via golden-section search converges
  3. Collision probability returns None for TLE-only pairs (no covariance)
  4. Collision probability computes correctly with operator-supplied covariance
  5. Risk classification uses miss-distance thresholds when Pc unavailable
  6. ConjunctionEvent serialisation handles None Pc
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conjunction_detector import (
    find_close_pairs,
    compute_collision_probability,
    risk_level,
    risk_level_from_miss_distance,
    risk_level_from_pc,
    ConjunctionEvent,
    refine_tca,
    propagate_catalogue,
)
from sgp4.api import Satrec
from tests.conftest import ISS_TLE_LINE1, ISS_TLE_LINE2


class TestSpatialFiltering:
    """Test kd-tree based spatial filtering."""

    def test_finds_close_pair(self):
        """Two objects 5 km apart should be detected at 10 km threshold."""
        positions = np.array([
            [7000.0, 0.0, 0.0],
            [7005.0, 0.0, 0.0],
            [14000.0, 0.0, 0.0],
        ])
        pairs = find_close_pairs(positions, threshold_km=10.0)
        assert len(pairs) == 1
        assert pairs[0] == (0, 1)

    def test_no_close_pairs(self):
        """Objects far apart should yield no pairs."""
        positions = np.array([
            [7000.0, 0.0, 0.0],
            [14000.0, 0.0, 0.0],
        ])
        pairs = find_close_pairs(positions, threshold_km=10.0)
        assert len(pairs) == 0

    def test_nan_positions_excluded(self):
        """Objects with NaN positions should be skipped."""
        positions = np.array([
            [7000.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
            [7001.0, 0.0, 0.0],
        ])
        pairs = find_close_pairs(positions, threshold_km=10.0)
        assert len(pairs) == 1
        # Should match indices 0 and 2 (not 1, which is NaN)
        assert (0, 2) in pairs

    def test_multiple_close_pairs(self):
        """Three objects within threshold should yield 3 pairs."""
        positions = np.array([
            [7000.0, 0.0, 0.0],
            [7002.0, 0.0, 0.0],
            [7004.0, 0.0, 0.0],
        ])
        pairs = find_close_pairs(positions, threshold_km=5.0)
        assert len(pairs) == 3  # (0,1), (0,2), (1,2)

    def test_fewer_than_two_objects(self):
        """Single object should return empty list."""
        positions = np.array([[7000.0, 0.0, 0.0]])
        pairs = find_close_pairs(positions, threshold_km=100.0)
        assert pairs == []


class TestCollisionProbability:
    """Test the Chan 2-D Pc computation."""

    def test_no_covariance_returns_none(self):
        """Without covariance, Pc must be None."""
        pc, status = compute_collision_probability(
            miss_distance_km=5.0,
            has_covariance=False
        )
        assert pc is None
        assert "UNAVAILABLE" in status

    def test_with_covariance_returns_float(self):
        """With covariance, Pc should be a non-negative float."""
        pc, status = compute_collision_probability(
            miss_distance_km=0.1,
            sigma_primary_km=0.5,
            sigma_secondary_km=0.5,
            has_covariance=True
        )
        assert pc is not None
        assert isinstance(pc, float)
        assert 0.0 <= pc <= 1.0
        assert "COMPUTED" in status

    def test_pc_decreases_with_distance(self):
        """Pc should decrease as miss distance increases."""
        pc_close, _ = compute_collision_probability(
            miss_distance_km=0.01, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, has_covariance=True
        )
        pc_far, _ = compute_collision_probability(
            miss_distance_km=10.0, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, has_covariance=True
        )
        assert pc_close > pc_far

    def test_pc_increases_with_larger_hard_body(self):
        """Larger hard body radius should increase Pc (at same miss distance)."""
        pc_small, _ = compute_collision_probability(
            miss_distance_km=0.1, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, hard_body_radius_m=10.0,
            has_covariance=True
        )
        pc_large, _ = compute_collision_probability(
            miss_distance_km=0.1, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, hard_body_radius_m=100.0,
            has_covariance=True
        )
        assert pc_large > pc_small

    def test_zero_miss_distance_maximum_pc(self):
        """Zero miss distance should give maximum Pc for given geometry."""
        pc_zero, _ = compute_collision_probability(
            miss_distance_km=0.0, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, has_covariance=True
        )
        pc_nonzero, _ = compute_collision_probability(
            miss_distance_km=1.0, sigma_primary_km=1.0,
            sigma_secondary_km=1.0, has_covariance=True
        )
        assert pc_zero >= pc_nonzero

    def test_known_pc_value(self):
        """
        Verify against hand-calculated reference:
        R = 20 m = 0.02 km, σ² = 1² + 1² = 2 km², d = 0 km
        Pc = R²/(2σ²) * exp(0) = 0.02²/4 = 0.0001 = 1e-4
        """
        pc, _ = compute_collision_probability(
            miss_distance_km=0.0,
            sigma_primary_km=1.0,
            sigma_secondary_km=1.0,
            hard_body_radius_m=20.0,
            has_covariance=True,
        )
        expected = (0.02 ** 2) / (2 * 2.0)  # 1e-4
        assert abs(pc - expected) < 1e-10, f"Pc={pc}, expected {expected}"


class TestRiskClassification:
    """Test risk level classification logic."""

    def test_red_from_pc(self):
        assert risk_level_from_pc(1e-3) == "RED"
        assert risk_level_from_pc(1e-4) == "RED"

    def test_yellow_from_pc(self):
        assert risk_level_from_pc(1e-5) == "YELLOW"
        assert risk_level_from_pc(1e-6) == "YELLOW"

    def test_green_from_pc(self):
        assert risk_level_from_pc(1e-8) == "GREEN"

    def test_none_pc_returns_none(self):
        assert risk_level_from_pc(None) is None

    def test_red_from_miss_distance(self):
        assert risk_level_from_miss_distance(0.5) == "RED"

    def test_yellow_from_miss_distance(self):
        assert risk_level_from_miss_distance(5.0) == "YELLOW"

    def test_green_from_miss_distance(self):
        assert risk_level_from_miss_distance(15.0) == "GREEN"

    def test_unified_risk_prefers_pc(self):
        """When Pc is available, unified risk_level should use it."""
        assert risk_level(pc=1e-3, miss_km=50.0) == "RED"

    def test_unified_risk_falls_back_to_miss(self):
        """When Pc is None, unified risk_level falls back to miss distance."""
        assert risk_level(pc=None, miss_km=0.5) == "RED"
        assert risk_level(pc=None, miss_km=5.0) == "YELLOW"
        assert risk_level(pc=None, miss_km=20.0) == "GREEN"


class TestConjunctionEventSerialization:
    """Test ConjunctionEvent to_dict handles all field states."""

    def test_to_dict_with_none_pc(self):
        ev = ConjunctionEvent(
            primary_idx=0, secondary_idx=1,
            primary_name="SAT-A", secondary_name="SAT-B",
            tca=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            miss_distance_km=5.0,
            relative_velocity_km_s=12.0,
            collision_probability=None,
            pc_status="UNAVAILABLE (No Covariance)",
            risk_level="YELLOW",
            primary_pos=[7000.0, 0.0, 0.0],
            secondary_pos=[7005.0, 0.0, 0.0],
            primary_vel=[0.0, 7.5, 0.0],
            secondary_vel=[0.0, -7.5, 0.0],
        )
        d = ev.to_dict()
        assert d["collision_probability"] is None
        assert "UNAVAILABLE" in d["pc_status"]
        assert d["risk_level"] == "YELLOW"
        assert d["miss_distance_km"] == 5.0

    def test_to_dict_with_valid_pc(self):
        ev = ConjunctionEvent(
            primary_idx=0, secondary_idx=1,
            primary_name="SAT-A", secondary_name="SAT-B",
            tca=datetime(2024, 1, 1, tzinfo=timezone.utc),
            miss_distance_km=0.1,
            relative_velocity_km_s=14.0,
            collision_probability=1e-3,
            pc_status="COMPUTED (Chan 2-D isotropic)",
            pc_method="Chan 2-D isotropic",
            risk_level="RED",
            primary_pos=[7000.0, 0.0, 0.0],
            secondary_pos=[7000.1, 0.0, 0.0],
            primary_vel=[0.0, 7.5, 0.0],
            secondary_vel=[0.0, -7.5, 0.0],
        )
        d = ev.to_dict()
        assert d["collision_probability"] == pytest.approx(1e-3)
        assert "COMPUTED" in d["pc_status"]


class TestTCARefinement:
    """Test golden-section TCA refinement with synthetic scenarios."""

    def test_refine_tca_same_object(self):
        """
        Two identical Satrec objects should have miss distance ≈ 0
        at all times (they are the same orbit).
        """
        sat = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

        tca, miss_km, rel_vel, r1, r2, v1, v2 = refine_tca(sat, sat, t0, t1)
        assert miss_km < 0.001, f"Same object miss = {miss_km} km, should be ~0"
        assert rel_vel < 0.001, f"Same object rel vel = {rel_vel} km/s, should be ~0"
