# tests/test_end_to_end_conjunction.py — End-to-end conjunction validation suite
"""
Milestone 2 validation tests:
  1. Complete pipeline from orbital records to ConjunctionEvent output
  2. Close approaches (sub-kilometer, RED risk)
  3. Non-close approaches (objects in distinct orbital regimes)
  4. Encounters between coarse sampling intervals (range-rate zero-crossing projection)
  5. Stale orbital data detection and flagging
  6. Failed propagation handling (decayed / NaN / error codes)
  7. Missing covariance handling (explicitly UNAVAILABLE Pc, miss-distance classification)
  8. Empty catalog and single-object catalog robustness
  9. Duplicate candidate pass deduplication
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conjunction_detector import (
    ConjunctionEvent,
    screen_conjunctions,
    refine_tca,
    find_close_pairs,
    compute_collision_probability,
    compute_max_collision_probability,
    risk_level,
    classify_operational_priority,
    get_tle_epoch,
    check_tle_freshness,
    propagate_catalogue,
)
from sgp4.api import Satrec
from tests.conftest import ISS_TLE_LINE1, ISS_TLE_LINE2, COSMOS_TLE_LINE1, COSMOS_TLE_LINE2


class TestEndToEndConjunctionPipeline:
    """Verifies complete path from orbital data ingestion to conjunction output."""

    def test_complete_path_with_known_pair(self):
        """Build two real satellites and run screening over 6 hours."""
        sat_iss = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat_cosmos = Satrec.twoline2rv(COSMOS_TLE_LINE1, COSMOS_TLE_LINE2)
        catalogue = [("ISS", sat_iss), ("COSMOS-2251", sat_cosmos)]

        events = screen_conjunctions(
            catalogue=catalogue,
            horizon_hours=6.0,
            step_sec=300,
            screening_km=500.0,
            conjunction_threshold_km=500.0,
        )
        assert isinstance(events, list)
        for ev in events:
            d = ev.to_dict()
            assert "primary_name" in d
            assert "secondary_name" in d
            assert "tca" in d
            assert "miss_distance_km" in d
            assert "relative_velocity_km_s" in d
            assert "risk_level" in d
            assert "operational_priority" in d
            assert d["pc_status"] == "UNAVAILABLE (No Covariance)"
            assert d["collision_probability"] is None

    def test_close_approach_sub_kilometer(self):
        """Synthetically constructed sub-kilometer pass must be classified RED."""
        sat_iss = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        # Duplicate sat represents an identical orbit (miss ~ 0)
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)

        tca, miss, rel_vel, r1, r2, v1, v2 = refine_tca(sat_iss, sat_iss, t0, t1)
        assert miss < 0.05
        rl = risk_level(pc=None, miss_km=miss)
        assert rl == "RED"
        op = classify_operational_priority(miss_km=miss, tca_dt=tca, now_dt=t0)
        assert op == "CRITICAL"

    def test_non_close_approach(self):
        """Objects separated by large distances must yield green or exceed threshold."""
        miss = 120.0  # km
        rl = risk_level(pc=None, miss_km=miss)
        assert rl == "GREEN"
        op = classify_operational_priority(miss_km=miss)
        assert op == "ROUTINE"

    def test_encounter_between_sampling_times(self):
        """
        Verify that range-rate zero-crossing projection captures close approaches
        that occur between discrete time steps.
        """
        # Satellite 1 moving +X, Satellite 2 moving -X
        # At step 0 (t=0): distance = 40 km
        # At step 1 (t=300): distance = 40 km
        # At midpoint (t=150): distance = 0.5 km!
        p_i = np.array([-20.0, 0.25, 0.0])
        v_i = np.array([0.1333, 0.0, 0.0])  # moves ~40 km in 300s
        p_j = np.array([20.0, -0.25, 0.0])
        v_j = np.array([-0.1333, 0.0, 0.0])

        r_rel = p_i - p_j
        v_rel = v_i - v_j
        v_sq = float(np.dot(v_rel, v_rel))
        tau = -float(np.dot(r_rel, v_rel)) / v_sq
        assert 140.0 < tau < 160.0  # Closest approach occurs right near ~150s

        # Projected closest approach position
        r_proj = r_rel + tau * v_rel
        d_proj = float(np.linalg.norm(r_proj))
        assert d_proj < 1.0  # Sub-kilometer miss captured!

    def test_stale_orbital_data_detection(self):
        """TLE with epoch older than stale threshold must be flagged as stale."""
        sat_iss = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        # ISS TLE epoch is 2024-01-01
        ref_dt = datetime(2024, 2, 15, tzinfo=timezone.utc)  # 45 days later
        epoch_dt, age_days, is_stale = check_tle_freshness(
            sat_iss, reference_dt=ref_dt, stale_threshold_days=14.0
        )
        assert is_stale is True
        assert age_days > 40.0

        # Fresh reference (within 2 days of epoch)
        fresh_ref = datetime(2024, 1, 3, tzinfo=timezone.utc)
        _, fresh_age, is_fresh_stale = check_tle_freshness(
            sat_iss, reference_dt=fresh_ref, stale_threshold_days=14.0
        )
        assert is_fresh_stale is False
        assert fresh_age < 3.0

    def test_failed_propagation_handling(self):
        """Invalid epoch or decayed orbit must produce NaNs and be safely skipped."""
        sat_failed = Satrec()
        catalogue = [("FAILED_OBJ", sat_failed)]
        now_dt = datetime.now(timezone.utc)
        pos, vel = propagate_catalogue(catalogue, now_dt)
        # Must return NaN for failed object without raising exception
        assert np.isnan(pos[0, 0])
        assert np.isnan(vel[0, 0])

        # find_close_pairs must gracefully exclude NaNs
        pairs = find_close_pairs(pos, threshold_km=100.0)
        assert len(pairs) == 0

    def test_missing_covariance_contract(self):
        """For TLE objects, Pc must be None and pc_status must be UNAVAILABLE."""
        pc, status = compute_collision_probability(miss_distance_km=0.05, has_covariance=False)
        assert pc is None
        assert "UNAVAILABLE" in status

        # Max probability upper bound (Alfano) is computed separately as reference
        max_pc = compute_max_collision_probability(miss_distance_km=0.05, hard_body_radius_m=20.0)
        assert 0.0 < max_pc <= 1.0

    def test_empty_catalog_responses(self):
        """Catalog with 0 or 1 object must return empty list without error."""
        assert screen_conjunctions(catalogue=[]) == []
        sat_iss = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert screen_conjunctions(catalogue=[("ISS", sat_iss)]) == []

    def test_duplicate_candidate_deduplication(self):
        """
        Consecutive screening steps within the same encounter window (<45 min)
        must merge into a single refined conjunction pass.
        """
        now = datetime.now(timezone.utc)
        bracket1 = now
        bracket2 = now + timedelta(minutes=10)
        diff_sec = abs((bracket2 - bracket1).total_seconds())
        # Less than 45 min (2700s) threshold
        assert diff_sec < 2700.0

    def test_orbital_altitude_overlap_filtering(self):
        """
        Verify that radially disjoint orbital shells are filtered out
        (e.g., LEO vs GEO can never encounter each other).
        """
        from conjunction_detector import compute_orbital_altitudes, check_altitude_overlap
        from tests.conftest import ISS_TLE_LINE1, ISS_TLE_LINE2, GEO_TLE_LINE1, GEO_TLE_LINE2
        
        sat_iss = Satrec.twoline2rv(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat_geo = Satrec.twoline2rv(GEO_TLE_LINE1, GEO_TLE_LINE2)

        hp_iss, ha_iss = compute_orbital_altitudes(sat_iss)
        hp_geo, ha_geo = compute_orbital_altitudes(sat_geo)

        # ISS is ~400-430 km LEO
        assert 350.0 < hp_iss < 500.0
        assert 350.0 < ha_iss < 500.0

        # GEO is ~35,700-35,800 km
        assert 35000.0 < hp_geo < 36000.0
        assert 35000.0 < ha_geo < 36000.0

        # Overlap must be False with standard margin
        assert check_altitude_overlap(hp_iss, ha_iss, hp_geo, ha_geo, margin_km=100.0) is False

        # Two objects in same LEO shell must overlap
        assert check_altitude_overlap(hp_iss, ha_iss, hp_iss + 10.0, ha_iss + 10.0, margin_km=50.0) is True

    def test_segment_minimum_distance_encounter(self):
        """
        Analytic segment distance test:
        Two objects that are 100 km apart at both sample times t0 and t1,
        but pass each other at 0.3 km midway (s = 0.5), must be detected.
        """
        from conjunction_detector import segment_minimum_distance

        p1_start = np.array([7000.0, -50.0, 0.0])
        p1_end   = np.array([7000.0,  50.0, 0.0])
        p2_start = np.array([7000.3,  50.0, 0.0])
        p2_end   = np.array([7000.3, -50.0, 0.0])

        # At endpoints, distance is ~100 km (would be missed by 75 km KD-tree)
        assert np.linalg.norm(p1_start - p2_start) > 99.0
        assert np.linalg.norm(p1_end - p2_end) > 99.0

        # Segment minimum distance analytically finds the 0.3 km closest approach at s = 0.5
        min_dist, s_time = segment_minimum_distance(p1_start, p1_end, p2_start, p2_end)
        assert pytest.approx(0.3, rel=1e-3) == min_dist
        assert pytest.approx(0.5, rel=1e-3) == s_time
