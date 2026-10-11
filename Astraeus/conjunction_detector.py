# conjunction_detector.py — Real Conjunction Detection Engine for OrbitGuard AI
"""
Implements pairwise conjunction screening, Time of Closest Approach (TCA)
computation, miss-distance calculation, and simplified collision probability
(Pc) estimation using the Chan/Alfano 2-D Pc formulation.

This replaces the fake "collision probability" that was simply
  count(debris_prob > 0.7) / total_objects
with a physically meaningful, geometry-based risk metric.
"""
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.spatial import cKDTree
from sgp4.api import Satrec, jday, SatrecArray
import pandas as pd

try:
    from config import (
        MU_EARTH, EARTH_RADIUS_KM,
        CONJUNCTION_DISTANCE_THRESHOLD_KM,
        SCREENING_VOLUME_KM,
        PROPAGATION_HORIZON_HOURS,
        PROPAGATION_STEP_SEC,
        COMBINED_HARD_BODY_RADIUS_M,
        DEFAULT_POSITION_SIGMA_KM,
        HIGH_RISK_PC_THRESHOLD,
        MEDIUM_RISK_PC_THRESHOLD,
    )
except ImportError:
    MU_EARTH = 398600.4418
    EARTH_RADIUS_KM = 6371.0
    CONJUNCTION_DISTANCE_THRESHOLD_KM = 25.0
    SCREENING_VOLUME_KM = 100.0
    PROPAGATION_HORIZON_HOURS = 72.0
    PROPAGATION_STEP_SEC = 60.0
    COMBINED_HARD_BODY_RADIUS_M = 20.0
    DEFAULT_POSITION_SIGMA_KM = 1.0
    HIGH_RISK_PC_THRESHOLD = 1e-4
    MEDIUM_RISK_PC_THRESHOLD = 1e-6


# ────────────────────────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────────────────────────
class ConjunctionEvent:
    """Represents a single conjunction (close approach) between two objects."""
    __slots__ = (
        "primary_idx", "secondary_idx",
        "primary_norad", "secondary_norad",
        "primary_name", "secondary_name",
        "primary_epoch", "secondary_epoch",
        "max_epoch_age_days", "is_stale",
        "tca", "miss_distance_km",
        "relative_velocity_km_s",
        "collision_probability",
        "max_possible_pc",
        "pc_status", "pc_method",
        "risk_level", "operational_priority",
        "primary_pos", "secondary_pos",
        "primary_vel", "secondary_vel",
    )

    def __init__(self, **kwargs):
        # Set defaults for all slots so old callers don't break
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))
        # Backwards compatibility: if pc_status not provided, infer
        if self.pc_status is None:
            if self.collision_probability is not None:
                self.pc_status = "COMPUTED"
            else:
                self.pc_status = "UNAVAILABLE (No Covariance)"
        if self.pc_method is None:
            self.pc_method = "N/A"
        if self.operational_priority is None:
            self.operational_priority = classify_operational_priority(
                self.miss_distance_km, self.tca, rel_vel=self.relative_velocity_km_s
            )

    def to_dict(self):
        pc = self.collision_probability
        max_pc = self.max_possible_pc
        tca_val = self.tca.isoformat() if isinstance(self.tca, datetime) else str(self.tca)
        p_ep = self.primary_epoch.isoformat() if isinstance(self.primary_epoch, datetime) else (str(self.primary_epoch) if self.primary_epoch else None)
        s_ep = self.secondary_epoch.isoformat() if isinstance(self.secondary_epoch, datetime) else (str(self.secondary_epoch) if self.secondary_epoch else None)

        return {
            "primary_idx": int(self.primary_idx) if self.primary_idx is not None else 0,
            "secondary_idx": int(self.secondary_idx) if self.secondary_idx is not None else 0,
            "primary_norad": int(self.primary_norad) if self.primary_norad is not None else 0,
            "secondary_norad": int(self.secondary_norad) if self.secondary_norad is not None else 0,
            "primary_name": getattr(self, "primary_name", "") or "",
            "secondary_name": getattr(self, "secondary_name", "") or "",
            "primary_epoch": p_ep,
            "secondary_epoch": s_ep,
            "max_epoch_age_days": round(float(self.max_epoch_age_days), 2) if self.max_epoch_age_days is not None else None,
            "is_stale": bool(self.is_stale) if self.is_stale is not None else False,
            "tca": tca_val,
            "miss_distance_km": round(float(self.miss_distance_km), 4) if self.miss_distance_km is not None else 0.0,
            "relative_velocity_km_s": round(float(self.relative_velocity_km_s), 4) if self.relative_velocity_km_s is not None else 0.0,
            "collision_probability": float(pc) if pc is not None else None,
            "max_possible_pc": float(max_pc) if max_pc is not None else None,
            "pc_status": self.pc_status or "UNAVAILABLE (No Covariance)",
            "pc_method": self.pc_method or "N/A",
            "risk_level": self.risk_level or "GREEN",
            "operational_priority": self.operational_priority or "ROUTINE",
            "primary_pos": [round(float(x), 3) for x in self.primary_pos] if getattr(self, "primary_pos", None) is not None else [0, 0, 0],
            "secondary_pos": [round(float(x), 3) for x in self.secondary_pos] if getattr(self, "secondary_pos", None) is not None else [0, 0, 0],
            "primary_vel": [round(float(x), 3) for x in self.primary_vel] if getattr(self, "primary_vel", None) is not None else [0, 0, 0],
            "secondary_vel": [round(float(x), 3) for x in self.secondary_vel] if getattr(self, "secondary_vel", None) is not None else [0, 0, 0],
        }


def get_tle_epoch(sat):
    """Convert Satrec epoch Julian Date to UTC datetime."""
    try:
        jd = getattr(sat, "jdsatepoch", 0.0) + getattr(sat, "jdsatepochF", 0.0)
        if jd <= 0:
            return None
        return datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)
    except Exception:
        return None


def check_tle_freshness(sat, reference_dt=None, stale_threshold_days=14.0):
    """
    Check if a TLE epoch is stale relative to reference_dt.
    Returns (epoch_dt, age_days, is_stale).
    """
    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)
    epoch_dt = get_tle_epoch(sat)
    if epoch_dt is None:
        return None, 999.0, True
    age_days = abs((reference_dt - epoch_dt).total_seconds()) / 86400.0
    return epoch_dt, round(age_days, 2), age_days > stale_threshold_days


def classify_operational_priority(miss_km, tca_dt=None, now_dt=None, rel_vel=None):
    """
    Operational Alert Priority classification (distinct from miss distance & Pc):
    - CRITICAL: miss < 1 km and TCA within 24h
    - HIGH: miss < 5 km or (miss < 10 km and TCA within 12h)
    - ELEVATED: miss < 10 km
    - ROUTINE: miss >= 10 km
    """
    if miss_km is None:
        return "ROUTINE"
    hours_to_tca = None
    if tca_dt is not None:
        if now_dt is None:
            now_dt = datetime.now(timezone.utc)
        if isinstance(tca_dt, datetime):
            hours_to_tca = max(0.0, (tca_dt - now_dt).total_seconds() / 3600.0)

    if miss_km < 1.0 and (hours_to_tca is None or hours_to_tca <= 24.0):
        return "CRITICAL"
    if miss_km < 5.0 or (hours_to_tca is not None and hours_to_tca <= 12.0 and miss_km < 10.0):
        return "HIGH"
    if miss_km < 10.0:
        return "ELEVATED"
    return "ROUTINE"


# ────────────────────────────────────────────────────────────────────
#  TLE Catalogue helpers
# ────────────────────────────────────────────────────────────────────
def load_tle_catalogue(cache_dir="tle_cache"):
    """
    Load the most recent TLE cache files and return a list of
    (name, Satrec) tuples for every valid object.
    """
    catalogue = []
    if not os.path.isdir(cache_dir):
        return catalogue

    # Gather all .txt cache files, newest first
    files = sorted(
        [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".txt")],
        key=os.path.getmtime,
        reverse=True,
    )
    seen_norad = set()
    for fpath in files:
        with open(fpath, "r") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        i = 0
        while i < len(lines) - 1:
            # Try to find a name line above the TLE pair
            name = ""
            if not lines[i].startswith("1 "):
                name = lines[i]
                i += 1
                if i >= len(lines) - 1:
                    break
            if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                line1, line2 = lines[i], lines[i + 1]
                try:
                    sat = Satrec.twoline2rv(line1, line2)
                    norad = sat.satnum
                    if norad not in seen_norad and sat.no_kozai > 0:
                        seen_norad.add(norad)
                        catalogue.append((name or f"NORAD-{norad}", sat))
                except Exception:
                    pass
                i += 2
            else:
                i += 1
    return catalogue


def propagate_catalogue(catalogue, epoch_dt):
    """
    Propagate every object in `catalogue` to `epoch_dt` (timezone-aware UTC).
    Returns (positions, velocities) as (N×3) numpy arrays in km and km/s.
    Invalid propagations are set to NaN.
    """
    n = len(catalogue)
    positions = np.full((n, 3), np.nan)
    velocities = np.full((n, 3), np.nan)

    jd, fr = jday(
        epoch_dt.year, epoch_dt.month, epoch_dt.day,
        epoch_dt.hour, epoch_dt.minute,
        epoch_dt.second + epoch_dt.microsecond / 1e6,
    )

    for idx, (_, sat) in enumerate(catalogue):
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            positions[idx] = r
            velocities[idx] = v

    return positions, velocities


# ────────────────────────────────────────────────────────────────────
#  Phase 1 — Spatial filtering (kd-tree)
# ────────────────────────────────────────────────────────────────────
def find_close_pairs(positions, threshold_km=None):
    """
    Fast spatial filter: returns pairs (i, j) with i < j whose
    Euclidean distance is below `threshold_km` at the given epoch.
    Uses scipy cKDTree for O(N log N) performance.
    """
    if threshold_km is None:
        threshold_km = SCREENING_VOLUME_KM

    valid = ~np.isnan(positions[:, 0])
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return []

    tree = cKDTree(positions[valid_idx])
    raw_pairs = tree.query_pairs(r=threshold_km, output_type="ndarray")

    # Map back to original indices
    return [(int(valid_idx[a]), int(valid_idx[b])) for a, b in raw_pairs]


# ── Orbital Geometry & Altitude Filtering ───────────────────────────
def compute_orbital_altitudes(sat, mu=MU_EARTH, r_earth=EARTH_RADIUS_KM):
    """
    Compute perigee and apogee altitudes (km) above Earth surface from a Satrec object.
    Returns (h_perigee_km, h_apogee_km).
    """
    try:
        n_rad_min = getattr(sat, "no_kozai", 0.0)
        if n_rad_min <= 0:
            return 0.0, 0.0
        n_rad_s = n_rad_min / 60.0
        a = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)
        e = getattr(sat, "ecco", 0.0)
        hp = a * (1.0 - e) - r_earth
        ha = a * (1.0 + e) - r_earth
        return float(hp), float(ha)
    except Exception:
        return 0.0, 0.0


def check_altitude_overlap(hp1, ha1, hp2, ha2, margin_km=100.0):
    """
    Check if two orbital altitude bands overlap within a conservative safety margin.
    If False, the two orbits are radially separated and can never intersect.
    """
    return not ((hp1 - ha2 > margin_km) or (hp2 - ha1 > margin_km))


def segment_minimum_distance(p1_start, p1_end, p2_start, p2_end):
    """
    Analytic closest distance between two objects moving linearly over interval [0, 1].
    r(s) = (p1_start - p2_start) + s * ((p1_end - p1_start) - (p2_end - p2_start))
         = r0 + s * Δr
    Returns:
        (min_distance_km, s_fractional_time)
    """
    r0 = np.asarray(p1_start, dtype=float) - np.asarray(p2_start, dtype=float)
    dr1 = np.asarray(p1_end, dtype=float) - np.asarray(p1_start, dtype=float)
    dr2 = np.asarray(p2_end, dtype=float) - np.asarray(p2_start, dtype=float)
    dr = dr1 - dr2
    dr_sq = float(np.dot(dr, dr))
    if dr_sq < 1e-12:
        return float(np.linalg.norm(r0)), 0.0
    s_star = -float(np.dot(r0, dr)) / dr_sq
    s_clamped = max(0.0, min(1.0, s_star))
    r_min = r0 + s_clamped * dr
    return float(np.linalg.norm(r_min)), s_clamped


# ────────────────────────────────────────────────────────────────────
#  Phase 2 — Refine TCA via golden-section search
# ────────────────────────────────────────────────────────────────────
def refine_tca(sat_a, sat_b, t_start, t_end, max_iter=30, tol_sec=0.1):
    """
    Given two Satrec objects and a bracketed time interval [t_start, t_end]
    (datetime UTC), refine the Time of Closest Approach via golden-section
    search and return (tca_datetime, miss_distance_km, rel_vel_km_s,
    pos_a, pos_b, vel_a, vel_b).

    Precision & Uncertainty Note:
    - `tol_sec=0.1` is the NUMERICAL minimization bracket stopping tolerance.
    - This numerical tolerance must NOT be conflated with physical prediction
      accuracy. Real physical ephemeris uncertainty in SGP4 TLE predictions in LEO
      is typically ±5 to ±60+ seconds (growing with lead time and solar activity)
      due to unmodeled drag, gravitational harmonics, and tracking intervals.
    """
    golden = (np.sqrt(5) - 1) / 2  # ≈ 0.618

    def _dist_at(dt):
        jd, fr = jday(dt.year, dt.month, dt.day,
                       dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        e1, r1, v1 = sat_a.sgp4(jd, fr)
        e2, r2, v2 = sat_b.sgp4(jd, fr)
        if e1 != 0 or e2 != 0:
            return 1e12, None, None, None, None
        r1, r2 = np.array(r1), np.array(r2)
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(r1 - r2), r1, r2, v1, v2

    a_sec = 0.0
    b_sec = (t_end - t_start).total_seconds()

    c_sec = b_sec - golden * (b_sec - a_sec)
    d_sec = a_sec + golden * (b_sec - a_sec)

    for _ in range(max_iter):
        if (b_sec - a_sec) < tol_sec:
            break
        fc, *_ = _dist_at(t_start + timedelta(seconds=c_sec))
        fd, *_ = _dist_at(t_start + timedelta(seconds=d_sec))
        if fc < fd:
            b_sec = d_sec
        else:
            a_sec = c_sec
        c_sec = b_sec - golden * (b_sec - a_sec)
        d_sec = a_sec + golden * (b_sec - a_sec)

    tca_sec = (a_sec + b_sec) / 2.0
    tca_dt = t_start + timedelta(seconds=tca_sec)
    dist, r1, r2, v1, v2 = _dist_at(tca_dt)

    rel_vel = np.linalg.norm(np.array(v1) - np.array(v2)) if v1 is not None else 0.0
    return tca_dt, dist, rel_vel, r1, r2, v1, v2


# ────────────────────────────────────────────────────────────────────
#  Phase 3 — Collision probability (2-D Pc, Chan 1997)
# ────────────────────────────────────────────────────────────────────
def compute_collision_probability(
    miss_distance_km,
    sigma_primary_km=None,
    sigma_secondary_km=None,
    hard_body_radius_m=None,
    has_covariance=False,
):
    """
    Simplified 2-D collision probability assuming circular, isotropic
    position uncertainty for both objects (Chan 1997 / Alfano 2005).

        Pc ≈ (R² / (2·σ²)) · exp(−d² / (2·σ²))

    where:
        R   = combined hard-body radius
        d   = miss distance
        σ²  = σ_primary² + σ_secondary²

    IMPORTANT: This formula is only valid when operator-supplied
    covariance (σ values) are available.  For TLE-only catalogue
    pairs, no real covariance exists and Pc MUST NOT be computed
    with an arbitrary default sigma.

    Parameters
    ----------
    has_covariance : bool
        If False (default for TLE pairs), returns (None, "UNAVAILABLE (No Covariance)").
        If True, uses the provided sigma values to compute Pc.

    Returns
    -------
    (pc_value, pc_status) : tuple
        pc_value is float or None.
        pc_status is a string describing the computation status.
    """
    if not has_covariance:
        return None, "UNAVAILABLE (No Covariance)"

    if sigma_primary_km is None:
        sigma_primary_km = DEFAULT_POSITION_SIGMA_KM
    if sigma_secondary_km is None:
        sigma_secondary_km = DEFAULT_POSITION_SIGMA_KM
    if hard_body_radius_m is None:
        hard_body_radius_m = COMBINED_HARD_BODY_RADIUS_M

    # Convert hard-body radius to km
    R_km = hard_body_radius_m / 1000.0
    sigma_sq = sigma_primary_km ** 2 + sigma_secondary_km ** 2

    if sigma_sq == 0:
        return 0.0, "COMPUTED (σ=0 degenerate)"

    d_sq = miss_distance_km ** 2
    Pc = (R_km ** 2 / (2.0 * sigma_sq)) * np.exp(-d_sq / (2.0 * sigma_sq))
    return float(np.clip(Pc, 0.0, 1.0)), "COMPUTED (Chan 2-D isotropic)"


def compute_max_collision_probability(miss_distance_km, hard_body_radius_m=None):
    """
    Compute Alfano maximum collision probability (upper bound).
    When covariance scale sigma is unknown, the maximum possible Pc occurs
    at sigma = d / sqrt(2), giving:
        Pc_max = R^2 / (e * d^2)   for d > R * sqrt(e / 2)
        Pc_max = 1.0               for d <= R * sqrt(e / 2)
    where R is combined hard-body radius in km, e = 2.71828...
    """
    if hard_body_radius_m is None:
        hard_body_radius_m = COMBINED_HARD_BODY_RADIUS_M
    R_km = hard_body_radius_m / 1000.0
    if miss_distance_km <= 0 or miss_distance_km <= R_km:
        return 1.0
    val = (R_km ** 2) / (np.e * (miss_distance_km ** 2))
    return float(np.clip(val, 0.0, 1.0))


def risk_level_from_pc(pc):
    """Classify a Pc value into a human-readable risk level."""
    if pc is None:
        return None  # Cannot classify without Pc
    if pc >= HIGH_RISK_PC_THRESHOLD:
        return "RED"
    elif pc >= MEDIUM_RISK_PC_THRESHOLD:
        return "YELLOW"
    return "GREEN"


def risk_level_from_miss_distance(miss_km):
    """
    Classify risk using calibrated miss-distance thresholds when
    no covariance / collision probability is available.

    Thresholds based on standard operational practice:
        < 1 km  → RED   (high risk, manoeuvre likely required)
        < 10 km → YELLOW (elevated risk, monitoring required)
        ≥ 10 km → GREEN  (routine tracking)
    """
    if miss_km < 1.0:
        return "RED"
    elif miss_km < 10.0:
        return "YELLOW"
    return "GREEN"


def risk_level(pc=None, miss_km=None):
    """
    Unified risk classification.
    Uses Pc if available, otherwise falls back to miss-distance thresholds.
    """
    if pc is not None:
        return risk_level_from_pc(pc)
    if miss_km is not None:
        return risk_level_from_miss_distance(miss_km)
    return "GREEN"


# ────────────────────────────────────────────────────────────────────
#  Main screening pipeline
# ────────────────────────────────────────────────────────────────────
def screen_conjunctions(
    catalogue=None,
    cache_dir="tle_cache",
    horizon_hours=None,
    step_sec=None,
    screening_km=None,
    conjunction_threshold_km=None,
    max_events=500,
):
    """
    Full conjunction screening pipeline:

    1. Load TLE catalogue (if not provided).
    2. Propagate all objects at each time-step over the horizon.
    3. At each step, use a kd-tree to find spatially close pairs.
    4. For each candidate pair, refine TCA via golden-section search.
    5. Compute collision probability (Pc) for each conjunction.
    6. Return a list of ConjunctionEvent objects sorted by miss distance ascending.
    """
    if horizon_hours is None:
        horizon_hours = PROPAGATION_HORIZON_HOURS
    if step_sec is None:
        step_sec = PROPAGATION_STEP_SEC
    if screening_km is None:
        screening_km = SCREENING_VOLUME_KM
    if conjunction_threshold_km is None:
        conjunction_threshold_km = CONJUNCTION_DISTANCE_THRESHOLD_KM

    # 1. Load catalogue
    if catalogue is None:
        catalogue = load_tle_catalogue(cache_dir)
    if len(catalogue) < 2:
        print("⚠️  Fewer than 2 objects in catalogue — cannot screen.")
        return []

    print(f"🛰️  Screening {len(catalogue)} objects over {horizon_hours}h "
          f"(step={step_sec}s, filter={screening_km} km, threshold={conjunction_threshold_km} km)")

    now = datetime.now(timezone.utc)
    # Limit steps for performance (sample coarsely, then refine)
    coarse_step = max(step_sec, 300)  # At least 5-min steps for coarse scan
    n_coarse = int(horizon_hours * 3600 / coarse_step)

    # Collect candidate pairs with pass tracking:
    # key: (i, j) → list of pass dicts: [{"t0": dt, "t1": dt, "min_dist": float}]
    candidate_passes = {}

    print(f"   Phase 1: Coarse scan ({n_coarse} steps @ {coarse_step}s)...")
    for step_idx in range(n_coarse + 1):
        epoch = now + timedelta(seconds=step_idx * coarse_step)
        positions, velocities = propagate_catalogue(catalogue, epoch)
        close = find_close_pairs(positions, screening_km)

        for i, j in close:
            p_i, p_j = positions[i], positions[j]
            v_i, v_j = velocities[i], velocities[j]
            dist = float(np.linalg.norm(p_i - p_j))
            key = (min(i, j), max(i, j))

            # Range-rate zero-crossing projection for encounters between sample steps
            r_rel = p_i - p_j
            v_rel = v_i - v_j
            v_sq = float(np.dot(v_rel, v_rel))
            t_proj_sec = 0.0
            if v_sq > 1e-6:
                tau = -float(np.dot(r_rel, v_rel)) / v_sq
                if -coarse_step <= tau <= 2 * coarse_step:
                    t_proj_sec = tau

            # Bracket centered around estimated encounter
            bracket_center = epoch + timedelta(seconds=t_proj_sec)
            t0 = bracket_center - timedelta(seconds=coarse_step)
            t1 = bracket_center + timedelta(seconds=coarse_step)

            if key not in candidate_passes:
                candidate_passes[key] = [{"t0": t0, "t1": t1, "min_dist": dist}]
            else:
                # Deduplicate: check if this detection belongs to an existing pass (< 45 min)
                matched_pass = False
                for p in candidate_passes[key]:
                    pass_center = p["t0"] + (p["t1"] - p["t0"]) / 2
                    if abs((bracket_center - pass_center).total_seconds()) < 2700.0:  # 45 min
                        # Merge into same pass: expand bracket and update minimum coarse distance
                        p["t0"] = min(p["t0"], t0)
                        p["t1"] = max(p["t1"], t1)
                        p["min_dist"] = min(p["min_dist"], dist)
                        matched_pass = True
                        break
                if not matched_pass:
                    # New distinct pass in a different orbit
                    candidate_passes[key].append({"t0": t0, "t1": t1, "min_dist": dist})

    total_candidates = sum(len(passes) for passes in candidate_passes.values())
    print(f"   Phase 1 complete: {total_candidates} candidate passes found across {len(candidate_passes)} object pairs.")

    if total_candidates == 0:
        return []

    # Precompute perigee/apogee altitudes for all objects in catalogue
    altitudes = [compute_orbital_altitudes(sat) for _, sat in catalogue]

    # 2. Refine TCA and compute Pc
    print(f"   Phase 2: Refining TCA for {total_candidates} candidate passes...")
    events = []

    for (i, j), passes in candidate_passes.items():
        # Altitude overlap pre-filter: skip pairs whose orbital shells never intersect
        hp_i, ha_i = altitudes[i]
        hp_j, ha_j = altitudes[j]
        if not check_altitude_overlap(hp_i, ha_i, hp_j, ha_j, margin_km=screening_km + 25.0):
            continue

        name_i, sat_i = catalogue[i]
        name_j, sat_j = catalogue[j]

        epoch_i, age_i, stale_i = check_tle_freshness(sat_i, now)
        epoch_j, age_j, stale_j = check_tle_freshness(sat_j, now)
        max_age_days = max(age_i, age_j)
        is_stale = stale_i or stale_j

        norad_i = getattr(sat_i, "satnum", 0)
        norad_j = getattr(sat_j, "satnum", 0)

        for p in passes:
            tca_dt, miss_km, rel_vel, r1, r2, v1, v2 = refine_tca(
                sat_i, sat_j, p["t0"], p["t1"]
            )

            if miss_km > conjunction_threshold_km:
                continue  # Not close enough after refinement

            # For TLE-only catalogue pairs, no covariance exists.
            # Pc is reported as None; risk is based on miss-distance thresholds.
            pc, pc_status = compute_collision_probability(
                miss_km, has_covariance=False
            )
            max_pc = compute_max_collision_probability(miss_km)
            rl = risk_level(pc=pc, miss_km=miss_km)
            op_prio = classify_operational_priority(miss_km, tca_dt, now, rel_vel)

            event = ConjunctionEvent(
                primary_idx=i,
                secondary_idx=j,
                primary_norad=norad_i,
                secondary_norad=norad_j,
                primary_name=name_i,
                secondary_name=name_j,
                primary_epoch=epoch_i,
                secondary_epoch=epoch_j,
                max_epoch_age_days=max_age_days,
                is_stale=is_stale,
                tca=tca_dt,
                miss_distance_km=miss_km,
                relative_velocity_km_s=rel_vel,
                collision_probability=pc,
                max_possible_pc=max_pc,
                pc_status=pc_status,
                pc_method="Chan 2-D isotropic" if pc is not None else "N/A",
                risk_level=rl,
                operational_priority=op_prio,
                primary_pos=r1 if r1 is not None else [0, 0, 0],
                secondary_pos=r2 if r2 is not None else [0, 0, 0],
                primary_vel=v1 if v1 is not None else [0, 0, 0],
                secondary_vel=v2 if v2 is not None else [0, 0, 0],
            )
            events.append(event)

    # Sort by miss distance ascending (closest first), since Pc is
    # unavailable for TLE-only pairs and cannot be used for ranking.
    events.sort(key=lambda e: e.miss_distance_km)

    # Limit output
    events = events[:max_events]

    red = sum(1 for e in events if e.risk_level == "RED")
    yellow = sum(1 for e in events if e.risk_level == "YELLOW")
    green = sum(1 for e in events if e.risk_level == "GREEN")
    print(f"   Phase 2 complete: {len(events)} conjunctions "
          f"(🔴 {red} RED, 🟡 {yellow} YELLOW, 🟢 {green} GREEN)")

    return events


# ────────────────────────────────────────────────────────────────────
#  Quick-screen for the web dashboard (lighter / faster)
# ────────────────────────────────────────────────────────────────────
def quick_screen(cache_dir="tle_cache", horizon_hours=24, max_events=100):
    """
    Lighter conjunction screening suitable for the dashboard refresh loop.
    Uses a 10-minute coarse step and a 50 km screening volume.
    """
    return screen_conjunctions(
        cache_dir=cache_dir,
        horizon_hours=horizon_hours,
        step_sec=600,           # 10-min coarse steps
        screening_km=50.0,      # tighter filter
        conjunction_threshold_km=CONJUNCTION_DISTANCE_THRESHOLD_KM,
        max_events=max_events,
    )


# ────────────────────────────────────────────────────────────────────
#  CLI entry point for standalone testing
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  OrbitGuard AI — Conjunction Detection Engine")
    print("=" * 60)
    events = screen_conjunctions(horizon_hours=24, step_sec=600)
    print(f"\nTotal events: {len(events)}")
    for ev in events[:20]:
        d = ev.to_dict()
        pc_str = f"{d['collision_probability']:.2e}" if d['collision_probability'] is not None else d['pc_status']
        print(f"  [{d['risk_level']}] {d['primary_name'][:20]:20s} ↔ "
              f"{d['secondary_name'][:20]:20s}  "
              f"miss={d['miss_distance_km']:.2f} km  "
              f"Pc={pc_str}  "
              f"TCA={d['tca']}")
