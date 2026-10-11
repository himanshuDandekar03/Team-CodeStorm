# avoidance_maneuver.py — Fuel-Efficient Collision Avoidance for OrbitGuard AI
"""
Computes impulsive collision-avoidance manoeuvres in the RTN
(Radial / In-Track / Cross-Track) frame, optimising for minimum ΔV.

Given a conjunction event (TCA, miss distance, relative positions &
velocities), this module:

  1. Determines the optimal manoeuvre direction (in-track is usually cheapest).
  2. Computes the ΔV required to increase the miss distance above a safe
     threshold.
  3. Evaluates fuel cost using the existing propulsion models.
  4. Returns a ranked list of manoeuvre options.
"""
import numpy as np
from datetime import datetime, timezone, timedelta

try:
    from config import (
        MU_EARTH,
        EARTH_RADIUS_KM,
        CONJUNCTION_DISTANCE_THRESHOLD_KM,
        COMBINED_HARD_BODY_RADIUS_M,
    )
except ImportError:
    MU_EARTH = 398600.4418
    EARTH_RADIUS_KM = 6371.0
    CONJUNCTION_DISTANCE_THRESHOLD_KM = 25.0
    COMBINED_HARD_BODY_RADIUS_M = 20.0


# ── RTN frame utilities ──────────────────────────────────────────────
def compute_rtn_frame(r_vec, v_vec):
    """
    Build the Radial / In-Track / Cross-Track (RTN) rotation matrix.

    R = r̂   (radial outward)
    N = r̂ × v̂  (cross-track, orbit normal)
    T = N × R  (in-track, roughly along velocity for circular orbits)

    Returns a 3×3 matrix whose rows are R, T, N unit vectors.
    """
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)

    R_hat = r / np.linalg.norm(r)
    N_hat_raw = np.cross(r, v)
    N_hat = N_hat_raw / np.linalg.norm(N_hat_raw)
    T_hat = np.cross(N_hat, R_hat)

    return np.array([R_hat, T_hat, N_hat])


def eci_to_rtn(vec_eci, rtn_matrix):
    """Transform a vector from ECI to RTN frame."""
    return rtn_matrix @ np.asarray(vec_eci, dtype=float)


def rtn_to_eci(vec_rtn, rtn_matrix):
    """Transform a vector from RTN to ECI frame."""
    return rtn_matrix.T @ np.asarray(vec_rtn, dtype=float)


# ── Miss-distance sensitivity to ΔV ─────────────────────────────────
def miss_distance_sensitivity(
    r_primary, v_primary, r_secondary, v_secondary, tca_dt, maneuver_dt,
):
    """
    Compute the linear sensitivity of miss distance to ΔV in each RTN
    direction.  Uses the Clohessy–Wiltshire (CW / Hill's) relative motion
    equations for a near-circular primary orbit.

    Returns:
        sensitivities: dict with keys "R", "T", "N" mapping to the
                       km of miss-distance change per m/s of ΔV.
    """
    r_pri = np.asarray(r_primary, dtype=float)
    v_pri = np.asarray(v_primary, dtype=float)

    rtn = compute_rtn_frame(r_pri, v_pri)

    # Orbital angular rate (rad/s)
    r_mag = np.linalg.norm(r_pri)
    n = np.sqrt(MU_EARTH / r_mag ** 3)

    # Time before TCA (seconds)
    dt = (tca_dt - maneuver_dt).total_seconds()
    if dt <= 0:
        dt = 1.0  # Guard: maneuver must be before TCA

    nt = n * dt
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)

    # CW state-transition matrix Φ(dt) — position part due to ΔV
    # For unit ΔV in R, T, N the position displacement at TCA is:
    #   δr_R = (1/n) sin(nt) ΔV_R  +  (2/n)(1 - cos(nt)) ΔV_T
    #   δr_T = (2/n)(cos(nt) - 1) ΔV_R  +  (1/n)(4 sin(nt) - 3nt) ΔV_T
    #   δr_N = (1/n) sin(nt) ΔV_N

    # Build the 3×3 matrix M where δr = M · ΔV  (in RTN frame)
    M = np.zeros((3, 3))
    M[0, 0] = sin_nt / n                     # δr_R / ΔV_R
    M[0, 1] = 2.0 * (1 - cos_nt) / n         # δr_R / ΔV_T
    M[1, 0] = 2.0 * (cos_nt - 1) / n         # δr_T / ΔV_R
    M[1, 1] = (4.0 * sin_nt - 3.0 * nt) / n  # δr_T / ΔV_T
    M[2, 2] = sin_nt / n                     # δr_N / ΔV_N

    # Sensitivity magnitude (km per (km/s) = km·s/km = s)
    # Convert to km per m/s  (÷1000)
    sens = {}
    for i, axis in enumerate(["R", "T", "N"]):
        col = M[:, i]
        sens[axis] = float(np.linalg.norm(col)) / 1000.0  # km / (m/s)

    return sens, M, rtn


# ── Manoeuvre computation ────────────────────────────────────────────
def compute_avoidance_maneuver(
    conjunction_event,
    target_miss_km=None,
    lead_time_hours=6.0,
):
    """
    Given a ConjunctionEvent, compute candidate avoidance manoeuvres
    in each RTN direction and return a ranked list by ΔV (lowest first).

    Parameters
    ----------
    conjunction_event : ConjunctionEvent
        Must have: tca, miss_distance_km, primary_pos, primary_vel,
                   secondary_pos, secondary_vel.
    target_miss_km : float, optional
        Desired minimum miss distance after manoeuvre (default: 2× threshold).
    lead_time_hours : float
        How many hours before TCA the manoeuvre is executed.

    Returns
    -------
    list of dict, each containing:
        direction, delta_v_m_s, delta_v_vector_eci, fuel_kg,
        new_miss_km, maneuver_time, description
    """
    if target_miss_km is None:
        target_miss_km = 2.0 * CONJUNCTION_DISTANCE_THRESHOLD_KM

    ev = conjunction_event
    tca_raw = getattr(ev, 'tca', None) if not isinstance(ev, dict) else ev.get('tca')
    if isinstance(tca_raw, datetime):
        tca = tca_raw
    else:
        try:
            tca = datetime.fromisoformat(str(tca_raw))
        except Exception:
            tca = datetime.now(timezone.utc) + timedelta(hours=12)

    maneuver_time = tca - timedelta(hours=lead_time_hours)

    def _get_val(prop, default):
        if isinstance(ev, dict):
            return ev.get(prop, default)
        return getattr(ev, prop, default)

    r_pri = np.asarray(_get_val('primary_pos', [7000.0, 0.0, 0.0]), dtype=float)
    v_pri = np.asarray(_get_val('primary_vel', [0.0, 7.5, 0.0]), dtype=float)
    r_sec = np.asarray(_get_val('secondary_pos', [7001.0, 0.0, 0.0]), dtype=float)
    v_sec = np.asarray(_get_val('secondary_vel', [0.0, 7.5, 0.0]), dtype=float)

    # Needed additional miss distance
    current_miss = float(_get_val('miss_distance_km', 1.0))
    delta_miss = max(target_miss_km - current_miss, 1.0)  # at least 1 km

    sens, M, rtn_matrix = miss_distance_sensitivity(
        r_pri, v_pri, r_sec, v_sec, tca, maneuver_time,
    )

    options = []
    for axis in ["T", "R", "N"]:  # In-track first (usually cheapest)
        sensitivity = sens[axis]
        if sensitivity < 1e-9:
            continue

        # ΔV required (m/s)
        dv_ms = delta_miss / sensitivity

        # ΔV vector in RTN (m/s)
        dv_rtn = np.zeros(3)
        idx = {"R": 0, "T": 1, "N": 2}[axis]
        dv_rtn[idx] = dv_ms / 1000.0  # convert m/s → km/s for ECI transform

        # Convert to ECI
        dv_eci = rtn_to_eci(dv_rtn, rtn_matrix) * 1000.0  # back to m/s

        # ── Forward propagation of post-burn state via CW equations ──
        # Compute the actual displacement at TCA caused by the ΔV using
        # the full CW state-transition sub-matrix (position due to velocity).
        #
        # ΔV in RTN (km/s):
        dv_rtn_kms = np.zeros(3)
        dv_rtn_kms[idx] = dv_ms / 1000.0

        # CW position displacement at TCA (km, in RTN frame):
        # δr = M · Δv   where M is the 3×3 CW Φ_rv sub-matrix
        delta_r_rtn = M @ dv_rtn_kms

        # Current miss vector in RTN frame
        miss_vec_eci = np.asarray(r_sec, dtype=float) - np.asarray(r_pri, dtype=float)
        miss_vec_rtn = eci_to_rtn(miss_vec_eci, rtn_matrix)

        # Post-burn miss vector = original miss vector + displacement
        # (displacement moves the primary, so the miss vector increases)
        new_miss_vec_rtn = miss_vec_rtn + delta_r_rtn
        new_miss_km = float(np.linalg.norm(new_miss_vec_rtn))

        # Fuel estimate (Tsiolkovsky, assuming satellite mass ~ 1000 kg, Isp ~ 300 s)
        m_sat = 1000.0  # kg
        isp = 300.0     # s
        g0 = 9.80665    # m/s²
        ve = isp * g0
        delta_v = abs(dv_ms)
        fuel_kg = m_sat * (1 - np.exp(-delta_v / ve))

        dir_name = {"R": "Radial", "T": "In-Track", "N": "Cross-Track"}[axis]
        options.append({
            "direction": axis,
            "direction_full": dir_name,
            "delta_v_m_s": round(float(delta_v), 4),
            "delta_v_vector_eci_m_s": [round(float(x), 4) for x in dv_eci],
            "fuel_kg": round(float(fuel_kg), 3),
            "new_miss_km": round(new_miss_km, 2),
            "current_miss_km": round(float(current_miss), 2),
            "maneuver_time": maneuver_time.isoformat(),
            "lead_time_hours": lead_time_hours,
            "sensitivity_km_per_ms": round(float(sensitivity), 4),
            "propagation_method": "CW relative motion (linearised)",
            "description": (
                f"{dir_name} burn of {delta_v:.2f} m/s at T-{lead_time_hours:.0f}h → "
                f"miss distance {new_miss_km:.1f} km "
                f"(fuel ≈ {fuel_kg:.1f} kg)"
            ),
        })

    # Sort by ΔV (cheapest first)
    options.sort(key=lambda o: o["delta_v_m_s"])

    # Build human-readable description
    for i, opt in enumerate(options):
        opt["rank"] = i + 1
        opt["description"] = (
            f"{opt['direction_full']} burn of {opt['delta_v_m_s']:.2f} m/s "
            f"at T-{lead_time_hours:.0f}h → "
            f"miss distance {opt['new_miss_km']:.1f} km "
            f"(fuel ≈ {opt['fuel_kg']:.1f} kg)"
        )

    return options


def compute_avoidance_for_events(events, lead_time_hours=6.0, max_events=10):
    """
    Compute avoidance manoeuvres for the highest-risk conjunction events.
    Returns a list of (event_dict, maneuver_options) tuples.
    """
    results = []
    for ev in events[:max_events]:
        try:
            options = compute_avoidance_maneuver(ev, lead_time_hours=lead_time_hours)
            results.append({
                "conjunction": ev.to_dict(),
                "maneuver_options": options,
                "recommended": options[0] if options else None,
            })
        except Exception as exc:
            print(f"⚠️  Avoidance computation failed for "
                  f"{getattr(ev, 'primary_name', '?')} ↔ "
                  f"{getattr(ev, 'secondary_name', '?')}: {exc}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from conjunction_detector import screen_conjunctions

    print("=" * 60)
    print("  OrbitGuard AI — Collision Avoidance Planner")
    print("=" * 60)

    events = screen_conjunctions(horizon_hours=24, step_sec=600)
    if not events:
        print("No conjunction events found — no avoidance needed.")
    else:
        red_events = [e for e in events if e.risk_level in ("RED", "YELLOW")]
        target_events = red_events[:5] if red_events else events[:3]

        results = compute_avoidance_for_events(target_events)
        for r in results:
            conj = r["conjunction"]
            print(f"\n🔴 {conj['primary_name']} ↔ {conj['secondary_name']}")
            print(f"   Miss: {conj['miss_distance_km']:.2f} km  |  Pc: {conj['collision_probability']:.2e}")
            for opt in r["maneuver_options"]:
                flag = "✅" if opt["rank"] == 1 else "  "
                print(f"   {flag} #{opt['rank']} {opt['description']}")
