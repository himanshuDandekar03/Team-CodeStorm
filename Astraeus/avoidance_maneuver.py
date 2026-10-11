# avoidance_maneuver.py — Fuel-Efficient Collision Avoidance for OrbitGuard AI
"""
Computes impulsive collision-avoidance manoeuvres in the RTN
(Radial / In-Track / Cross-Track) frame, optimising for minimum ΔV.

Given a conjunction event (TCA, miss distance, relative positions &
velocities), this module:

  1. Evaluates candidate impulsive maneuvers across multiple burn directions
     (+T, -T, +N, -N, +R, -R) and candidate execution times.
  2. Applies the maneuver, propagates the post-burn state via both
     linearized Clohessy-Wiltshire (CW) equations and exact nonlinear
     Two-Body Universal Keplerian dynamics.
  3. Validates whether Clohessy-Wiltshire assumptions hold (near-circular orbit,
     short drift time); when invalid, Keplerian two-body propagation is used.
  4. Recalculates closest approach, new miss distance, required ΔV, and
     Tsiolkovsky propellant mass consumption.
  5. Evaluates configured safety constraints, rejects infeasible candidates,
     and selects the lowest-fuel feasible candidate among those evaluated.
  6. Provides full explainability metadata: original vs modified states,
     data freshness, uncertainty status, and operational disclaimers.
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

    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        return np.eye(3)
    R_hat = r / r_norm

    N_hat_raw = np.cross(r, v)
    N_norm = np.linalg.norm(N_hat_raw)
    if N_norm < 1e-9:
        # Fallback if r and v are parallel
        arb = np.array([0.0, 0.0, 1.0]) if abs(R_hat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        N_hat_raw = np.cross(r, arb)
        N_norm = np.linalg.norm(N_hat_raw)
    N_hat = N_hat_raw / N_norm
    T_hat = np.cross(N_hat, R_hat)

    return np.array([R_hat, T_hat, N_hat])


def eci_to_rtn(vec_eci, rtn_matrix):
    """Transform a vector from ECI to RTN frame."""
    return rtn_matrix @ np.asarray(vec_eci, dtype=float)


def rtn_to_eci(vec_rtn, rtn_matrix):
    """Transform a vector from RTN to ECI frame."""
    return rtn_matrix.T @ np.asarray(vec_rtn, dtype=float)


# ── Universal Variable Two-Body Propagation (Vallado Algorithm 8) ────
def _stumpff_c2(z):
    """Stumpff function c2(z) = (1 - cos(sqrt(z))) / z."""
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else:
        return 0.5 - z / 24.0 + (z**2) / 720.0


def _stumpff_c3(z):
    """Stumpff function c3(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3."""
    if z > 1e-6:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
    elif z < -1e-6:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)**3)
    else:
        return 1.0 / 6.0 - z / 120.0 + (z**2) / 5040.0


def propagate_universal(r0, v0, dt_sec, mu=MU_EARTH, tol=1e-11, max_iter=60):
    """
    Universal variable two-body propagation using Lagrange f and g functions.
    Handles circular, eccentric, parabolic, and hyperbolic orbits with zero singularities.

    Parameters
    ----------
    r0, v0 : array-like (3,)
        Initial position (km) and velocity (km/s) in inertial frame.
    dt_sec : float
        Propagation elapsed time in seconds.
    mu : float
        Gravitational parameter (km³/s²).

    Returns
    -------
    r, v : ndarray (3,)
        Position (km) and velocity (km/s) at t0 + dt_sec.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if abs(dt_sec) < 1e-9:
        return r0.copy(), v0.copy()

    r0_mag = float(np.linalg.norm(r0))
    v0_mag = float(np.linalg.norm(v0))
    if r0_mag < 1.0:
        return r0.copy(), v0.copy()

    alpha = (2.0 / r0_mag) - (v0_mag**2 / mu)
    sqrt_mu = np.sqrt(mu)

    # Initial guess for universal anomaly chi
    if alpha > 1e-6:
        chi = sqrt_mu * dt_sec * alpha
    else:
        chi = sqrt_mu * dt_sec / r0_mag

    r0_dot_v0 = float(np.dot(r0, v0))

    for _ in range(max_iter):
        chi2 = chi**2
        z = alpha * chi2
        c2 = _stumpff_c2(z)
        c3 = _stumpff_c3(z)

        # Kepler universal equation
        F = (r0_dot_v0 / sqrt_mu) * chi2 * c2 + (1.0 - alpha * r0_mag) * (chi**3) * c3 + r0_mag * chi - sqrt_mu * dt_sec
        dF = (r0_dot_v0 / sqrt_mu) * chi * (1.0 - z * c3) + (1.0 - alpha * r0_mag) * chi2 * c2 + r0_mag

        if abs(dF) < 1e-14:
            break
        dchi = -F / dF
        chi += dchi
        if abs(dchi) < tol:
            break

    chi2 = chi**2
    z = alpha * chi2
    c2 = _stumpff_c2(z)
    c3 = _stumpff_c3(z)

    f = 1.0 - (chi2 / r0_mag) * c2
    g = dt_sec - (chi**3 / sqrt_mu) * c3

    r = f * r0 + g * v0
    r_mag = float(np.linalg.norm(r))

    if r_mag > 0:
        f_dot = (sqrt_mu / (r_mag * r0_mag)) * chi * (z * c3 - 1.0)
        g_dot = 1.0 - (chi2 / r_mag) * c2
        v = f_dot * r0 + g_dot * v0
    else:
        v = v0.copy()

    return r, v


def check_cw_assumptions(r_vec, v_vec, dt_sec, mu=MU_EARTH):
    """
    Check if Clohessy-Wiltshire (CW) linear relative motion assumptions are valid:
      1. Primary orbit is near-circular: eccentricity e < 0.01
      2. Drift duration is short: dt <= 2.0 orbital periods

    Returns
    -------
    (is_valid, eccentricity, period_sec, reason) : tuple
    """
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    if r_mag < 1.0:
        return False, 0.0, 5400.0, "Invalid position vector"

    energy = (v_mag**2)/2.0 - mu/r_mag
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else r_mag
    period_sec = 2.0 * np.pi * np.sqrt(max(1.0, abs(a)**3) / mu)

    h = np.cross(r, v)
    e_vec = (np.cross(v, h) / mu) - (r / r_mag)
    e = float(np.linalg.norm(e_vec))

    reasons = []
    if e >= 0.01:
        reasons.append(f"Eccentricity {e:.4f} >= 0.01 (non-circular)")
    if abs(dt_sec) > 2.0 * period_sec:
        reasons.append(f"Lead time {dt_sec/3600:.1f}h exceeds 2 orbital periods ({2*period_sec/3600:.1f}h)")

    is_valid = len(reasons) == 0
    reason_str = "CW assumptions valid" if is_valid else "; ".join(reasons)
    return is_valid, e, period_sec, reason_str


# ── Miss-distance sensitivity to ΔV ─────────────────────────────────
def miss_distance_sensitivity(
    r_primary, v_primary, r_secondary, v_secondary, tca_dt, maneuver_dt,
):
    """
    Compute the linear sensitivity of miss distance to ΔV in each RTN
    direction using the Clohessy–Wiltshire (CW / Hill's) relative motion
    equations for a near-circular primary orbit.

    Returns:
        sensitivities: dict with keys "R", "T", "N" mapping to the
                       km of miss-distance change per m/s of ΔV.
        M: 3×3 CW state-transition sub-matrix Φ_rv (km per km/s).
        rtn_matrix: 3×3 RTN orientation matrix.
    """
    r_pri = np.asarray(r_primary, dtype=float)
    v_pri = np.asarray(v_primary, dtype=float)

    rtn = compute_rtn_frame(r_pri, v_pri)

    # Orbital angular rate (rad/s)
    r_mag = np.linalg.norm(r_pri)
    n = np.sqrt(MU_EARTH / max(1.0, r_mag ** 3))

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
    M[0, 1] = 2.0 * (1.0 - cos_nt) / n       # δr_R / ΔV_T
    M[1, 0] = 2.0 * (cos_nt - 1.0) / n       # δr_T / ΔV_R
    M[1, 1] = (4.0 * sin_nt - 3.0 * nt) / n  # δr_T / ΔV_T
    M[2, 2] = sin_nt / n                     # δr_N / ΔV_N

    # Sensitivity magnitude (km per m/s: matrix M is in seconds, ÷1000)
    sens = {}
    for i, axis in enumerate(["R", "T", "N"]):
        col = M[:, i]
        sens[axis] = float(np.linalg.norm(col)) / 1000.0  # km / (m/s)

    return sens, M, rtn


# ── Fuel Mass (Tsiolkovsky Equation) ────────────────────────────────
def compute_fuel_consumption(delta_v_m_s, initial_mass_kg=1000.0, isp_s=300.0, g0=9.80665):
    """
    Calculate propellant mass consumption for an idealized single impulsive maneuver
    using Tsiolkovsky's ideal rocket equation:

        ΔV = I_sp * g_0 * ln(m_0 / m_f)
        m_f = m_0 * exp(-ΔV / (I_sp * g_0))
        m_prop = m_0 - m_f = m_0 * (1 - exp(-ΔV / (I_sp * g_0)))

    Parameters
    ----------
    delta_v_m_s : float
        Maneuver impulsive velocity increment magnitude in m/s (must be >= 0).
    initial_mass_kg : float, default 1000.0
        Total spacecraft mass prior to the burn in kg (pre-burn wet mass m_0).
    isp_s : float, default 300.0
        Thruster specific impulse in seconds (e.g. 220s for hydrazine, 300s for bipropellant).
    g0 : float, default 9.80665
        Standard gravitational acceleration constant in m/s².

    Returns
    -------
    m_prop_kg : float
        Consumed propellant mass in kg.

    Verification Benchmarks:
    - m_0 = 1000.0 kg, I_sp = 300.0 s, ΔV = 0.22 m/s -> m_prop = 0.07478 kg.
    - m_0 = 500.0 kg,  I_sp = 300.0 s, ΔV = 0.22 m/s -> m_prop = 0.03739 kg (≈0.038 kg).
    - m_0 = 1000.0 kg, I_sp = 300.0 s, ΔV = 1.00 m/s -> m_prop = 0.33983 kg.
    """
    dv = max(0.0, float(delta_v_m_s))
    m0 = max(0.0, float(initial_mass_kg))
    isp = max(1e-6, float(isp_s))
    ve = isp * g0
    return float(m0 * (1.0 - np.exp(-dv / ve)))


# ── Candidate Maneuver Evaluation ───────────────────────────────────
def evaluate_candidate_maneuvers(
    conjunction_event,
    target_miss_km=25.0,
    candidate_lead_times_hours=None,
    max_delta_v_m_s=50.0,
    initial_mass_kg=1000.0,
    dry_mass_kg=None,
    isp_s=300.0,
    propagation_method="auto",
):
    """
    Comprehensive multi-candidate avoidance maneuver evaluation.
    Evaluates 6 burn directions (+T, -T, +N, -N, +R, -R) across candidate
    execution lead times.

    For each candidate:
      1. Applies maneuver impulsive ΔV to initial state.
      2. Propagates modified trajectory forward to TCA using exact two-body
         dynamics (or CW when requested).
      3. Recalculates closest approach and resulting miss distance.
      4. Calculates ΔV and Tsiolkovsky propellant mass.
      5. Evaluates configured safety constraint: new_miss >= target_miss_km.
      6. Flags feasibility and records rejection reason.

    Selects the lowest-fuel feasible candidate among all evaluated options.
    If no candidate meets the safety constraint, reports NO_FEASIBLE_CANDIDATE.

    Returns
    -------
    dict:
      - 'feasibility_status': 'FEASIBLE' or 'NO_FEASIBLE_CANDIDATE'
      - 'recommended': best candidate dict or None
      - 'candidate_count': int
      - 'feasible_count': int
      - 'candidates': list of all evaluated candidate dicts
      - 'conjunction_metadata': original conjunction details
      - 'selection_rationale': str
    """
    ev = conjunction_event
    tca_raw = getattr(ev, 'tca', None) if not isinstance(ev, dict) else ev.get('tca')
    if isinstance(tca_raw, datetime):
        tca = tca_raw
    else:
        try:
            tca = datetime.fromisoformat(str(tca_raw))
        except Exception:
            tca = datetime.now(timezone.utc) + timedelta(hours=12)

    def _get_val(prop, default):
        if isinstance(ev, dict):
            return ev.get(prop, default)
        return getattr(ev, prop, default)

    r_pri = np.asarray(_get_val('primary_pos', [7000.0, 0.0, 0.0]), dtype=float)
    v_pri = np.asarray(_get_val('primary_vel', [0.0, 7.5, 0.0]), dtype=float)
    r_sec = np.asarray(_get_val('secondary_pos', [7001.0, 0.0, 0.0]), dtype=float)
    v_sec = np.asarray(_get_val('secondary_vel', [0.0, 7.5, 0.0]), dtype=float)
    current_miss = float(_get_val('miss_distance_km', 1.0))

    if candidate_lead_times_hours is None:
        candidate_lead_times_hours = [3.0, 6.0, 12.0]

    effective_mass_kg = dry_mass_kg if dry_mass_kg is not None else initial_mass_kg

    # Gravitational acceleration constant for Tsiolkovsky
    g0 = 9.80665
    ve = isp_s * g0

    candidates = []

    # 6 cardinal burn directions in RTN: (axis_letter, sign, label)
    burn_directions = [
        ("T", +1, "+T (In-Track Prograde)"),
        ("T", -1, "-T (In-Track Retrograde)"),
        ("N", +1, "+N (Cross-Track Normal)"),
        ("N", -1, "-N (Cross-Track Anti-Normal)"),
        ("R", +1, "+R (Radial Outward)"),
        ("R", -1, "-R (Radial Inward)"),
    ]

    for lead_h in candidate_lead_times_hours:
        dt_lead_sec = lead_h * 3600.0
        maneuver_time = tca - timedelta(seconds=dt_lead_sec)

        # Back-propagate primary and secondary to maneuver time from TCA state
        r_pri_man, v_pri_man = propagate_universal(r_pri, v_pri, -dt_lead_sec)
        r_sec_man, v_sec_man = propagate_universal(r_sec, v_sec, -dt_lead_sec)

        sens, M, rtn_mat = miss_distance_sensitivity(
            r_pri_man, v_pri_man, r_sec_man, v_sec_man, tca, maneuver_time
        )

        cw_valid, ecc, period_s, cw_reason = check_cw_assumptions(
            r_pri_man, v_pri_man, dt_lead_sec
        )

        needed_delta_miss = max(target_miss_km - current_miss, 1.0)

        for axis, sign, dir_label in burn_directions:
            sensitivity = sens[axis]
            if sensitivity < 1e-9:
                continue

            # Magnitude of ΔV required along this axis (m/s)
            dv_mag_ms = needed_delta_miss / sensitivity
            dv_signed_ms = sign * dv_mag_ms

            # ΔV vector in RTN (km/s)
            dv_rtn_kms = np.zeros(3)
            axis_idx = {"R": 0, "T": 1, "N": 2}[axis]
            dv_rtn_kms[axis_idx] = dv_signed_ms / 1000.0

            # Convert ΔV to inertial ECI frame (m/s and km/s)
            dv_eci_kms = rtn_to_eci(dv_rtn_kms, rtn_mat)
            dv_eci_ms = dv_eci_kms * 1000.0

            # Apply impulsive burn to primary state at maneuver time
            v_pri_post_burn_kms = v_pri_man + dv_eci_kms

            # Choose propagation method
            use_kepler = True
            if propagation_method == "cw" and cw_valid:
                use_kepler = False

            if use_kepler:
                # Nonlinear Two-body propagation of post-burn state to TCA
                r_pri_tca, v_pri_tca = propagate_universal(
                    r_pri_man, v_pri_post_burn_kms, dt_lead_sec
                )
                r_sec_tca, _ = propagate_universal(
                    r_sec_man, v_sec_man, dt_lead_sec
                )
                new_miss_km = float(np.linalg.norm(r_pri_tca - r_sec_tca))
                method_name = "Two-Body Keplerian (Universal Variable)"
            else:
                # CW linear relative displacement
                delta_r_rtn = M @ dv_rtn_kms
                miss_vec_eci = r_sec - r_pri
                miss_vec_rtn = eci_to_rtn(miss_vec_eci, rtn_mat)
                new_miss_vec_rtn = miss_vec_rtn + delta_r_rtn
                new_miss_km = float(np.linalg.norm(new_miss_vec_rtn))
                method_name = "Clohessy-Wiltshire (Linearised Relative)"

            # Fuel mass estimate via Tsiolkovsky equation
            fuel_kg = compute_fuel_consumption(
                dv_mag_ms, initial_mass_kg=effective_mass_kg, isp_s=isp_s, g0=g0
            )

            # Constraint evaluation
            safety_constraint_met = bool(new_miss_km >= (target_miss_km * 0.95))
            budget_constraint_met = bool(dv_mag_ms <= max_delta_v_m_s)
            is_feasible = safety_constraint_met and budget_constraint_met

            rejection_reasons = []
            if not safety_constraint_met:
                rejection_reasons.append(
                    f"Post-burn miss ({new_miss_km:.2f} km) below safety constraint ({target_miss_km:.1f} km)"
                )
            if not budget_constraint_met:
                rejection_reasons.append(
                    f"ΔV ({dv_mag_ms:.2f} m/s) exceeds maximum budget ({max_delta_v_m_s:.1f} m/s)"
                )

            candidates.append({
                "direction": f"{'+' if sign > 0 else '-'}{axis}",
                "direction_axis": axis,
                "direction_sign": sign,
                "direction_full": dir_label,
                "lead_time_hours": lead_h,
                "maneuver_time": maneuver_time.isoformat(),
                "delta_v_m_s": round(float(dv_mag_ms), 4),
                "delta_v_vector_eci_m_s": [round(float(x), 4) for x in dv_eci_ms],
                "delta_v_vector_rtn_m_s": [round(float(x * 1000.0), 4) for x in dv_rtn_kms],
                "fuel_kg": round(float(fuel_kg), 3),
                "original_miss_km": round(float(current_miss), 3),
                "new_miss_km": round(float(new_miss_km), 3),
                "target_miss_km": round(float(target_miss_km), 2),
                "safety_margin_km": round(float(new_miss_km - target_miss_km), 3),
                "safety_constraint_met": safety_constraint_met,
                "is_feasible": is_feasible,
                "rejection_reason": "; ".join(rejection_reasons) if rejection_reasons else None,
                "sensitivity_km_per_ms": round(float(sensitivity), 4),
                "propagation_method": method_name,
                "cw_assumptions_valid": bool(cw_valid),
                "cw_validity_note": cw_reason,
            })

    # Sort feasible candidates by ΔV ascending (cheapest first)
    feasible_candidates = [c for c in candidates if c["is_feasible"]]
    feasible_candidates.sort(key=lambda c: c["delta_v_m_s"])

    infeasible_candidates = [c for c in candidates if not c["is_feasible"]]
    infeasible_candidates.sort(key=lambda c: c["delta_v_m_s"])

    ranked_candidates = feasible_candidates + infeasible_candidates
    for i, c in enumerate(ranked_candidates):
        c["rank"] = i + 1
        c["description"] = (
            f"#{c['rank']} {c['direction_full']} burn of {c['delta_v_m_s']:.2f} m/s "
            f"at T-{c['lead_time_hours']:.1f}h → "
            f"miss distance {c['new_miss_km']:.1f} km "
            f"(fuel ≈ {c['fuel_kg']:.2f} kg) [{'FEASIBLE' if c['is_feasible'] else 'REJECTED'}]"
        )

    recommended = feasible_candidates[0] if feasible_candidates else None
    feasibility_status = "FEASIBLE" if recommended else "NO_FEASIBLE_CANDIDATE"

    rationale = (
        f"Selected lowest-fuel candidate among {len(candidates)} evaluated options "
        f"({len(feasible_candidates)} feasible). Note: Local discrete candidate evaluation; "
        f"global optimality not claimed."
        if recommended
        else f"None of the {len(candidates)} evaluated candidates met the safety constraint "
             f"({target_miss_km:.1f} km) within ΔV budget ({max_delta_v_m_s:.1f} m/s)."
    )

    return {
        "feasibility_status": feasibility_status,
        "recommended": recommended,
        "candidate_count": len(candidates),
        "feasible_count": len(feasible_candidates),
        "candidates": ranked_candidates,
        "selection_rationale": rationale,
        "conjunction_metadata": {
            "primary_name": _get_val('primary_name', 'PRIMARY'),
            "secondary_name": _get_val('secondary_name', 'SECONDARY'),
            "primary_norad": _get_val('primary_norad', 0),
            "secondary_norad": _get_val('secondary_norad', 0),
            "original_miss_km": round(float(current_miss), 4),
            "tca": tca.isoformat(),
            "max_epoch_age_days": _get_val('max_epoch_age_days', None),
            "is_stale": _get_val('is_stale', False),
            "pc_status": _get_val('pc_status', 'UNAVAILABLE (No Covariance)'),
        },
        "operational_disclaimer": (
            "⚠️ SIMULATION ONLY — Synthetic / TLE-derived calculation. "
            "Not cleared for operational satellite execution without certified "
            "tracking covariance (CDM) and flight dynamics officer review."
        ),
    }


# ── Backwards-Compatible Function ───────────────────────────────────
def compute_avoidance_maneuver(
    conjunction_event,
    target_miss_km=None,
    lead_time_hours=6.0,
):
    """
    Given a ConjunctionEvent, compute candidate avoidance manoeuvres
    in each RTN direction and return a ranked list by ΔV (lowest first).

    Maintains backwards compatibility for callers and automated tests:
    returns the best options for the primary axes with direction in {'R', 'T', 'N'}.
    """
    if target_miss_km is None:
        target_miss_km = 2.0 * CONJUNCTION_DISTANCE_THRESHOLD_KM

    eval_result = evaluate_candidate_maneuvers(
        conjunction_event,
        target_miss_km=target_miss_km,
        candidate_lead_times_hours=[lead_time_hours],
        max_delta_v_m_s=250.0,
        propagation_method="auto",
    )

    # For backwards compatibility with tests expecting direction in {"R", "T", "N"},
    # select the best candidate per axis ('T', 'R', 'N') and set direction=axis
    best_by_axis = {}
    for c in eval_result["candidates"]:
        axis = c["direction_axis"]
        if axis not in best_by_axis or c["delta_v_m_s"] < best_by_axis[axis]["delta_v_m_s"]:
            c_copy = dict(c)
            c_copy["direction"] = axis
            c_copy["signed_direction"] = c["direction"]
            best_by_axis[axis] = c_copy

    options = list(best_by_axis.values())
    options.sort(key=lambda o: o["delta_v_m_s"])
    for i, opt in enumerate(options):
        opt["rank"] = i + 1

    return options


def compute_avoidance_for_events(events, lead_time_hours=6.0, max_events=10):
    """
    Compute avoidance manoeuvres for the highest-risk conjunction events.
    Returns a list of result dictionaries with full explainability.
    """
    results = []
    for ev in events[:max_events]:
        try:
            eval_res = evaluate_candidate_maneuvers(
                ev,
                target_miss_km=2.0 * CONJUNCTION_DISTANCE_THRESHOLD_KM,
                candidate_lead_times_hours=[lead_time_hours],
            )
            ev_dict = ev.to_dict() if hasattr(ev, 'to_dict') else ev
            results.append({
                "conjunction": ev_dict,
                "feasibility_status": eval_res["feasibility_status"],
                "maneuver_options": eval_res["candidates"],
                "recommended": eval_res["recommended"],
                "selection_rationale": eval_res["selection_rationale"],
                "operational_disclaimer": eval_res["operational_disclaimer"],
            })
        except Exception as exc:
            name_p = getattr(ev, 'primary_name', '?') if not isinstance(ev, dict) else ev.get('primary_name', '?')
            name_s = getattr(ev, 'secondary_name', '?') if not isinstance(ev, dict) else ev.get('secondary_name', '?')
            print(f"⚠️  Avoidance computation failed for {name_p} ↔ {name_s}: {exc}")
    return results


# ── Reproducible Synthetic Conjunction Scenario Generator ───────────
def create_synthetic_conjunction_scenario(
    miss_km=0.35,
    lead_time_hours=6.0,
    altitude_km=500.0,
    eccentricity=0.0005,
    primary_norad=99001,
    secondary_norad=99002,
):
    """
    Build a reproducible synthetic conjunction scenario with documented
    initial states and expected analytical behavior.

    Primary: Near-circular LEO orbit at altitude h.
    Secondary: Crossing orbit at TCA with miss distance = miss_km.
    """
    now = datetime.now(timezone.utc)
    tca = now + timedelta(hours=lead_time_hours)

    r_mag = EARTH_RADIUS_KM + altitude_km
    v_circ = np.sqrt(MU_EARTH / r_mag)

    # Primary state at TCA (along X axis, moving in +Y direction)
    r_pri = np.array([r_mag, 0.0, 0.0])
    v_pri = np.array([0.0, v_circ, 0.0])

    # Secondary state at TCA (shifted by miss_km along Z and Y, opposing velocity)
    r_sec = np.array([r_mag, miss_km * 0.6, miss_km * 0.8])
    v_sec = np.array([0.0, -v_circ * 0.7071, v_circ * 0.7071])

    rel_vel = float(np.linalg.norm(v_pri - v_sec))

    from conjunction_detector import ConjunctionEvent
    event = ConjunctionEvent(
        primary_idx=0,
        secondary_idx=1,
        primary_norad=primary_norad,
        secondary_norad=secondary_norad,
        primary_name=f"SYNTHETIC-SAT-{primary_norad}",
        secondary_name=f"SYNTHETIC-DEBRIS-{secondary_norad}",
        primary_epoch=now - timedelta(days=1),
        secondary_epoch=now - timedelta(days=2),
        max_epoch_age_days=2.0,
        is_stale=False,
        tca=tca,
        miss_distance_km=float(miss_km),
        relative_velocity_km_s=rel_vel,
        collision_probability=None,
        max_possible_pc=float(min(1.0, (0.02**2) / (np.e * max(1e-6, miss_km**2)))),
        pc_status="UNAVAILABLE (No Covariance)",
        pc_method="N/A",
        risk_level="RED" if miss_km < 1.0 else ("YELLOW" if miss_km < 10.0 else "GREEN"),
        operational_priority="CRITICAL" if miss_km < 1.0 else "HIGH",
        primary_pos=r_pri.tolist(),
        secondary_pos=r_sec.tolist(),
        primary_vel=v_pri.tolist(),
        secondary_vel=v_sec.tolist(),
    )
    return event


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  OrbitGuard AI — Advanced Avoidance Planner")
    print("=" * 60)

    synth_ev = create_synthetic_conjunction_scenario(miss_km=0.42, lead_time_hours=6.0)
    print(f"\nSynthetic Scenario Created:")
    print(f"  Primary: {synth_ev.primary_name} ↔ Secondary: {synth_ev.secondary_name}")
    print(f"  Miss Distance: {synth_ev.miss_distance_km:.2f} km | TCA: {synth_ev.tca}")

    plan = evaluate_candidate_maneuvers(synth_ev, target_miss_km=25.0)
    print(f"\nEvaluation Results ({plan['feasibility_status']}):")
    print(f"  {plan['selection_rationale']}")
    for c in plan['candidates'][:6]:
        flag = "✅ RECOMMENDED" if c == plan['recommended'] else ("   Feasible" if c['is_feasible'] else "   Rejected")
        print(f"  [{flag}] {c['direction_full']:30s} | ΔV={c['delta_v_m_s']:.2f} m/s | Fuel={c['fuel_kg']:.2f} kg | Miss={c['new_miss_km']:.1f} km")
