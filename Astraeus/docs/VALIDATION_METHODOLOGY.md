# OrbitGuard AI — Validation Methodology

**Version:** 1.0
**Date:** 2026-10-11

---

## 1. Reference Coordinate Frame

All orbital computations use the **TEME (True Equator Mean Equinox)**
frame, which is the native output of the SGP4/SDP4 propagator. Positions
are in **km**, velocities in **km/s**.

The RTN (Radial / In-Track / Cross-Track) frame is constructed from the
primary satellite's state vector:

```
R̂ = r / |r|                    (radial outward)
N̂ = (r × v) / |r × v|         (orbit normal, cross-track)
T̂ = N̂ × R̂                     (in-track, roughly along velocity)
```

The RTN matrix `M = [R̂; T̂; N̂]` is orthonormal with det(M) = +1
(right-handed). Transformations:
- ECI → RTN: `v_rtn = M · v_eci`
- RTN → ECI: `v_eci = M^T · v_rtn`

---

## 2. SGP4 Propagation

OrbitGuard uses the `python-sgp4` library (Brandon Rhodes), which
implements the USAF SGP4/SDP4 analytical propagator exactly as
documented in Hoots & Roehrich (1980) / Vallado et al. (2006).

**Key conventions:**
- `Satrec.no_kozai`: Kozai mean motion in **radians/minute**
- `Satrec.ecco`: Eccentricity (dimensionless)
- `Satrec.inclo`: Inclination in **radians**
- `Satrec.bstar`: B* drag term (1/Earth-radii)
- `sat.sgp4(jd, fr)` returns `(error_code, r_km, v_km_s)` in TEME

**Unit conversion:**
```python
n_rad_s = sat.no_kozai / 60.0     # rad/min → rad/s
n_rev_day = n_rad_s * 86400 / (2π)  # rad/s → rev/day
```

---

## 3. Derived Orbital Elements

From the mean motion `n` (rad/s), orbital elements are derived using
Keplerian two-body mechanics with μ_Earth = 398600.4418 km³/s²:

| Quantity | Formula | Units |
|---|---|---|
| Semi-major axis | a = (μ / n²)^(1/3) | km |
| Altitude | h = a − R_E (R_E = 6371 km) | km |
| Perigee radius | r_p = a(1 − e) | km |
| Apogee radius | r_a = a(1 + e) | km |
| Orbital period | T = 2π√(a³/μ) | seconds |
| Specific energy | ε = v²/2 − μ/r | km²/s² |

**Validation references:**
- ISS: a ≈ 6797 km, h ≈ 426 km, T ≈ 92.6 min, n ≈ 15.5 rev/day
- GEO: a ≈ 42164 km, h ≈ 35793 km, T ≈ 23.93 h, n ≈ 1.0 rev/day

---

## 4. Conjunction Detection

### 4.1 Coarse Screening

At each time-step, all object positions are inserted into a **scipy
cKDTree**. Pairs within the screening radius (default 100 km for full
scan, 50 km for dashboard quick-scan) are flagged as candidates.

**Complexity:** O(N log N) per time-step.

### 4.2 TCA Refinement

For each candidate pair, the Time of Closest Approach is refined using
**golden-section search** on the scalar distance function d(t) within
the bracketed interval [t_coarse − Δt, t_coarse + Δt].

Golden-section search converges to within `tol_sec` (default 0.1 s)
in at most 30 iterations (each halving the interval by the golden
ratio ≈ 0.618).

**Limitation:** Golden-section search finds a local minimum. If the
distance function is not unimodal within the bracket, the global
minimum may be missed. See `docs/LIMITATIONS.md` for details.

---

## 5. Collision Probability

### 5.1 Chan 2-D Pc Formulation (Chan 1997)

When operator-supplied covariance is available:

```
Pc = (R² / 2σ²) · exp(−d² / 2σ²)
```

where:
- R = combined hard-body radius (default 20 m = 0.02 km)
- d = miss distance at TCA (km)
- σ² = σ_primary² + σ_secondary² (combined 1-σ position uncertainty)

This is the isotropic-covariance simplification of the full Alfano/Chan
encounter-plane integral.

### 5.2 Covariance Availability

**TLE-only pairs:** Public catalogue TLEs do not include covariance
matrices. For these pairs, Pc is reported as `null` with
`pc_status: "UNAVAILABLE (No Covariance)"`.

Risk classification falls back to miss-distance thresholds:

| Miss Distance | Risk Level |
|---|---|
| < 1 km | RED |
| 1–10 km | YELLOW |
| ≥ 10 km | GREEN |

---

## 6. Collision Avoidance — Clohessy–Wiltshire Equations

### 6.1 CW State-Transition Matrix (Φ_rv)

For a near-circular primary orbit with angular rate
`n = √(μ/r³)` rad/s, the CW equations give the position
displacement at time `dt` after an impulsive ΔV:

```
δr_R = (sin(nt)/n)·ΔV_R + (2(1−cos(nt))/n)·ΔV_T
δr_T = (2(cos(nt)−1)/n)·ΔV_R + ((4sin(nt)−3nt)/n)·ΔV_T
δr_N = (sin(nt)/n)·ΔV_N
```

The 3×3 matrix `M` (Φ_rv sub-matrix) encodes this. The expected
sparsity pattern is:
- M[0,2] = M[1,2] = M[2,0] = M[2,1] = 0 (decoupled cross-track)

### 6.2 Post-Burn Miss Distance

The post-burn miss distance is computed by forward-propagating the
ΔV through the CW matrix:

```python
delta_r_rtn = M @ dv_rtn_kms          # CW displacement at TCA
miss_vec_rtn = eci_to_rtn(r_sec - r_pri, rtn_matrix)
new_miss_vec = miss_vec_rtn + delta_r_rtn
new_miss_km = |new_miss_vec|
```

### 6.3 Fuel Estimation

Propellant mass is computed via the Tsiolkovsky rocket equation:

```
m_fuel = m_sat · (1 − exp(−ΔV / (I_sp · g₀)))
```

Default parameters: m_sat = 1000 kg, I_sp = 300 s, g₀ = 9.80665 m/s².

---

## 7. Space Weather

### 7.1 NOAA SWPC Data Sources

| Parameter | Endpoint | Units |
|---|---|---|
| Kp index | `/products/noaa-planetary-k-index.json` | 0–9 |
| F10.7 flux | `/products/summary/10cm-flux.json` | SFU |
| Solar wind | `/products/summary/solar-wind-speed.json` | km/s |
| X-ray flux | `/json/goes/primary/xrays-6-hour.json` | W/m² |

### 7.2 Drag Multiplier (Empirical)

```
drag_mult = 1 + 0.015 · (F10.7 − 70) · exp(−altitude / 500)
```

**This is an empirical scaling approximation, NOT NRLMSISE-00.**
It models the qualitative relationship between solar flux, altitude,
and atmospheric density enhancement. It is bounded below at 0.2.

### 7.3 SGP4 Isolation

SGP4 uses the TLE's B* parameter for drag modelling. It does **not**
ingest real-time F10.7 or Kp values. The space weather module provides
environmental context and qualitative drag trends but does **not**
modify propagated trajectories.

---

## 8. Test Strategy

All tests use synthetic inputs with known expected outputs. No
network calls are made during testing.

| Test File | What It Validates |
|---|---|
| `test_orbital_propagation.py` | TLE parsing, unit conversion, derived elements, SGP4 vectors |
| `test_conjunction_detector.py` | kd-tree filtering, Pc formula, risk classification, serialisation |
| `test_avoidance_maneuver.py` | RTN frame, CW matrix, Tsiolkovsky, forward propagation |
| `test_space_weather.py` | X-ray classes, drag formula, G-scale, SGP4 isolation |
