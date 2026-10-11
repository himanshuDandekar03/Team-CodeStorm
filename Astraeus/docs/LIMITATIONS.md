# OrbitGuard AI — Known Limitations

**Version:** 1.0
**Date:** 2026-10-11

---

## 1. TLE Accuracy, SGP4 Degradation, and Epoch Staleness

**Limitation:** Two-Line Element (TLE) sets and SGP4 propagation are analytical approximations of orbital trajectories. Ephemeris error growth is non-linear and cannot be represented by a single universal lookup table.

As established by astrodynamics literature (Vallado & Kelso 2013; Flohrer et al. 2008), SGP4 positional error growth depends heavily on:
1. **Perigee Altitude:** Atmospheric drag is the dominant source of in-track error in LEO (< 600 km). At 350 km altitude, unmodeled drag density fluctuations can cause errors to grow by tens of kilometers in 24 hours. In higher LEO (> 800 km) and MEO/GEO, drag is minimal and errors grow far more slowly.
2. **Space Weather Activity:** Solar storms (elevated $F_{10.7} > 150$ sfu, geomagnetic storms $Kp \ge 5$) heat the upper thermosphere, dramatically inflating atmospheric density and causing sudden in-track ephemeris divergence.
3. **Ballistic Coefficient ($B^*$):** High area-to-mass debris objects (e.g. solar panel fragments) experience much faster orbit decay and SGP4 degradation than dense, compact payloads.
4. **Tracking Recency:** Space surveillance sensor observation frequency dictates the quality of the TLE fit at epoch.

**Operational Distinction:** OrbitGuard AI displays **Orbital Data Age** (time elapsed since TLE epoch) and **Prediction Horizon** (time from now to TCA) as separate, factual metadata fields. The system does not present unvalidated heuristic error bounds as true physical covariances.

**Mitigation:** CelesTrak cache staleness is tracked and flagged (`is_stale = True` when epoch age exceeds 3.0 days).

---

## 2. No Covariance Data & Collision Probability Bounds

**Limitation:** Public TLEs do not include covariance (uncertainty) matrices. Without covariance, the physical collision probability ($P_c$) cannot be computed.

**Implementation Standard:**
- OrbitGuard AI reports `collision_probability: null` and `pc_status: "UNAVAILABLE (No Covariance)"` for all TLE-sourced conjunction events.
- An auxiliary metric `max_possible_pc` is calculated using Alfano's (2005) maximum dilution formula:
  $$P_{c,\max} = \frac{R^2}{e \cdot d^2}$$
  This is strictly an **analytical upper bound** under the worst-case isotropic covariance assumption, NOT an actual collision probability.
- Risk classification uses calibrated miss-distance thresholds (< 1.0 km RED, < 10.0 km YELLOW, ≥ 10.0 km GREEN) and operational alert priority ranking.

**What is needed for operational Pc:** Conjunction Data Messages (CDMs, CCSDS 508.0-B-1) containing $3 \times 3$ encounter-frame or $6 \times 6$ inertial covariance matrices.

---

## 3. Coarse Sampling Blind Spots & Resolution Methods

**Limitation:** When screening over a prediction horizon using discrete time intervals (e.g., $\Delta t = 600$ seconds), two high-speed objects with relative velocity $v_{\text{rel}} \approx 12$ km/s travel 7,200 km between sample steps. If their encounter sphere transit time is only 10–15 seconds, snapshot spatial filters (such as KD-Trees) evaluated strictly at $t_k$ and $t_{k+1}$ can miss close approaches occurring midway between samples.

**Mitigations Implemented in OrbitGuard AI:**
1. **Range-Rate Zero-Crossing Projection:** For candidate pairs, evaluates $\tau = -\frac{\Delta \mathbf{r} \cdot \Delta \mathbf{v}}{\|\Delta \mathbf{v}\|^2}$ to estimate encounter time between steps.
2. **Analytic Segment Minimum Distance:** Closed-form minimum distance calculation over linear segments $[t_k, t_{k+1}]$ via `segment_minimum_distance()`.
3. **Orbital Altitude Pre-Filtering:** Pre-computes perigee and apogee altitudes via `compute_orbital_altitudes()` and eliminates radially disjoint orbits ($[h_{p1}, h_{a1}] \cap [h_{p2}, h_{a2}] = \emptyset$) using `check_altitude_overlap()`.
4. **Adaptive Refinement Bracketing:** Centered around the projected encounter epoch rather than discrete grid points.

---

## 4. Golden-Section TCA Refinement & Numerical Tolerance

**Numerical vs. Physical Uncertainty Distinction:**
- **Numerical Stopping Criterion:** Golden-section search converges when the search bracket width $\Delta t < \text{tol\_sec}$ (default $0.1$ seconds). This is purely a mathematical stopping criterion for 1D function minimization.
- **Physical Ephemeris Uncertainty:** Achieving $0.1$ s numerical convergence does **not** imply that the real-world satellite encounter time is known to $0.1$ s accuracy. Physical ephemeris uncertainty in SGP4 TLE predictions in LEO is typically **$\pm 5$ to $\pm 60+$ seconds** due to atmospheric density fluctuations, tracking observation spacing, and geopotential errors.

**Local vs. Global Minimum:** Golden-section search finds a local minimum assuming unimodality within $[t_{\text{coarse}} \pm \Delta t]$. If co-planar objects pass each other multiple times within an enlarged bracket, multi-pass tracking ensures separate passes (>45 min) are tracked independently.

---

## 5. Avoidance Maneuver Modeling & Optimization Scope

**1. Universal Keplerian vs. CW Linearisation:**
- The Clohessy-Wiltshire (Hill's) equations assume a circular primary orbit ($e = 0$) and linear relative gravity.
- OrbitGuard AI uses CW sensitivity solely to guide initial directional selection, while **all modified candidate trajectories are propagated using exact non-linear Universal Variable Keplerian mechanics (`propagate_universal()`, Vallado Algorithm 8)**.
- `check_cw_assumptions()` explicitly flags when CW linear assumptions are violated ($e \ge 0.01$ or drift $> 2$ orbits).

**2. Single-Pair Separation (No Multi-Object Collision Chaining):**
- The maneuver planner evaluates separation exclusively for the specific primary–secondary conjunction pair.
- In dense orbital environments (e.g., Starlink constellation shells or Sun-Synchronous Orbits), an impulsive maneuver that successfully clears the primary secondary object could theoretically reduce clearance with a third, unmaneuvered catalog satellite.
- OrbitGuard AI does not yet perform full-catalog secondary collision chaining for candidate maneuvers.

**3. Evaluated Candidates vs. Global Optimality:**
- The optimizer evaluates discrete candidate maneuvers (6 cardinal RTN burn directions: $+T, -T, +N, -N, +R, -R$ across candidate lead times).
- The recommended candidate is the **lowest-fuel feasible candidate among those evaluated**. Global optimality across continuous burn vectors and continuous execution times is neither claimed nor proven.

---

## 6. Space Weather & Atmospheric Drag

**Limitation:** The atmospheric drag multiplier uses an empirical
formula:

```
drag_mult = 1 + 0.015 · (F10.7 − 70) · exp(−alt / 500)
```

This is **NOT** the NRLMSISE-00 or JB2008 atmospheric density model.
It provides qualitative drag trend indication only.

**Impact:** The drag multiplier should not be used for quantitative
orbit lifetime predictions or precise drag force computations.

**SGP4 isolation:** SGP4 internally models drag via the TLE's B*
parameter. The space weather module's F10.7/Kp data does **not**
modify SGP4 propagation in any way.

---

## 7. Machine Learning Assessment & Scientific Deprecation

### 7.1 Debris Detection Ensemble
- Trained on CelesTrak TLE filenames where label = 0 if 'active' in filename else 1.
- **Misrepresentation:** The model outputs a binary active-satellite vs. debris fragment classification score based on static orbital elements and instantaneous snapshot position. It does **not** predict collision probability or collision risk.
- **Operational Reality:** A debris fragment in GEO has a high debris score but zero collision probability with a LEO satellite. OrbitGuard AI does not use this score for conjunction risk.

### 7.2 RNN "Trajectory Prediction" Pseudomodel
- **Critical Code Finding:** Lines 341–343 of `model_training.py` unsqueeze length-1 tensors and train `TrajRNN` on `TensorDataset(X, X)` to minimize `loss(out, X)`.
- The model functions as a static scalar autoencoder reconstructing feature vectors of length 1. It contains zero temporal steps, zero numerical propagation, and zero orbital physics.
- **Status:** Formally deprecated. SGP4 and universal variable two-body propagation (`propagate_universal`) are the only scientifically valid trajectory engines in OrbitGuard AI.

---

## 8. Dashboard & API Security

- The default admin credentials (`admin` / `astraeus2025`) are hard-coded in demonstration mode and should be changed in production
- Session management uses Flask cookie-based sessions
- No HTTPS enforcement, rate limiting, or CSRF protection
- API endpoints are protected by session check but are vulnerable to session fixation

---

## 9. Performance & Scaling Bottlenecks

### 9.1 Measured Performance on Full Catalog (19,552 objects)
- **Object Ingestion:** 19,552 Satrec objects in 0.418 s.
- **Coarse Spatial Screening (KD-Tree):** 3.63 billion equivalent pairwise checks evaluated in 1.392 s (73.2 ms per discrete epoch).
- **Phase 2 TCA Refinement Latency:** Because dense LEO constellations (e.g. Starlink) share common orbital shells, a 75 km coarse filter over 12 hours identifies 129,426 candidate passes. Evaluating golden-section 1D minimization serially in Python across 129,426 passes creates a ~2-minute latency.
- **Remedy:** Implement an analytic orbital apogee/perigee altitude overlap pre-filter ($[q_1 - \delta, Q_1 + \delta] \cap [q_2 - \delta, Q_2 + \delta] \ne \emptyset$) to eliminate non-intersecting coplanar shells prior to TCA refinement.

