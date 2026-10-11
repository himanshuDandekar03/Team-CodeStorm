# OrbitGuard AI — Known Limitations

**Version:** 1.0
**Date:** 2026-10-11

---

## 1. TLE Accuracy & Epoch Staleness

**Limitation:** Two-Line Elements (TLEs) from public catalogues are
low-fidelity state estimates. Position uncertainty grows rapidly after
the TLE epoch:

| Time Since TLE Epoch | Approximate Position Error (LEO) |
|---|---|
| 0 hours | 1–5 km |
| 24 hours | 5–20 km |
| 72 hours | 20–100+ km |

**Impact:** Conjunction predictions at the end of the 72-hour screening
horizon may have significant positional error. Users should not rely on
TLE-based conjunction assessments as operational collision avoidance
decisions without supplementary tracking data.

**Mitigation:** OrbitGuard downloads fresh TLEs from CelesTrak and uses
the most recently cached versions. However, CelesTrak typically lags
1–12 hours behind real-time tracking.

---

## 2. No Covariance Data

**Limitation:** Public TLEs do not include covariance (uncertainty)
matrices. Without covariance, the collision probability (Pc) cannot
be computed.

**Impact:** OrbitGuard reports `collision_probability: null` and
`pc_status: "UNAVAILABLE (No Covariance)"` for all TLE-sourced
conjunction events. Risk classification uses miss-distance thresholds
instead (< 1 km RED, < 10 km YELLOW, ≥ 10 km GREEN).

**What would be needed:** Conjunction Data Messages (CDMs) from the
18th Space Defense Squadron or equivalent operators, which include
covariance matrices in the encounter plane.

---

## 3. Coarse Sampling Blind Spots

**Limitation:** The coarse screening phase samples positions at
discrete time intervals (default 300–600 seconds). Two satellites
with relative velocity of 15 km/s travel 4,500–9,000 km between
samples. Close approaches occurring strictly between sample points
may be missed by the kd-tree spatial filter.

**Impact:** Some genuine conjunction events could be undetected.
The probability of missing a close approach increases with:
- Higher relative velocity
- Larger coarse step size
- Smaller screening volume threshold

**Mitigation:**
- The screening volume (50–100 km) is intentionally larger than the
  conjunction threshold (25 km) to provide a buffer
- TCA refinement uses golden-section search at 0.1 s precision
  within bracketed intervals

**Unimplemented alternative:** An orbital apogee/perigee proximity
pre-filter and Δr·Δv = 0 zero-crossing root-finder would provide
more rigorous detection guarantees but would increase computational cost.

---

## 4. Golden-Section TCA Refinement

**Limitation:** Golden-section search finds a local minimum of the
distance function d(t) assuming it is unimodal within the bracketed
interval. If two objects pass each other multiple times within a
single bracket (e.g., co-planar LEO objects with slightly different
periods), the search may converge to a local minimum that is not the
global closest approach.

**Impact:** The reported TCA may not correspond to the actual minimum
miss distance within the screening horizon.

**Mitigation:** The bracket is set to [t_coarse ± Δt_coarse], which
is typically 300–600 seconds. For most encounter geometries, the
distance function is unimodal within this interval.

---

## 5. Clohessy–Wiltshire (CW) Linearisation

**Limitation:** The CW relative motion equations assume:
1. The primary orbit is **circular** (eccentricity = 0)
2. The relative separation is **small** compared to the orbit radius
3. No perturbations (J2, drag, SRP) during the drift period

**Impact:** For eccentric orbits, large miss distances, or long lead
times, the CW-predicted post-burn miss distance may differ from the
true miss distance.

**Accuracy envelope:**
- For e < 0.01 and lead time < 1 orbit: error < 5%
- For e > 0.05 or lead time > 5 orbits: error may exceed 20%

**Mitigation:** Users can reduce lead time to improve CW accuracy.
A full numerical propagation of the post-burn state would eliminate
this linearisation error but is not yet implemented.

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

## 7. ML Model Limitations

### 7.1 Debris Detection Ensemble

- Trained on CelesTrak active/debris labels which may contain errors
- No independent ground-truth validation dataset was used
- Accuracy, precision, recall, and F1 were not measured against
  reference catalogues (e.g., ESA DISCOS)

### 7.2 RNN Trajectory Prediction

- The LSTM/GRU models predict future orbital features but were not
  validated against SGP4 reference trajectories
- Prediction error over 24/48/72 hours has not been measured
- These models should not be used for operational conjunction assessment

---

## 8. Dashboard & API Security

- The default admin credentials (`admin` / `astraeus2025`) are
  hard-coded and should be changed in production
- Session management uses Flask's default cookie-based sessions
- No HTTPS enforcement, rate limiting, or CSRF protection
- API endpoints are protected by session check but are vulnerable
  to session fixation

---

## 9. Performance Limitations

- Full-catalogue conjunction screening (N > 10,000 objects over 72h)
  may take several minutes on a single CPU
- The background data update thread runs every 5 minutes; real-time
  updates are not supported
- No database persistence — all data is held in memory and lost on
  server restart
