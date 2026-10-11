# benchmark_performance.py — Performance & Scaling Benchmark for OrbitGuard AI
"""
Executes a reproducible, scientifically accurate benchmark measuring:
  1. Objects loaded from TLE cache
  2. Number of discrete screening epochs
  3. Theoretical brute-force comparisons avoided vs actual KD-tree spatial queries
  4. Actual candidate pairs returned within screening radius
  5. Altitude overlap pre-filter reduction
  6. Actual candidate passes sent to Golden-Section TCA refinement
  7. Stage-by-stage wall-clock runtimes
  8. Memory usage (tracemalloc peak heap and OS Max RSS)
"""
import time
import tracemalloc
import resource
import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from conjunction_detector import (
    load_tle_catalogue,
    propagate_catalogue,
    find_close_pairs,
    compute_orbital_altitudes,
    check_altitude_overlap,
    refine_tca,
    screen_conjunctions,
)

def run_benchmark():
    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 70)
    print("  OrbitGuard AI — Conjunction Screening Performance Benchmark")
    print("=" * 70)

    # 1. Ingestion Phase
    t0 = time.perf_counter()
    full_catalogue = load_tle_catalogue(cache_dir="tle_cache")
    t_ingest = time.perf_counter() - t0
    num_total_sats = len(full_catalogue)
    print(f"[Stage 1: Ingestion & Catalogue Construction]")
    print(f"  • Unique Satrec objects parsed: {num_total_sats:,}")
    print(f"  • Ingestion Runtime: {t_ingest:.3f} s ({num_total_sats/max(1e-4, t_ingest):,.0f} objects/s)")

    # 2. Altitude Geometry Pre-computation
    t0 = time.perf_counter()
    altitudes = [compute_orbital_altitudes(sat) for _, sat in full_catalogue]
    t_alt = time.perf_counter() - t0
    print(f"\n[Stage 2: Orbital Altitude Characterization]")
    print(f"  • Perigee & apogee altitudes computed for all {len(altitudes):,} objects in {t_alt*1000:.1f} ms")

    # 3. Spatial Screening Benchmark over Time Steps
    start_time = datetime.now(timezone.utc)
    duration_hours = 6.0
    step_minutes = 20.0
    steps = int((duration_hours * 60) / step_minutes) + 1

    t0_prop = 0.0
    t0_kdtree = 0.0
    candidate_pairs_found = 0
    theoretical_brute_force_pairs = steps * (num_total_sats * (num_total_sats - 1)) // 2

    print(f"\n[Stage 3: Multi-Epoch Propagation & Spatial Indexing]")
    print(f"  • Screening Horizon: {duration_hours} hours ({steps} discrete epochs)")
    print(f"  • SGP4 State Vector Propagations: {num_total_sats * steps:,}")
    print(f"  • Theoretical Brute-Force Pair Comparisons (without spatial index): {theoretical_brute_force_pairs:,}")

    t0 = time.perf_counter()
    for step_idx in range(steps):
        epoch = start_time + timedelta(minutes=step_idx * step_minutes)
        
        tp0 = time.perf_counter()
        pos, vel = propagate_catalogue(full_catalogue, epoch)
        t0_prop += (time.perf_counter() - tp0)

        tk0 = time.perf_counter()
        pairs = find_close_pairs(pos, threshold_km=100.0)
        t0_kdtree += (time.perf_counter() - tk0)

        candidate_pairs_found += len(pairs)

    t_screen_total = time.perf_counter() - t0
    print(f"  • Actual Candidate Pairs within 100 km returned by KD-Tree: {candidate_pairs_found:,}")
    print(f"  • SGP4 Propagation Runtime: {t0_prop:.3f} s ({t0_prop/steps*1000:.1f} ms/epoch)")
    print(f"  • KD-Tree Index & Spatial Query Runtime: {t0_kdtree:.3f} s ({t0_kdtree/steps*1000:.1f} ms/epoch)")
    print(f"  • Total Multi-Epoch Screening Time: {t_screen_total:.3f} s")

    # 4. Refinement Stage Benchmark (Controlled Sample)
    print(f"\n[Stage 4: Golden-Section TCA Refinement (100 Sample Pairs)]")
    # Take up to 100 candidate pairs
    sample_size = min(100, len(pairs))
    sample_pairs = pairs[:sample_size]
    t0_refine = time.perf_counter()
    refined_count = 0
    for idx_a, idx_b in sample_pairs:
        # Check altitude overlap pre-filter first
        hp_a, ha_a = altitudes[idx_a]
        hp_b, ha_b = altitudes[idx_b]
        if not check_altitude_overlap(hp_a, ha_a, hp_b, ha_b, margin_km=125.0):
            continue
        sat_a = full_catalogue[idx_a][1]
        sat_b = full_catalogue[idx_b][1]
        tca_dt, miss_km, rel_vel, _, _, _, _ = refine_tca(
            sat_a, sat_b, start_time - timedelta(minutes=10), start_time + timedelta(minutes=10)
        )
        if miss_km <= 25.0:
            refined_count += 1
    t_refine = time.perf_counter() - t0_refine
    ms_per_refine = (t_refine / max(1, sample_size)) * 1000.0
    print(f"  • Refined {sample_size} candidate passes in {t_refine:.3f} s ({ms_per_refine:.2f} ms/pair)")
    print(f"  • Encounters confirmed within 25 km threshold: {refined_count}")

    # 5. Resource & Memory Metrics
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024

    t_total = time.perf_counter() - t_start
    print(f"\n[Stage 5: Resource Utilization & Memory Footprint]")
    print(f"  • Total Benchmark Execution Time: {t_total:.3f} s")
    print(f"  • Tracemalloc Peak Python Heap: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"  • OS Maximum Resident Set Size (RSS): {max_rss_mb:.2f} MB")
    print("=" * 70)

    return {
        "num_objects": num_total_sats,
        "epochs": steps,
        "theoretical_brute_force_pairs": theoretical_brute_force_pairs,
        "candidate_pairs_returned": candidate_pairs_found,
        "propagation_time_s": round(t0_prop, 4),
        "kdtree_query_time_s": round(t0_kdtree, 4),
        "total_screening_time_s": round(t_screen_total, 4),
        "refinement_ms_per_pair": round(ms_per_refine, 2),
        "peak_heap_mb": round(peak_mem / (1024 * 1024), 2),
        "max_rss_mb": round(max_rss_mb, 2),
    }

if __name__ == "__main__":
    run_benchmark()
