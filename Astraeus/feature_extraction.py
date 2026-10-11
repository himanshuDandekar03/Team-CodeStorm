# orbital_pipeline/feature_extraction.py
import os
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

EARTH_RADIUS = 6371
MU = 398600.4418  # km^3/s^2

TLE_DIR = "tle_data"
CACHE_DIR = "tle_cache"  # New cache directory
OUTPUT_FILE = "orbital_features.csv"

def parse_tle_file(file_path, label):
    data = []
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]  # remove empty lines

    i = 0
    while i < len(lines) - 1:
        # Check if lines start with proper TLE markers
        if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
            try:
                sat = Satrec.twoline2rv(lines[i], lines[i+1])
                # Mean motion: sat.no_kozai is in radians/minute
                # Convert to rad/s for orbital mechanics, and revs/day for display/records
                mm_rad_min = sat.no_kozai
                if mm_rad_min == 0:
                    i += 2
                    continue
                n_rad_s = mm_rad_min / 60.0                  # rad/s
                mm_rev_day = mm_rad_min * 1440.0 / (2 * np.pi)  # revs/day

                # Orbital elements
                ecc = sat.ecco
                inc = sat.inclo
                raan = sat.nodeo
                argp = sat.argpo
                ma = sat.mo

                # Semi-major axis (km) using n in rad/s: a = (MU / n^2)^(1/3)
                a = (MU / (n_rad_s**2))**(1/3)
                alt = a - EARTH_RADIUS
                perigee = a * (1 - ecc)
                apogee = a * (1 + ecc)
                orbital_period = 2*np.pi*np.sqrt(a**3 / MU)

                # Propagate current position/velocity
                now_utc = datetime.now(timezone.utc)
                jd, fr = jday(now_utc.year, now_utc.month, now_utc.day,
                              now_utc.hour, now_utc.minute, now_utc.second)
                e_code, r, v = sat.sgp4(jd, fr)
                if e_code != 0:
                    i += 2
                    continue  # skip TLEs that fail propagation

                distance = np.linalg.norm(r)
                velocity_mag = np.linalg.norm(v)
                energy = velocity_mag**2 / 2 - MU/distance

                data.append({
                    'eccentricity': ecc,
                    'inclination': inc,
                    'raan': raan,
                    'arg_perigee': argp,
                    'mean_anomaly': ma,
                    'mean_motion': mm_rev_day,
                    'semi_major_axis': a,
                    'altitude': alt,
                    'perigee': perigee,
                    'apogee': apogee,
                    'orbital_period': orbital_period,
                    'distance': distance,
                    'velocity': velocity_mag,
                    'energy': energy,
                    'label': label,
                    # Position vector (km)
                    'pos_x': r[0],
                    'pos_y': r[1],
                    'pos_z': r[2],
                    # Velocity vector (km/s)
                    'vel_x': v[0],
                    'vel_y': v[1],
                    'vel_z': v[2]
                })
            except Exception as ex:
                print(f"Skipping invalid TLE in {file_path} at lines {i}-{i+1}: {ex}")
            i += 2
        else:
            # Skip any non-TLE lines (titles, comments)
            i += 1
    return data


# orbital_pipeline/feature_extraction.py
def parse_tle_lines(lines, label=0):
    """
    Same as parse_tle_file but works on raw TLE lines in memory.
    Returns a DataFrame with orbital features.
    Handles TLE format with optional satellite names and empty lines.
    """
    data = []
    
    # Clean lines - remove empty lines and strip whitespace
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    i = 0
    while i < len(clean_lines) - 1:
        # Look for TLE line 1
        if clean_lines[i].startswith('1 '):
            line1 = clean_lines[i]
            # Look for corresponding TLE line 2
            if i + 1 < len(clean_lines) and clean_lines[i + 1].startswith('2 '):
                line2 = clean_lines[i + 1]
                try:
                    sat = Satrec.twoline2rv(line1, line2)
                    mm_rad_min = sat.no_kozai
                    if mm_rad_min == 0:
                        i += 2
                        continue

                    n_rad_s = mm_rad_min / 60.0                  # rad/s
                    mm_rev_day = mm_rad_min * 1440.0 / (2 * np.pi)  # revs/day

                    ecc = sat.ecco
                    inc = sat.inclo
                    raan = sat.nodeo
                    argp = sat.argpo
                    ma = sat.mo

                    a = (MU / (n_rad_s**2))**(1/3)
                    alt = a - EARTH_RADIUS
                    perigee = a * (1 - ecc)
                    apogee = a * (1 + ecc)
                    orbital_period = 2*np.pi*np.sqrt(a**3 / MU)

                    now_utc = datetime.now(timezone.utc)
                    jd, fr = jday(now_utc.year, now_utc.month, now_utc.day, now_utc.hour, now_utc.minute, now_utc.second)
                    e_code, r, v = sat.sgp4(jd, fr)
                    if e_code != 0:
                        i += 2
                        continue

                    distance = np.linalg.norm(r)
                    velocity_mag = np.linalg.norm(v)
                    energy = velocity_mag**2 / 2 - MU/distance

                    data.append({
                        'eccentricity': ecc,
                        'inclination': inc,
                        'raan': raan,
                        'arg_perigee': argp,
                        'mean_anomaly': ma,
                        'mean_motion': mm_rev_day,
                        'semi_major_axis': a,
                        'altitude': alt,
                        'perigee': perigee,
                        'apogee': apogee,
                        'orbital_period': orbital_period,
                        'distance': distance,
                        'velocity': velocity_mag,
                        'energy': energy,
                        'label': label,
                        # Position vector (km)
                        'pos_x': r[0],
                        'pos_y': r[1],
                        'pos_z': r[2],
                        # Velocity vector (km/s)
                        'vel_x': v[0],
                        'vel_y': v[1],
                        'vel_z': v[2]
                    })
                except Exception as ex:
                    print(f"Skipping invalid TLE at line {i}: {ex}")
                
                i += 2  # Move to next potential TLE pair
            else:
                i += 1  # Line 1 found but no matching line 2, move forward
        else:
            i += 1  # Not a TLE line 1, move forward
    
    print(f"Processed {len(data)} valid satellites from TLE data")
    return pd.DataFrame(data)


def create_dataset():
    all_data = []
    
    # First try to use cached data
    cache_files = []
    if os.path.exists(CACHE_DIR):
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.txt')]
    
    # If we have recent cache files, use them
    if cache_files:
        print(f"📁 Found {len(cache_files)} cached TLE files")
        for file in cache_files:
            # Determine label from filename
            label = 0 if 'active' in file.lower() else 1
            try:
                data = parse_tle_file(os.path.join(CACHE_DIR, file), label)
                all_data.extend(data)
                print(f"Processed {len(data)} objects from {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Fallback to regular TLE_DIR files
    if not all_data and os.path.exists(TLE_DIR):
        print(f"📁 Using TLE files from {TLE_DIR}")
        for file in os.listdir(TLE_DIR):
            if not file.endswith('.txt'):
                continue
            # Label: 0 = active satellite, 1 = debris
            label = 0 if 'active' in file.lower() else 1
            try:
                data = parse_tle_file(os.path.join(TLE_DIR, file), label)
                all_data.extend(data)
                print(f"Processed {len(data)} objects from {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if not all_data:
        print("No valid TLEs found! Please run data collection first.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}, total rows: {len(df)}")
    print(f"   - Active satellites: {len(df[df['label'] == 0])}")
    print(f"   - Debris objects: {len(df[df['label'] == 1])}")

def get_cached_dataset():
    """Get dataset from cache or existing files without recreating"""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            print(f"Loaded existing dataset: {len(df)} objects")
            return df
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
    
    # If no existing dataset, create one
    print("Creating new dataset from available TLE data...")
    create_dataset()
    
    # Try to load the newly created dataset
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return df
        except Exception as e:
            print(f"Error loading newly created dataset: {e}")
    
    return None

if __name__ == "__main__":
    create_dataset()
