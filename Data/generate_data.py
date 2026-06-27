"""
Synthetic Login Event Generator
================================
Generates realistic user authentication telemetry for behavioral anomaly
detection research. Anomalies are injected with known ground truth labels,
enabling precision/recall evaluation of detection models.

Signal types modeled:
  - Impossible travel (geographic velocity exceeding physical possibility)
  - Typing speed deviation (behavioral biometric anomaly)
  - Click-through rate deviation (session behavioral anomaly)
  - Session timing anomaly (login outside established baseline hours)

Reference: MITRE ATT&CK T1078 - Valid Accounts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- Configuration ---
N_USERS = 50
NORMAL_EVENTS_PER_USER = 60   # ~2 months of daily logins
N_ANOMALIES = 100              # injected anomalies across all users
OUTPUT_PATH = "login_events.csv"

# City pool with lat/lon and timezone offset from UTC
CITIES = {
    "New York":     {"lat": 40.71, "lon": -74.01, "tz": -5},
    "Chicago":      {"lat": 41.88, "lon": -87.63, "tz": -6},
    "Los Angeles":  {"lat": 34.05, "lon": -118.24, "tz": -8},
    "Houston":      {"lat": 29.76, "lon": -95.37, "tz": -6},
    "Boston":       {"lat": 42.36, "lon": -71.06, "tz": -5},
    "Seattle":      {"lat": 47.61, "lon": -122.33, "tz": -8},
    "Atlanta":      {"lat": 33.75, "lon": -84.39, "tz": -5},
    "Denver":       {"lat": 39.74, "lon": -104.98, "tz": -7},
}

ANOMALY_CITIES = {
    "Beijing":      {"lat": 39.91, "lon": 116.39, "tz": 8},
    "Moscow":       {"lat": 55.75, "lon": 37.62,  "tz": 3},
    "Lagos":        {"lat": 6.52,  "lon": 3.38,   "tz": 1},
    "Tehran":       {"lat": 35.69, "lon": 51.39,  "tz": 3.5},
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometers."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def generate_normal_events():
    """Generate baseline login events for all users."""
    events = []
    base_date = datetime(2024, 1, 1, 8, 0, 0)

    for user_id in range(1, N_USERS + 1):
        # Each user has a home city they log in from consistently
        home_city_name = random.choice(list(CITIES.keys()))
        home_city = CITIES[home_city_name]

        # Baseline behavioral profile per user
        base_typing_speed = np.random.normal(65, 8)   # WPM
        base_ctr = np.random.normal(0.35, 0.05)       # click-through rate
        base_hour = np.random.randint(7, 10)           # typical login hour

        for i in range(NORMAL_EVENTS_PER_USER):
            event_time = base_date + timedelta(days=i, hours=np.random.normal(0, 0.5))
            login_hour = event_time.hour + np.random.randint(-1, 2)
            login_hour = max(6, min(22, base_hour + np.random.randint(-1, 2)))

            events.append({
                "event_id": f"U{user_id:03d}_N{i:04d}",
                "user_id": f"U{user_id:03d}",
                "timestamp": event_time.replace(hour=login_hour),
                "city": home_city_name,
                "latitude": home_city["lat"] + np.random.uniform(-0.05, 0.05),
                "longitude": home_city["lon"] + np.random.uniform(-0.05, 0.05),
                "typing_speed_wpm": max(20, np.random.normal(base_typing_speed, 5)),
                "click_through_rate": min(1.0, max(0.0, np.random.normal(base_ctr, 0.04))),
                "login_hour": login_hour,
                "is_anomaly": 0,
                "anomaly_type": "none",
                "resources_accessed": np.random.randint(2, 6),
                "data_volume_mb": round(np.random.uniform(0.5, 10.0), 2)
            })

    return events, {f"U{uid:03d}": list(CITIES.keys())[uid % len(CITIES)] for uid in range(1, N_USERS + 1)}


def inject_anomalies(events, user_home_cities):
    """Inject labeled anomalies into the event stream."""
    anomaly_events = []
    users = [f"U{uid:03d}" for uid in range(1, N_USERS + 1)]

    # Get last normal event timestamp per user for impossible travel calc
    df_normal = pd.DataFrame(events)
    last_event = df_normal.groupby("user_id")["timestamp"].max().to_dict()

    anomaly_types = {
        "impossible_travel":   int(N_ANOMALIES * 0.30),
        "typing_speed":        int(N_ANOMALIES * 0.20),
        "click_through":       int(N_ANOMALIES * 0.15),
        "session_timing":      int(N_ANOMALIES * 0.15),
        "privilege_escalation": int(N_ANOMALIES * 0.10),
        "data_exfiltration":   int(N_ANOMALIES * 0.10),
    }

    idx = 0
    for anomaly_type, count in anomaly_types.items():
        for _ in range(count):
            user = random.choice(users)
            home_city_name = user_home_cities.get(user, "New York")
            home_city = CITIES[home_city_name]
            last_ts = last_event.get(user, datetime(2024, 3, 1))

            if anomaly_type == "impossible_travel":
                # Login from foreign city within minutes of last domestic login
                anomaly_city_name = random.choice(list(ANOMALY_CITIES.keys()))
                anomaly_city = ANOMALY_CITIES[anomaly_city_name]
                minutes_later = random.randint(5, 30)
                event_time = last_ts + timedelta(minutes=minutes_later)
                distance_km = haversine_km(
                    home_city["lat"], home_city["lon"],
                    anomaly_city["lat"], anomaly_city["lon"]
                )
                speed_kmh = distance_km / (minutes_later / 60)

                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time,
                    "city": anomaly_city_name,
                    "latitude": anomaly_city["lat"],
                    "longitude": anomaly_city["lon"],
                    "typing_speed_wpm": np.random.normal(65, 8),
                    "click_through_rate": np.random.normal(0.35, 0.04),
                    "login_hour": event_time.hour,
                    "is_anomaly": 1,
                    "anomaly_type": "impossible_travel",
                    "implied_speed_kmh": round(speed_kmh, 1)
                })

            elif anomaly_type == "typing_speed":
                # Typing speed far outside user baseline (attacker unfamiliar with system)
                event_time = last_ts + timedelta(days=random.randint(1, 5), hours=random.randint(-2, 2))
                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time,
                    "city": home_city_name,
                    "latitude": home_city["lat"],
                    "longitude": home_city["lon"],
                    "typing_speed_wpm": np.random.choice([
                        np.random.normal(10, 3),   # far too slow
                        np.random.normal(130, 5)   # far too fast (scripted)
                    ]),
                    "click_through_rate": np.random.normal(0.35, 0.04),
                    "login_hour": random.randint(7, 10),
                    "is_anomaly": 1,
                    "anomaly_type": "typing_speed",
                    "implied_speed_kmh": None
                })

            elif anomaly_type == "click_through":
                event_time = last_ts + timedelta(days=random.randint(1, 5))
                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time,
                    "city": home_city_name,
                    "latitude": home_city["lat"],
                    "longitude": home_city["lon"],
                    "typing_speed_wpm": np.random.normal(65, 8),
                    "click_through_rate": np.random.choice([
                        np.random.uniform(0.0, 0.05),  # near zero — bot-like
                        np.random.uniform(0.90, 1.0)   # near perfect — scripted
                    ]),
                    "login_hour": random.randint(7, 10),
                    "is_anomaly": 1,
                    "anomaly_type": "click_through",
                    "implied_speed_kmh": None
                })

            elif anomaly_type == "session_timing":
                # Login at 3am — far outside user baseline
                event_time = last_ts + timedelta(days=random.randint(1, 5))
                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time.replace(hour=random.randint(1, 4)),
                    "city": home_city_name,
                    "latitude": home_city["lat"],
                    "longitude": home_city["lon"],
                    "typing_speed_wpm": np.random.normal(65, 8),
                    "click_through_rate": np.random.normal(0.35, 0.04),
                    "login_hour": random.randint(1, 4),
                    "is_anomaly": 1,
                    "anomaly_type": "session_timing",
                    "implied_speed_kmh": None
                })
            elif anomaly_type == "data_exfiltration":
                # Abnormally large data transfer volume — exfiltration signal
                event_time = last_ts + timedelta(days=random.randint(1, 5))
                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time,
                    "city": home_city_name,
                    "latitude": home_city["lat"],
                    "longitude": home_city["lon"],
                    "typing_speed_wpm": np.random.normal(65, 8),
                    "click_through_rate": np.random.normal(0.35, 0.04),
                    "login_hour": random.randint(7, 10),
                    "is_anomaly": 1,
                    "anomaly_type": "data_exfiltration",
                    "implied_speed_kmh": None,
                    "resources_accessed": np.random.randint(2, 6),
                    "data_volume_mb": round(np.random.uniform(500, 2000), 2)
                })
            elif anomaly_type == "privilege_escalation":
                # Abnormal resource access spike — privilege escalation signal
                event_time = last_ts + timedelta(days=random.randint(1, 5))
                anomaly_events.append({
                    "event_id": f"ANOM_{idx:04d}",
                    "user_id": user,
                    "timestamp": event_time,
                    "city": home_city_name,
                    "latitude": home_city["lat"],
                    "longitude": home_city["lon"],
                    "typing_speed_wpm": np.random.normal(65, 8),
                    "click_through_rate": np.random.normal(0.35, 0.04),
                    "login_hour": random.randint(7, 10),
                    "is_anomaly": 1,
                    "anomaly_type": "privilege_escalation",
                    "implied_speed_kmh": None,
                    "resources_accessed": np.random.randint(20, 50),
                    "data_volume_mb": round(np.random.uniform(0.5, 10.0), 2)
                })
            idx += 1
            last_event[user] = event_time

    return anomaly_events


if __name__ == "__main__":
    print("Generating normal events...")
    normal_events, user_home_cities = generate_normal_events()

    print("Injecting anomalies...")
    anomaly_events = inject_anomalies(normal_events, user_home_cities)

    all_events = normal_events + anomaly_events
    df = pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)
    df["implied_speed_kmh"] = df.get("implied_speed_kmh", None)

    df.to_csv(f"data/{OUTPUT_PATH}", index=False)

    print(f"\nDataset summary:")
    print(f"  Total events : {len(df)}")
    print(f"  Normal events: {df['is_anomaly'].eq(0).sum()}")
    print(f"  Anomalies    : {df['is_anomaly'].eq(1).sum()}")
    print(f"  Anomaly rate : {df['is_anomaly'].mean():.2%}")
    print(f"\nAnomaly breakdown:")
    print(df[df['is_anomaly']==1]['anomaly_type'].value_counts().to_string())
    print(f"\nSaved to data/{OUTPUT_PATH}")
