"""
features.py
===========
Data ingestion and feature engineering for the behavioral anomaly
detection pipeline.
"""

import sys
import os
import pandas as pd

# Resolve Data directory relative to this file's location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from generate_data import haversine_km


def load_and_engineer(path):
    """
    Load raw login events and engineer all behavioral features.

    Returns
    -------
    df : pd.DataFrame
        Full event log with engineered columns appended.
    features : pd.DataFrame
        Numerical feature matrix ready for model input.
    """
    df = pd.read_csv(path)
    df = df.astype({'timestamp': 'datetime64[ns]'}).sort_values(
        ['user_id', 'timestamp']
    ).reset_index(drop=True)

    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['prev_latitude']  = df.groupby('user_id')['latitude'].shift(1)
    df['prev_longitude'] = df.groupby('user_id')['longitude'].shift(1)

    df['hours_since_last_login'] = (
        (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600
    )

    df['distance_from_last_login'] = df.apply(
        lambda row: haversine_km(
            row['latitude'], row['longitude'],
            row['prev_latitude'], row['prev_longitude']
        ), axis=1
    )

    df['velocity_kmh'] = df.apply(
        lambda row: row['distance_from_last_login'] / row['hours_since_last_login']
        if row['hours_since_last_login'] > 0 else 0, axis=1
    )

    features = df[[
        'typing_speed_wpm',
        'click_through_rate',
        'login_hour',
        'hours_since_last_login',
        'distance_from_last_login',
        'velocity_kmh',
        'resources_accessed',
        'data_volume_mb'
    ]].fillna(0)

    return df, features
