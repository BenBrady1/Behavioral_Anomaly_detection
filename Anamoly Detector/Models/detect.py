"""
Behavioral Anomaly Detection via Isolation Forest
==================================================
Implements an unsupervised machine learning pipeline for detecting
identity-based threats in authentication event telemetry. Detection
signals are derived from behavioral biometrics and geospatial velocity
analysis, mapped to MITRE ATT&CK T1078 (Valid Accounts).

Methodology:
    Feature engineering extracts six behavioral signals from raw login
    events. An Isolation Forest model (Liu et al., 2008) is trained on
    the full dataset without labeled anomalies, exploiting the principle
    that anomalous observations require fewer random partitions to isolate
    than normal observations. Ground truth labels, available due to the
    synthetic nature of the dataset, enable rigorous precision/recall
    evaluation.

References:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest.
        IEEE International Conference on Data Mining, 413-422.
    MITRE ATT&CK. (2024). Valid Accounts (T1078).
        https://attack.mitre.org/techniques/T1078/
    NIST Special Publication 800-207. (2020). Zero Trust Architecture.
        https://csrc.nist.gov/pubs/sp/800/207/final
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------------
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Local module import
# Extend the Python path to include the Data directory so the Haversine
# distance function defined in the data generation module can be reused
# without duplication, consistent with the DRY (Don't Repeat Yourself)
# principle.
# ---------------------------------------------------------------------------
sys.path.append('Data')
from generate_data import haversine_km

# ---------------------------------------------------------------------------
# Visualization constants
# Consistent color palette applied across all figures for visual coherence.
# ---------------------------------------------------------------------------
COLOR_NORMAL  = 'steelblue'   # All normal (non-anomalous) observations
COLOR_ANOMALY = 'crimson'     # All anomalous observations (any type)
COLOR_TRAVEL  = '#E63946'     # Impossible travel anomalies
COLOR_TYPING  = '#F4A261'     # Typing speed anomalies
COLOR_CTR     = '#2A9D8F'     # Click-through rate anomalies
COLOR_TIMING  = '#9B5DE5'     # Session timing anomalies
ALPHA_NORMAL  = 0.35          # Transparency for normal point mass
ALPHA_ANOMALY = 0.85          # Transparency for anomaly highlights

# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------
# Load the synthetic authentication event log. The timestamp column is
# parsed at ingestion to avoid downstream type coercion errors. Events are
# sorted by user and time to enable per-user sequential feature derivation
# in subsequent steps.
# ---------------------------------------------------------------------------
df = pd.read_csv('Data/login_events.csv')
df = df.astype({'timestamp': 'datetime64[ns]'}).sort_values(
    ['user_id', 'timestamp']
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING — SEQUENTIAL BEHAVIORAL SIGNALS
# ---------------------------------------------------------------------------
# For each login event, derive the prior login timestamp and geographic
# coordinates within the same user context. The pandas shift(1) operation
# retrieves the immediately preceding record per group, producing NaN for
# each user's first event (no prior baseline exists).
# ---------------------------------------------------------------------------
df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
df['prev_latitude']  = df.groupby('user_id')['latitude'].shift(1)
df['prev_longitude'] = df.groupby('user_id')['longitude'].shift(1)

# Temporal delta: elapsed time in hours between consecutive logins.
# Used as the denominator in geospatial velocity calculation.
df['hours_since_last_login'] = (
    (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600
)

# Geospatial delta: great-circle distance in kilometers between the current
# and prior login location, computed via the Haversine formula.
# Rows with missing prior coordinates (first login per user) return NaN.
df['distance_from_last_login'] = df.apply(
    lambda row: haversine_km(
        row['latitude'], row['longitude'],
        row['prev_latitude'], row['prev_longitude']
    ), axis=1
)

# Geospatial velocity: kilometers per hour implied by the location change
# and elapsed time. Values exceeding approximately 1,200 km/h (commercial
# aviation speed) indicate physically impossible travel — the primary
# detection signal for MITRE ATT&CK T1078 credential compromise.
# Division-by-zero protection is applied for same-second consecutive events.
df['velocity_kmh'] = df.apply(
    lambda row: row['distance_from_last_login'] / row['hours_since_last_login']
    if row['hours_since_last_login'] > 0 else 0, axis=1
)

# ---------------------------------------------------------------------------
# 3. FEATURE MATRIX CONSTRUCTION
# ---------------------------------------------------------------------------
# Retain only numerical features required by the Isolation Forest model.
# Categorical identifiers (user_id, city, anomaly_type) and metadata columns
# are excluded. NaN values arising from first-login records are imputed
# with zero, representing the absence of a velocity or distance signal
# rather than a meaningful behavioral measurement.
# ---------------------------------------------------------------------------
features = df[[
    'typing_speed_wpm',
    'click_through_rate',
    'login_hour',
    'hours_since_last_login',
    'distance_from_last_login',
    'velocity_kmh'
]]
features = features.fillna(0)

# ---------------------------------------------------------------------------
# 4. MODEL TRAINING — ISOLATION FOREST
# ---------------------------------------------------------------------------
# The Isolation Forest algorithm (Liu et al., 2008) detects anomalies by
# recursively partitioning the feature space using random splits. Anomalous
# observations, being few and distinct, are isolated in fewer splits than
# normal observations, yielding shorter average path lengths across the
# ensemble of trees.
#
# contamination=0.025 reflects the known anomaly rate of the synthetic
# dataset (74 anomalies / 3074 total events ~2.41%), approximated to
# 2.5% to account for boundary uncertainty.
# n_estimators=100 follows the original paper's recommended ensemble size.
# random_state=42 ensures full reproducibility.
#
# Note: The model is trained on the full dataset without withheld labels.
# This is intentional — Isolation Forest is unsupervised and does not use
# ground truth labels during training. Labels are used exclusively for
# post-hoc evaluation.
# ---------------------------------------------------------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.025,
    random_state=42
)

# Fit the model and generate binary predictions in a single call.
# sklearn convention: +1 = inlier (normal), -1 = outlier (anomalous).
df['predict'] = model.fit(features).predict(features)

# Remap sklearn's +1/-1 convention to binary labels consistent with the
# ground truth column (0 = normal, 1 = anomalous) for evaluation clarity.
df['predicted_binary'] = df['predict'].apply(lambda x: 1 if x == -1 else 0)

# Raw anomaly scores: mean path length across all trees, normalized.
# More negative values indicate shorter average isolation paths and higher
# anomaly likelihood. Scores are retained in raw form per sklearn convention.
df['anomaly_score'] = model.score_samples(features)

# ---------------------------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------------------------
# Ground truth labels are available because the dataset is synthetic —
# anomalies were injected with known identities during generation.
# This enables supervised evaluation metrics on an unsupervised model,
# which is standard practice in anomaly detection benchmarking.
# ---------------------------------------------------------------------------
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(df['is_anomaly'], df['predicted_binary']))

print("CONFUSION MATRIX")
print("Rows: Actual | Columns: Predicted")
print(confusion_matrix(df['is_anomaly'], df['predicted_binary']))

# ---------------------------------------------------------------------------
# 6. OUTPUT PERSISTENCE
# ---------------------------------------------------------------------------
# Save the enriched event log including engineered features, model
# predictions, and anomaly scores for downstream analysis and visualization.
# ---------------------------------------------------------------------------
df.to_csv('Outputs/anomaly_detection_results.csv', index=False)

# ---------------------------------------------------------------------------
# 7. VISUALIZATION — ANOMALY SCORE DISTRIBUTION
# ---------------------------------------------------------------------------
# Plot the distribution of raw anomaly scores stratified by ground truth
# label. Separation between the normal and anomalous distributions provides
# visual confirmation that the learned feature space meaningfully
# discriminates between the two classes. The overlap region corresponds
# directly to the false positive and false negative counts in the confusion
# matrix.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    df[df['is_anomaly'] == 0]['anomaly_score'],
    bins=40, alpha=0.6, color=COLOR_NORMAL, label='Normal (Ground Truth)'
)
ax.hist(
    df[df['is_anomaly'] == 1]['anomaly_score'],
    bins=40, alpha=0.6, color=COLOR_ANOMALY, label='Anomaly (Ground Truth)'
)

ax.set_xlabel(
    'Isolation Forest Anomaly Score (raw; more negative = more anomalous)',
    fontsize=11
)
ax.set_ylabel('Event Count', fontsize=11)
ax.set_title(
    'Anomaly Score Distribution by Ground Truth Label\n'
    'Isolation Forest — Behavioral Authentication Telemetry',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Outputs/score_distribution.png', dpi=150)
plt.close()
print("Saved: Outputs/score_distribution.png")

# ---------------------------------------------------------------------------
# 8. VISUALIZATION — VELOCITY VS. ANOMALY SCORE (PRIMARY SCATTER)
# ---------------------------------------------------------------------------
# Plots geospatial velocity (x-axis) against the raw Isolation Forest
# anomaly score (y-axis), colored by ground truth label.
#
# Expected pattern:
#   - Normal events cluster near velocity=0 with scores ~-0.40 to -0.55
#   - Behavioral anomalies (typing/CTR/timing) appear at low velocity
#     but with more negative scores, caught by non-geospatial signals
#   - Impossible travel anomalies extend far right on the velocity axis
#     with the most negative scores, reflecting their extreme feature
#     space isolation distance
#
# This plot demonstrates the dominant contribution of impossible travel
# to overall model sensitivity while confirming that behavioral signals
# provide complementary coverage for location-masked credential abuse.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

normal   = df[df['is_anomaly'] == 0]
anomalous = df[df['is_anomaly'] == 1]

ax.scatter(
    normal['velocity_kmh'], normal['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=18, label='Normal'
)
ax.scatter(
    anomalous['velocity_kmh'], anomalous['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_ANOMALY, s=45, label='Anomaly', zorder=5
)

ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=11)
ax.set_ylabel('Anomaly Score (raw; more negative = more anomalous)', fontsize=11)
ax.set_title(
    'Geospatial Velocity vs. Anomaly Score\n'
    'Impossible travel anomalies extend far right; behavioral anomalies visible at low velocity',
    fontsize=12
)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('Outputs/velocity_vs_score.png', dpi=150)
plt.close()
print("Saved: Outputs/velocity_vs_score.png")

# ---------------------------------------------------------------------------
# 9. VISUALIZATION — CONFUSION MATRIX HEATMAP
# ---------------------------------------------------------------------------
# Normalized confusion matrix (rates rather than counts) to ensure the
# color scale reflects per-class accuracy rather than class imbalance.
# The normal class (n=3000) would visually dominate a count-based matrix
# and obscure the anomaly detection rates of interest.
#
# Color interpretation:
#   Green (high value): correct classifications (true negative, true positive)
#   Red   (low value):  misclassifications (false positive, false negative)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    confusion_matrix(df['is_anomaly'], df['predicted_binary'], normalize='true'),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    xticklabels=['Normal', 'Anomaly'],
    yticklabels=['Normal', 'Anomaly'],
    ax=ax
)

ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('Actual Label', fontsize=11)
ax.set_title(
    'Confusion Matrix — Isolation Forest Anomaly Detection\n'
    '(Normalized by actual class)',
    fontsize=12
)

plt.tight_layout()
plt.savefig('Outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: Outputs/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 10. VISUALIZATION — PER-SIGNAL ANOMALY SCATTER PLOTS (2x2 GRID)
# ---------------------------------------------------------------------------
# Four scatter plots, one per injected anomaly type, each plotting the
# relevant detection signal (x-axis) against the Isolation Forest anomaly
# score (y-axis). Normal events are shown as the background point mass;
# anomalies of the specific type are overlaid in a distinct color.
#
# Purpose: isolate the contribution of each individual signal to the
# model's overall detection capability and demonstrate that the model
# responds to each anomaly type independently.
#
# Anomaly types and their primary diagnostic signals:
#   Impossible Travel  → velocity_kmh (geospatial)
#   Typing Speed       → typing_speed_wpm (behavioral biometric)
#   Click-Through Rate → click_through_rate (session behavioral)
#   Session Timing     → login_hour (temporal baseline)
# ---------------------------------------------------------------------------

# Convenience subsets for each anomaly type
normal_df  = df[df['anomaly_type'] == 'none']
travel_df  = df[df['anomaly_type'] == 'impossible_travel']
typing_df  = df[df['anomaly_type'] == 'typing_speed']
ctr_df     = df[df['anomaly_type'] == 'click_through']
timing_df  = df[df['anomaly_type'] == 'session_timing']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Per-Signal Anomaly Scatter Plots\n'
    'Each panel isolates one anomaly type against its primary detection signal',
    fontsize=14, y=1.02
)

# ── Panel A: Impossible Travel — velocity_kmh ────────────────────────────
ax = axes[0, 0]
ax.scatter(
    normal_df['velocity_kmh'], normal_df['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal'
)
ax.scatter(
    travel_df['velocity_kmh'], travel_df['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_TRAVEL, s=55, label='Impossible Travel', zorder=5
)
ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'A. Impossible Travel\n'
    'Physically impossible transit velocity (>1,200 km/h)\n'
    'Maps to MITRE ATT&CK T1078 — geolocation anomaly',
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel B: Typing Speed — typing_speed_wpm ─────────────────────────────
# Typing speed far below baseline indicates an attacker unfamiliar with the
# system or using manual, deliberate navigation. Speed far above baseline
# indicates scripted or automated access. Both extremes appear as outliers
# in the univariate distribution of this feature.
ax = axes[0, 1]
ax.scatter(
    normal_df['typing_speed_wpm'], normal_df['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal'
)
ax.scatter(
    typing_df['typing_speed_wpm'], typing_df['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_TYPING, s=55, label='Typing Speed Anomaly', zorder=5
)
ax.set_xlabel('Typing Speed (WPM)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'B. Typing Speed Deviation\n'
    'Far below baseline (unfamiliar attacker) or far above (scripted)\n'
    'Maps to MITRE ATT&CK T1078 — behavioral biometric anomaly',
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel C: Click-Through Rate — click_through_rate ─────────────────────
# Click-through rate near zero indicates bot-like navigation (no genuine
# user interaction with page elements). Rate near 1.0 indicates scripted
# enumeration of all available links, characteristic of credential-stuffing
# tools and automated reconnaissance. Both extremes deviate significantly
# from the normal distribution centered near 0.35.
ax = axes[1, 0]
ax.scatter(
    normal_df['click_through_rate'], normal_df['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal'
)
ax.scatter(
    ctr_df['click_through_rate'], ctr_df['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_CTR, s=55, label='CTR Anomaly', zorder=5
)
ax.set_xlabel('Click-Through Rate', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'C. Click-Through Rate Deviation\n'
    'Near 0.0 (bot-like) or near 1.0 (scripted enumeration)\n'
    'Maps to MITRE ATT&CK T1078 — session behavioral anomaly',
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel D: Session Timing — login_hour ─────────────────────────────────
# Login events between 1–4 AM fall outside the established baseline hour
# range of 7–10 AM for all synthetic users. Temporal outliers of this type
# are consistent with attacker activity occurring outside the victim's
# normal working hours, a well-documented indicator of account compromise
# in enterprise security telemetry.
ax = axes[1, 1]
ax.scatter(
    normal_df['login_hour'], normal_df['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal'
)
ax.scatter(
    timing_df['login_hour'], timing_df['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_TIMING, s=55, label='Session Timing Anomaly', zorder=5
)
ax.set_xlabel('Login Hour (0–23)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'D. Session Timing Anomaly\n'
    'Login at 1–4 AM, outside user baseline of 7–10 AM\n'
    'Maps to MITRE ATT&CK T1078 — temporal baseline deviation',
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('Outputs/per_signal_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Outputs/per_signal_scatter.png")

print("\nAll outputs written to Outputs/")
