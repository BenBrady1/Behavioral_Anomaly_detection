"""
Behavioral Anomaly Detection via Isolation Forest and One-Class SVM
====================================================================
Implements an unsupervised machine learning pipeline for detecting
identity-based threats in authentication event telemetry. Detection
signals are derived from behavioral biometrics, geospatial velocity
analysis, resource access patterns, and data transfer volume — mapped
to MITRE ATT&CK T1078 (Valid Accounts) and T1041 (Exfiltration Over
Command and Control Channel).

Methodology:
    Feature engineering extracts eight behavioral signals from raw login
    events. Two unsupervised models are trained and compared: Isolation
    Forest (Liu et al., 2008) and One-Class SVM (Scholkopf et al., 1999).
    Ground truth labels, available due to the synthetic nature of the
    dataset, enable rigorous precision/recall evaluation of both models.

References:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest.
        IEEE International Conference on Data Mining, 413-422.
    Scholkopf, B., et al. (1999). Support Vector Method for Novelty
        Detection. Advances in Neural Information Processing Systems.
    MITRE ATT&CK. (2024). Valid Accounts (T1078).
        https://attack.mitre.org/techniques/T1078/
    MITRE ATT&CK. (2024). Exfiltration Over C2 Channel (T1041).
        https://attack.mitre.org/techniques/T1041/
    NIST Special Publication 800-207. (2020). Zero Trust Architecture.
        https://csrc.nist.gov/pubs/sp/800/207/final
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------------
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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
COLOR_NORMAL    = 'steelblue'  # All normal (non-anomalous) observations
COLOR_ANOMALY   = 'crimson'    # All anomalous observations (any type)
COLOR_TRAVEL    = '#E63946'    # Impossible travel anomalies
COLOR_TYPING    = '#F4A261'    # Typing speed anomalies
COLOR_CTR       = '#2A9D8F'    # Click-through rate anomalies
COLOR_TIMING    = '#9B5DE5'    # Session timing anomalies
COLOR_RESOURCES = '#FF6B35'    # Privilege escalation anomalies
COLOR_EXFIL     = '#B5179E'    # Data exfiltration anomalies
ALPHA_NORMAL    = 0.35         # Transparency for normal point mass
ALPHA_ANOMALY   = 0.85         # Transparency for anomaly highlights

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
# Retain only numerical features required by the models. Categorical
# identifiers (user_id, city, anomaly_type) and metadata columns are
# excluded. NaN values arising from first-login records are imputed with
# zero, representing the absence of a signal rather than a measurement.
# ---------------------------------------------------------------------------
features = df[[
    'typing_speed_wpm',
    'click_through_rate',
    'login_hour',
    'hours_since_last_login',
    'distance_from_last_login',
    'velocity_kmh',
    'resources_accessed',
    'data_volume_mb'
]]
features = features.fillna(0)

# ---------------------------------------------------------------------------
# 4. MODEL 1 — ISOLATION FOREST
# ---------------------------------------------------------------------------
# The Isolation Forest algorithm (Liu et al., 2008) detects anomalies by
# recursively partitioning the feature space using random splits. Anomalous
# observations, being few and distinct, are isolated in fewer splits than
# normal observations, yielding shorter average path lengths across the
# ensemble of trees.
#
# contamination=0.033 reflects the known anomaly rate of the synthetic
# dataset (100 anomalies / 3100 total events = 3.23%), approximated to
# 3.3% to account for boundary uncertainty.
# n_estimators=100 follows the original paper's recommended ensemble size.
# random_state=42 ensures full reproducibility.
#
# The model is trained on the full dataset without withheld labels.
# This is intentional — Isolation Forest is unsupervised and does not use
# ground truth labels during training. Labels are used exclusively for
# post-hoc evaluation.
# ---------------------------------------------------------------------------
if_model = IsolationForest(
    n_estimators=100,
    contamination=0.033,
    random_state=42
)

# sklearn convention: +1 = inlier (normal), -1 = outlier (anomalous).
df['if_predict'] = if_model.fit(features).predict(features)

# Remap to binary labels consistent with ground truth (0=normal, 1=anomaly).
df['if_predicted_binary'] = df['if_predict'].apply(lambda x: 1 if x == -1 else 0)

# Raw anomaly scores: more negative = shorter average isolation path =
# higher anomaly likelihood. Retained in raw form per sklearn convention.
df['anomaly_score'] = if_model.score_samples(features)

# ---------------------------------------------------------------------------
# 5. MODEL 2 — ONE-CLASS SVM
# ---------------------------------------------------------------------------
# One-Class SVM (Scholkopf et al., 1999) learns a decision boundary around
# normal data in a kernel-transformed feature space via the RBF kernel.
# Points falling outside the boundary are classified as anomalous.
#
# Compared to Isolation Forest:
#   - More sensitive to the shape of the normal distribution
#   - Higher precision on clean, well-separated data
#   - Less scalable to high-dimensional or noisy production telemetry
#
# nu=0.033 is the upper bound on the fraction of outliers and approximates
# the known anomaly rate, analogous to the contamination parameter above.
# ---------------------------------------------------------------------------
svm_model = OneClassSVM(kernel='rbf', nu=0.033)
svm_model.fit(features)

# sklearn convention: +1 = inlier (normal), -1 = outlier (anomalous).
df['svm_predict'] = svm_model.predict(features)
df['svm_predicted_binary'] = df['svm_predict'].apply(lambda x: 1 if x == -1 else 0)

# ---------------------------------------------------------------------------
# 6. MODEL EVALUATION — COMPARATIVE RESULTS
# ---------------------------------------------------------------------------
# Ground truth labels are available because anomalies were injected with
# known identities during synthetic data generation. This enables supervised
# evaluation metrics on both unsupervised models — standard practice in
# anomaly detection benchmarking (Chandola et al., 2009).
# ---------------------------------------------------------------------------
print("=" * 60)
print("ISOLATION FOREST — CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(df['is_anomaly'], df['if_predicted_binary']))
print("CONFUSION MATRIX (Rows: Actual | Columns: Predicted)")
print(confusion_matrix(df['is_anomaly'], df['if_predicted_binary']))

print("\n" + "=" * 60)
print("ONE-CLASS SVM — CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(df['is_anomaly'], df['svm_predicted_binary']))
print("CONFUSION MATRIX (Rows: Actual | Columns: Predicted)")
print(confusion_matrix(df['is_anomaly'], df['svm_predicted_binary']))

# ---------------------------------------------------------------------------
# 7. OUTPUT PERSISTENCE
# ---------------------------------------------------------------------------
# Save the enriched event log including engineered features, both models'
# predictions, and Isolation Forest anomaly scores for downstream analysis.
# ---------------------------------------------------------------------------
df.to_csv('Outputs/anomaly_detection_results.csv', index=False)

# ---------------------------------------------------------------------------
# 8. VISUALIZATION — ANOMALY SCORE DISTRIBUTION
# ---------------------------------------------------------------------------
# Distribution of raw Isolation Forest anomaly scores stratified by ground
# truth label. Separation between distributions confirms the learned feature
# space meaningfully discriminates between normal and anomalous events.
# The overlap region corresponds to false positives and false negatives.
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
# 9. VISUALIZATION — VELOCITY VS. ANOMALY SCORE
# ---------------------------------------------------------------------------
# Geospatial velocity (x-axis) vs. Isolation Forest anomaly score (y-axis).
# Normal events cluster near velocity=0. Behavioral anomalies (typing, CTR,
# timing, privilege escalation, exfiltration) appear at low velocity with
# more negative scores — caught by non-geospatial signals. Impossible travel
# anomalies extend far right, confirming geospatial velocity as the dominant
# single detection signal.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
normal_all   = df[df['is_anomaly'] == 0]
anomalous_all = df[df['is_anomaly'] == 1]
ax.scatter(
    normal_all['velocity_kmh'], normal_all['anomaly_score'],
    alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=18, label='Normal'
)
ax.scatter(
    anomalous_all['velocity_kmh'], anomalous_all['anomaly_score'],
    alpha=ALPHA_ANOMALY, color=COLOR_ANOMALY, s=45, label='Anomaly', zorder=5
)
ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=11)
ax.set_ylabel('Anomaly Score (raw; more negative = more anomalous)', fontsize=11)
ax.set_title(
    'Geospatial Velocity vs. Anomaly Score\n'
    'Impossible travel extends far right; behavioral anomalies visible at low velocity',
    fontsize=12
)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('Outputs/velocity_vs_score.png', dpi=150)
plt.close()
print("Saved: Outputs/velocity_vs_score.png")

# ---------------------------------------------------------------------------
# 10. VISUALIZATION — ISOLATION FOREST CONFUSION MATRIX
# ---------------------------------------------------------------------------
# Normalized by actual class so color scale reflects per-class accuracy
# rather than class imbalance. The normal class (n=3000) would otherwise
# visually dominate a count-based matrix.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(df['is_anomaly'], df['if_predicted_binary'], normalize='true'),
    annot=True, fmt='.2f', cmap='RdYlGn',
    xticklabels=['Normal', 'Anomaly'],
    yticklabels=['Normal', 'Anomaly'], ax=ax
)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('Actual Label', fontsize=11)
ax.set_title(
    'Confusion Matrix — Isolation Forest\n(Normalized by actual class)',
    fontsize=12
)
plt.tight_layout()
plt.savefig('Outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: Outputs/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 11. VISUALIZATION — PER-SIGNAL ANOMALY SCATTER PLOTS (2x3 GRID)
# ---------------------------------------------------------------------------
# Six scatter plots, one per anomaly type, each plotting the primary
# detection signal (x-axis) against the Isolation Forest anomaly score
# (y-axis). Normal events form the background point mass; the specific
# anomaly type is overlaid in a distinct color.
#
# Layout:
#   [0,0] A. Impossible Travel    → velocity_kmh
#   [0,1] B. Typing Speed         → typing_speed_wpm
#   [0,2] C. Click-Through Rate   → click_through_rate
#   [1,0] D. Session Timing       → login_hour
#   [1,1] E. Privilege Escalation → resources_accessed
#   [1,2] F. Data Exfiltration    → data_volume_mb
# ---------------------------------------------------------------------------

# Subsets per anomaly type
normal_df = df[df['anomaly_type'] == 'none']
travel_df = df[df['anomaly_type'] == 'impossible_travel']
typing_df = df[df['anomaly_type'] == 'typing_speed']
ctr_df    = df[df['anomaly_type'] == 'click_through']
timing_df = df[df['anomaly_type'] == 'session_timing']
priv_df   = df[df['anomaly_type'] == 'privilege_escalation']
exfil_df  = df[df['anomaly_type'] == 'data_exfiltration']

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle(
    'Per-Signal Anomaly Scatter Plots\n'
    'Each panel isolates one anomaly type against its primary detection signal',
    fontsize=14, y=1.02
)

# ── Panel A [0,0]: Impossible Travel — velocity_kmh ──────────────────────
# Geospatial velocity exceeding commercial aviation speed (~1,200 km/h)
# indicates a physically impossible location transition between consecutive
# logins. This is the strongest single signal in the dataset, producing
# the most negative anomaly scores and clearest separation from normal.
ax = axes[0, 0]
ax.scatter(normal_df['velocity_kmh'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(travel_df['velocity_kmh'], travel_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_TRAVEL, s=55,
           label='Impossible Travel', zorder=5)
ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'A. Impossible Travel\n'
    'Physically impossible transit velocity (>1,200 km/h)\n'
    'Maps to MITRE ATT&CK T1078 — geolocation anomaly', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel B [0,1]: Typing Speed — typing_speed_wpm ───────────────────────
# Typing speed far below baseline indicates an attacker unfamiliar with the
# system. Speed far above baseline indicates scripted or automated access.
# Both extremes produce outliers relative to the per-population distribution
# centered near 65 WPM, and are caught by the model as behavioral deviations
# independent of geographic location.
ax = axes[0, 1]
ax.scatter(normal_df['typing_speed_wpm'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(typing_df['typing_speed_wpm'], typing_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_TYPING, s=55,
           label='Typing Speed Anomaly', zorder=5)
ax.set_xlabel('Typing Speed (WPM)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'B. Typing Speed Deviation\n'
    'Far below baseline (unfamiliar attacker) or far above (scripted)\n'
    'Maps to MITRE ATT&CK T1078 — behavioral biometric anomaly', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel C [0,2]: Click-Through Rate — click_through_rate ───────────────
# Click-through rate near zero indicates bot-like navigation with no genuine
# user interaction. Rate near 1.0 indicates scripted enumeration of all
# available links, characteristic of credential-stuffing tools and automated
# reconnaissance. Both extremes deviate significantly from the normal
# distribution centered near 0.35.
ax = axes[0, 2]
ax.scatter(normal_df['click_through_rate'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(ctr_df['click_through_rate'], ctr_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_CTR, s=55,
           label='CTR Anomaly', zorder=5)
ax.set_xlabel('Click-Through Rate', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'C. Click-Through Rate Deviation\n'
    'Near 0.0 (bot-like) or near 1.0 (scripted enumeration)\n'
    'Maps to MITRE ATT&CK T1078 — session behavioral anomaly', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel D [1,0]: Session Timing — login_hour ───────────────────────────
# Login events between 1-4 AM fall outside the established baseline hour
# range of 7-10 AM for all synthetic users. Temporal outliers of this type
# are consistent with attacker activity occurring outside the victim's
# normal working hours — a well-documented indicator of account compromise
# in enterprise security telemetry.
ax = axes[1, 0]
ax.scatter(normal_df['login_hour'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(timing_df['login_hour'], timing_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_TIMING, s=55,
           label='Session Timing Anomaly', zorder=5)
ax.set_xlabel('Login Hour (0-23)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'D. Session Timing Anomaly\n'
    'Login at 1-4 AM, outside user baseline of 7-10 AM\n'
    'Maps to MITRE ATT&CK T1078 — temporal baseline deviation', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel E [1,1]: Privilege Escalation — resources_accessed ─────────────
# A sudden spike in unique resources accessed per session indicates lateral
# movement. Legitimate users access a consistent, role-bounded set of
# resources (2-5 per session). An attacker probing resources outside their
# granted scope produces an abnormally high count (20-50 per session).
# The account credentials are valid — the access pattern is the signal.
# Maps to MITRE ATT&CK T1078 as a post-authentication lateral movement
# indicator.
ax = axes[1, 1]
ax.scatter(normal_df['resources_accessed'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(priv_df['resources_accessed'], priv_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_RESOURCES, s=55,
           label='Privilege Escalation', zorder=5)
ax.set_xlabel('Resources Accessed (count)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'E. Privilege Escalation\n'
    'Abnormal spike in unique resources accessed per session\n'
    'Maps to MITRE ATT&CK T1078 — lateral movement indicator', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# ── Panel F [1,2]: Data Exfiltration — data_volume_mb ────────────────────
# Normal sessions transfer 0.5-10 MB. Exfiltration events transfer
# 500-2,000 MB — one to three orders of magnitude above baseline —
# consistent with bulk data staging or direct transfer to an external
# destination. Maps to MITRE ATT&CK T1041 (Exfiltration Over C2 Channel),
# which is distinct from T1078. These two techniques frequently appear
# together in kill chain analysis: T1078 enables access, T1041 completes
# the objective.
ax = axes[1, 2]
ax.scatter(normal_df['data_volume_mb'], normal_df['anomaly_score'],
           alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
ax.scatter(exfil_df['data_volume_mb'], exfil_df['anomaly_score'],
           alpha=ALPHA_ANOMALY, color=COLOR_EXFIL, s=55,
           label='Data Exfiltration', zorder=5)
ax.set_xlabel('Data Volume (MB)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title(
    'F. Data Exfiltration\n'
    'Abnormally large transfer volume (500-2,000 MB)\n'
    'Maps to MITRE ATT&CK T1041 — Exfiltration Over C2 Channel', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('Outputs/per_signal_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Outputs/per_signal_scatter.png")

print("\nAll outputs written to Outputs/")
