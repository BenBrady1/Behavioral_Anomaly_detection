import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""
visualize.py
============
All visualizations for the behavioral anomaly detection pipeline.
Produces four output files in the Outputs/ directory:
    score_distribution.png  — anomaly score histogram by ground truth
    velocity_vs_score.png   — geospatial velocity vs anomaly score scatter
    confusion_matrix.png    — normalized Isolation Forest confusion matrix
    per_signal_scatter.png  — 2x3 grid, one panel per anomaly type
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Color palette — consistent across all figures
# ---------------------------------------------------------------------------
COLOR_NORMAL    = 'steelblue'
COLOR_ANOMALY   = 'crimson'
COLOR_TRAVEL    = '#E63946'
COLOR_TYPING    = '#F4A261'
COLOR_CTR       = '#2A9D8F'
COLOR_TIMING    = '#9B5DE5'
COLOR_RESOURCES = '#FF6B35'
COLOR_EXFIL     = '#B5179E'
ALPHA_NORMAL    = 0.35
ALPHA_ANOMALY   = 0.85


def score_distribution(df):
    """
    Histogram of raw Isolation Forest anomaly scores by ground truth label.
    Separation between distributions confirms the feature space discriminates
    between normal and anomalous events.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[df['is_anomaly'] == 0]['anomaly_score'],
            bins=40, alpha=0.6, color=COLOR_NORMAL, label='Normal (Ground Truth)')
    ax.hist(df[df['is_anomaly'] == 1]['anomaly_score'],
            bins=40, alpha=0.6, color=COLOR_ANOMALY, label='Anomaly (Ground Truth)')
    ax.set_xlabel('Isolation Forest Anomaly Score (raw; more negative = more anomalous)', fontsize=11)
    ax.set_ylabel('Event Count', fontsize=11)
    ax.set_title(
        'Anomaly Score Distribution by Ground Truth Label\n'
        'Isolation Forest — Behavioral Authentication Telemetry', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Outputs', 'score_distribution.png'), dpi=150)
    plt.close()
    print("Saved: Outputs/score_distribution.png")


def velocity_scatter(df):
    """
    Scatter plot of geospatial velocity vs anomaly score colored by ground
    truth label. Impossible travel anomalies extend far right; behavioral
    anomalies are visible at low velocity — demonstrating multi-signal coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    normal    = df[df['is_anomaly'] == 0]
    anomalous = df[df['is_anomaly'] == 1]
    ax.scatter(normal['velocity_kmh'], normal['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=18, label='Normal')
    ax.scatter(anomalous['velocity_kmh'], anomalous['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_ANOMALY, s=45, label='Anomaly', zorder=5)
    ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=11)
    ax.set_ylabel('Anomaly Score (raw; more negative = more anomalous)', fontsize=11)
    ax.set_title(
        'Geospatial Velocity vs. Anomaly Score\n'
        'Impossible travel extends far right; behavioral anomalies visible at low velocity',
        fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Outputs', 'velocity_vs_score.png'), dpi=150)
    plt.close()
    print("Saved: Outputs/velocity_vs_score.png")


def confusion_heatmap(df):
    """
    Normalized confusion matrix for Isolation Forest predictions.
    Normalized by actual class so color reflects per-class accuracy
    rather than raw count imbalance.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        confusion_matrix(df['is_anomaly'], df['if_predicted_binary'], normalize='true'),
        annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'], ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_title('Confusion Matrix — Isolation Forest\n(Normalized by actual class)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Outputs', 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("Saved: Outputs/confusion_matrix.png")


def per_signal_scatter(df):
    """
    2x3 grid — one panel per anomaly type, each showing its primary
    detection signal vs anomaly score. Normal events form the background;
    the specific anomaly type is overlaid in a distinct color.

    Layout:
        [0,0] A. Impossible Travel    → velocity_kmh
        [0,1] B. Typing Speed         → typing_speed_wpm
        [0,2] C. Click-Through Rate   → click_through_rate
        [1,0] D. Session Timing       → login_hour
        [1,1] E. Privilege Escalation → resources_accessed
        [1,2] F. Data Exfiltration    → data_volume_mb
    """
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
        fontsize=14, y=1.02)

    # Panel A [0,0] — Impossible Travel
    ax = axes[0, 0]
    ax.scatter(normal_df['velocity_kmh'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(travel_df['velocity_kmh'], travel_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_TRAVEL, s=55, label='Impossible Travel', zorder=5)
    ax.set_xlabel('Geospatial Velocity (km/h)', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('A. Impossible Travel\nPhysically impossible transit velocity (>1,200 km/h)\nMaps to MITRE ATT&CK T1078 — geolocation anomaly', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel B [0,1] — Typing Speed
    ax = axes[0, 1]
    ax.scatter(normal_df['typing_speed_wpm'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(typing_df['typing_speed_wpm'], typing_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_TYPING, s=55, label='Typing Speed Anomaly', zorder=5)
    ax.set_xlabel('Typing Speed (WPM)', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('B. Typing Speed Deviation\nFar below baseline (unfamiliar attacker) or far above (scripted)\nMaps to MITRE ATT&CK T1078 — behavioral biometric anomaly', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel C [0,2] — Click-Through Rate
    ax = axes[0, 2]
    ax.scatter(normal_df['click_through_rate'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(ctr_df['click_through_rate'], ctr_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_CTR, s=55, label='CTR Anomaly', zorder=5)
    ax.set_xlabel('Click-Through Rate', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('C. Click-Through Rate Deviation\nNear 0.0 (bot-like) or near 1.0 (scripted enumeration)\nMaps to MITRE ATT&CK T1078 — session behavioral anomaly', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel D [1,0] — Session Timing
    ax = axes[1, 0]
    ax.scatter(normal_df['login_hour'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(timing_df['login_hour'], timing_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_TIMING, s=55, label='Session Timing Anomaly', zorder=5)
    ax.set_xlabel('Login Hour (0-23)', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('D. Session Timing Anomaly\nLogin at 1-4 AM, outside user baseline of 7-10 AM\nMaps to MITRE ATT&CK T1078 — temporal baseline deviation', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel E [1,1] — Privilege Escalation
    ax = axes[1, 1]
    ax.scatter(normal_df['resources_accessed'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(priv_df['resources_accessed'], priv_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_RESOURCES, s=55, label='Privilege Escalation', zorder=5)
    ax.set_xlabel('Resources Accessed (count)', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('E. Privilege Escalation\nAbnormal spike in unique resources accessed per session\nMaps to MITRE ATT&CK T1078 — lateral movement indicator', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel F [1,2] — Data Exfiltration
    ax = axes[1, 2]
    ax.scatter(normal_df['data_volume_mb'], normal_df['anomaly_score'],
               alpha=ALPHA_NORMAL, color=COLOR_NORMAL, s=12, label='Normal')
    ax.scatter(exfil_df['data_volume_mb'], exfil_df['anomaly_score'],
               alpha=ALPHA_ANOMALY, color=COLOR_EXFIL, s=55, label='Data Exfiltration', zorder=5)
    ax.set_xlabel('Data Volume (MB)', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title('F. Data Exfiltration\nAbnormally large transfer volume (500-2,000 MB)\nMaps to MITRE ATT&CK T1041 — Exfiltration Over C2 Channel', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Outputs', 'per_signal_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: Outputs/per_signal_scatter.png")
