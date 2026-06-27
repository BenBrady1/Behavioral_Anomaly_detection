"""
evaluate.py
===========
Model evaluation — classification reports and confusion matrix outputs
for both Isolation Forest and One-Class SVM predictions.
"""

from sklearn.metrics import classification_report, confusion_matrix


def report(df):
    """
    Print classification reports and confusion matrices for both models.
    Ground truth labels are available because anomalies were injected with
    known identities during synthetic data generation — standard practice
    in anomaly detection benchmarking (Chandola et al., 2009).
    """
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

def composite_risk_score(df):
    df['anomaly_score_normalized'] = (df['anomaly_score'] - df['anomaly_score'].min()) / (df['anomaly_score'].max() - df['anomaly_score'].min())
    df['anomaly_score_normalized'] = 1 - df['anomaly_score_normalized']
    df['composite_risk_score'] = (
        df['anomaly_score_normalized'] * 0.5 +
        df['if_predicted_binary']      * 0.25 +
        df['svm_predicted_binary']     * 0.25
    )
    return df

def attck_taxonomy(df):
    """
    Map each event's anomaly type to its formal MITRE ATT&CK technique ID.
    T1078 — Valid Accounts: covers credential-based anomalies.
    T1041 — Exfiltration Over C2 Channel: covers data transfer anomalies.
    """
    taxonomy = {
        'none':                 'N/A',
        'impossible_travel':    'T1078',
        'typing_speed':         'T1078',
        'click_through':        'T1078',
        'session_timing':       'T1078',
        'privilege_escalation': 'T1078',
        'data_exfiltration':    'T1041',
    }
    df['attck_technique'] = df['anomaly_type'].map(taxonomy)
    return df