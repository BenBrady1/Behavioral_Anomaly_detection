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
