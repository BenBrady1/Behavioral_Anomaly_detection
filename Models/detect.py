"""
detect.py
=========
Orchestrator for the behavioral anomaly detection pipeline.
Run from the project root: python Models/detect.py

Pipeline order:
    1. features.py         — load data, engineer signals
    2. isolation_forest.py — fit Isolation Forest, score events
    3. one_class_svm.py    — fit One-Class SVM, score events
    4. evaluate.py         — print classification reports
    5. Output CSV saved
    6. visualize.py        — generate all plots
"""

import sys
import os

# Allow imports from Models/ and Data/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))

import features
import isolation_forest
import one_class_svm
import evaluate
import visualize

# 1. Load data and engineer features
df, X = features.load_and_engineer(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'login_events.csv')
)

# 2. Isolation Forest
df, if_model = isolation_forest.run(df, X)

# 3. One-Class SVM
df, svm_model = one_class_svm.run(df, X)

# 4. Evaluate both models
evaluate.report(df)
df = evaluate.composite_risk_score(df)
print(df['composite_risk_score'].describe())

df = evaluate.attck_taxonomy(df)
print(df[['anomaly_type', 'attck_technique']].value_counts())


# 5. Save enriched output
out_path = os.path.join(os.path.dirname(__file__), '..', 'Outputs', 'anomaly_detection_results.csv')
df.to_csv(out_path, index=False)
print("Saved: Outputs/anomaly_detection_results.csv")

# 6. Visualizations
visualize.score_distribution(df)
visualize.confusion_heatmap(df)
visualize.per_signal_scatter(df)
visualize.risk_score_plot(df)

print("\nPipeline complete.")
