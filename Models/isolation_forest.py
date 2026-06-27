"""
isolation_forest.py
===================
Trains and applies the Isolation Forest anomaly detection model.

Reference: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest.
    IEEE International Conference on Data Mining, 413-422.
"""

from sklearn.ensemble import IsolationForest


def run(df, features, contamination=0.033, n_estimators=100, random_state=42):
    """
    Fit Isolation Forest and append predictions and anomaly scores to df.

    Parameters
    ----------
    contamination : float
        Expected anomaly rate. Set to match known dataset rate (3.23%).
    n_estimators : int
        Number of trees in the ensemble per Liu et al. (2008) recommendation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Input dataframe with three new columns:
            if_predict          — raw sklearn output (+1 normal, -1 anomaly)
            if_predicted_binary — remapped to (0 normal, 1 anomaly)
            anomaly_score       — raw score_samples output (more negative = more anomalous)
    model : IsolationForest
        Fitted model instance.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )

    df['if_predict']          = model.fit(features).predict(features)
    df['if_predicted_binary'] = df['if_predict'].apply(lambda x: 1 if x == -1 else 0)
    df['anomaly_score']       = model.score_samples(features)

    return df, model
