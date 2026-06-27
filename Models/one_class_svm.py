"""
one_class_svm.py
================
Trains and applies the One-Class SVM anomaly detection model.

Reference: Scholkopf, B., et al. (1999). Support Vector Method for Novelty
    Detection. Advances in Neural Information Processing Systems.
"""

from sklearn.svm import OneClassSVM


def run(df, features, nu=0.033, kernel='rbf'):
    """
    Fit One-Class SVM and append predictions to df.

    Parameters
    ----------
    nu : float
        Upper bound on the fraction of outliers. Analogous to contamination
        in Isolation Forest. Set to match known dataset anomaly rate (3.23%).
    kernel : str
        Kernel function. RBF allows non-linear decision boundaries in the
        original feature space via the kernel trick.

    Returns
    -------
    df : pd.DataFrame
        Input dataframe with two new columns:
            svm_predict          — raw sklearn output (+1 normal, -1 anomaly)
            svm_predicted_binary — remapped to (0 normal, 1 anomaly)
    model : OneClassSVM
        Fitted model instance.
    """
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(features)

    df['svm_predict']          = model.predict(features)
    df['svm_predicted_binary'] = df['svm_predict'].apply(lambda x: 1 if x == -1 else 0)

    return df, model
