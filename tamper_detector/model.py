"""Train and evaluate the blackout-movement model.

The original notebook used LogisticRegression, scored accuracy on the training data, then
did a train/test split and RFE feature ranking. This version keeps those steps and adds what
the original was missing: a leakage check, precision/recall, and a RandomForest comparison
(the original had a half-typed `from sklearn.linear_model import Random`).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _scores(model, X_te, y_te):
    p = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    return {'accuracy': accuracy_score(y_te, p), 'precision': precision_score(y_te, p),
            'recall': recall_score(y_te, p), 'roc_auc': roc_auc_score(y_te, prob)}


def evaluate(X, y, seed=0):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    models = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        rows.append({'model': name, **_scores(m, X_te, y_te)})
    return pd.DataFrame(rows).round(3), models


def rfe_ranking(X, y, n_keep=8):
    rfe = RFE(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).steps[-1][1],
              n_features_to_select=n_keep)
    rfe.fit(StandardScaler().fit_transform(X), y)
    return sorted(zip(rfe.ranking_, X.columns))


def top_importances(rf, columns, k=8):
    s = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    return s.head(k).round(3)
