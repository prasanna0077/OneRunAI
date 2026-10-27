"""Shared ML helpers: pipelines, train/evaluate loop, feature importance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(estimator, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


@dataclass
class TrainedSuite:
    """Everything a module page needs after training a family of models."""
    models: dict                       # name -> fitted Pipeline
    metrics: pd.DataFrame              # one row per model
    roc: dict                          # name -> (fpr, tpr, auc)
    importance: pd.DataFrame           # feature x model (each column sums to 1)
    feature_names: list
    best_model: str
    confusion: dict = field(default_factory=dict)    # name -> 2x2 array
    n_train: int = 0
    n_test: int = 0
    base_rate: float = 0.0

    def predict_proba(self, name: str, X: pd.DataFrame) -> np.ndarray:
        return self.models[name].predict_proba(X[self.feature_names])[:, 1]


def _importance(pipe: Pipeline, n: int) -> np.ndarray:
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_, dtype=float)
    else:
        v = np.abs(np.ravel(est.coef_))
    return v / v.sum() if v.sum() else np.zeros(n)


def fit_suite(X: pd.DataFrame, y: pd.Series, estimators: dict, scale_for: tuple = (),
              seed: int = 42, threshold: float = 0.5, test_size: float = 0.25) -> TrainedSuite:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    models, rows, roc, imp, cm = {}, [], {}, {}, {}
    for name, est in estimators.items():
        pipe = make_pipeline(est, scale=name in scale_for).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= threshold).astype(int)
        auc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        models[name], roc[name], imp[name] = pipe, (fpr, tpr, auc), _importance(pipe, X.shape[1])
        cm[name] = confusion_matrix(yte, pred)
        rows.append(dict(Model=name, AUC=auc, Accuracy=accuracy_score(yte, pred),
                         Precision=precision_score(yte, pred, zero_division=0),
                         Recall=recall_score(yte, pred, zero_division=0), F1=f1_score(yte, pred, zero_division=0)))
    metrics = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    importance = pd.DataFrame(imp, index=X.columns)
    return TrainedSuite(models=models, metrics=metrics, roc=roc, importance=importance,
                        feature_names=list(X.columns), best_model=metrics.loc[0, "Model"], confusion=cm,
                        n_train=len(Xtr), n_test=len(Xte), base_rate=float(y.mean()))
