# -*- coding: utf-8 -*-
"""
Isolation Forest baseline.

This is a thin wrapper around sklearn.ensemble.IsolationForest.
Input:
    X: numpy.ndarray [N, W, F]
Score:
    Larger means more anomalous.
"""

from __future__ import absolute_import, print_function

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestBaseline(object):
    def __init__(self, feature_mode="flatten", n_estimators=300,
                 contamination="auto", random_state=42, n_jobs=-1,
                 standardize=True):
        self.feature_mode = feature_mode
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _features(self, X):
        if X.ndim != 3:
            raise ValueError("X should be [N,W,F], got {}".format(X.shape))

        if self.feature_mode == "flatten":
            feat = X.reshape(X.shape[0], -1)
        elif self.feature_mode == "mean_std":
            feat = np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
        else:
            raise ValueError("Unknown feature_mode={}".format(self.feature_mode))
        return feat.astype(np.float32)

    def fit(self, X_train_normal):
        feat = self._features(X_train_normal)
        if self.scaler is not None:
            feat = self.scaler.fit_transform(feat)
        self.model.fit(feat)
        return self

    def score(self, X):
        feat = self._features(X)
        if self.scaler is not None:
            feat = self.scaler.transform(feat)
        return -self.model.decision_function(feat)

    def predict(self, X, threshold):
        return (self.score(X) >= threshold).astype(np.int64)


if __name__ == "__main__":
    X = np.random.randn(128, 100, 68).astype(np.float32)
    model = IsolationForestBaseline(feature_mode="mean_std").fit(X[:80])
    print(model.score(X[80:]).shape)
