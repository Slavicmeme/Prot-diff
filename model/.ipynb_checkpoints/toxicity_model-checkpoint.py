import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class ToxicityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def filter_non_toxic(self, X):
        preds = self.predict(X)
        return [vec for vec, label in zip(X, preds) if label == 0]  # 0 = non-toxic
