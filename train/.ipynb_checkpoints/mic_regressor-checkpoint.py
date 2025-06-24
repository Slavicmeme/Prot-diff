from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

class MICRegressor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def filter_by_threshold(self, X, threshold=1.2):
        preds = self.predict(X)
        return [vec for vec, mic in zip(X, preds) if mic < threshold]
