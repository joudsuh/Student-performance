
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def get_feature_importance(X, y):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return np.argsort(model.feature_importances_)[::-1][:10]
