from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os
import numpy as np

models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

X_dummy = np.random.rand(100, 6)
y_dummy = np.random.randint(0, 2, 100)

print("Creating demo models...")

lr = LogisticRegression(random_state=42)
lr.fit(X_dummy, y_dummy)
joblib.dump(lr, os.path.join(models_dir, 'logistic_regression_model.pkl'))
print("Logistic Regression created")

rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_dummy, y_dummy)
joblib.dump(rf, os.path.join(models_dir, 'random_forest_model.pkl'))
print(" Random Forest created")

xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
xgb_model.fit(X_dummy, y_dummy)
joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
print(" XGBoost created")

print("\nAll demo models created!")