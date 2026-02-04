#!/usr/bin/env python3
# model_training.py — robust trainer (version-safe), saves main + quantiles
import os, sys, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=== AgriAI: training start ===")
print(f"[info] Python: {sys.version.split()[0]}  pandas: {pd.__version__}")

# Load data
try:
    from utils import load_and_merge_data
    df = load_and_merge_data()
except Exception as e:
    print("[error] data load failed:", e)
    sys.exit(1)

needed = ['Item','average_rain_fall_mm_per_year','avg_temp','pesticides_tonnes','hg/ha_yield']
missing = [c for c in needed if c not in df.columns]
if missing:
    print("[error] Missing columns:", missing)
    sys.exit(1)

num_cols = ['average_rain_fall_mm_per_year','avg_temp','pesticides_tonnes','hg/ha_yield']
for c in num_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
before, df = len(df), df.dropna(subset=needed).copy()
print(f"[info] rows before/after cleaning: {before}/{len(df)}")
if len(df) < 30: print("[warn] Very few rows (<30); quality may be low.")

X = df[['Item','average_rain_fall_mm_per_year','avg_temp','pesticides_tonnes']]
y = df['hg/ha_yield']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=min(0.2, 0.3 if len(df)<50 else 0.2), random_state=42
)

# Version-safe OneHotEncoder
from sklearn import __version__ as sklver
def ver_tuple(v):
    try: a,b,*_ = v.split("."); return int(a),int(b)
    except: return (0,0)
use_sparse_output = ver_tuple(sklver) >= (1,2)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

numeric_features = ['average_rain_fall_mm_per_year','avg_temp','pesticides_tonnes']
categorical_features = ['Item']

numeric_transformer = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if use_sparse_output else OneHotEncoder(handle_unknown="ignore", sparse=False)
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                  ('cat', ohe, categorical_features)])

rf = Pipeline([('preprocessor', preprocessor),
               ('model', RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))])
gbr = Pipeline([('preprocessor', preprocessor),
                ('model', GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=3))])

def eval_print(name, y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat); r2 = r2_score(y_true, y_hat)
    print(f"[metric] {name:10s} MAE: {mae:8.2f} | R2: {r2:6.3f}")
    return mae, r2

ok_main = True
try:
    print("[info] Fitting RandomForest...")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    eval_print("RandomForest", y_test, rf_pred)
except Exception as e:
    print("[error] RF failed:", e)

try:
    print("[info] Fitting GradientBoosting...")
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    eval_print("GradBoost", y_test, gbr_pred)
except Exception as e:
    print("[error] GBR failed:", e); ok_main = False

model_to_save = gbr if ok_main else rf

# Quantile models (P10/P90)
q_models = None
try:
    q10 = Pipeline([('preprocessor', preprocessor),
                    ('model', GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=400,
                                                        learning_rate=0.05, max_depth=3, random_state=42))])
    q90 = Pipeline([('preprocessor', preprocessor),
                    ('model', GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=400,
                                                        learning_rate=0.05, max_depth=3, random_state=42))])
    print("[info] Fitting quantile models...")
    q10.fit(X_train, y_train); q90.fit(X_train, y_train)
    q_models = {"p10": q10, "p90": q90}
except Exception as e:
    print("[warn] Quantile models failed:", e)

os.makedirs("models", exist_ok=True)
joblib.dump(model_to_save, "models/crop_yield_model.joblib"); print("[save] models/crop_yield_model.joblib")
if q_models: joblib.dump(q_models, "models/quantiles.joblib"); print("[save] models/quantiles.joblib")
print("=== AgriAI: training done ===")
