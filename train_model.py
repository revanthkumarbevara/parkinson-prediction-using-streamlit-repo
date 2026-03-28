"""
train_model.py  —  Upgraded: XGBoost + SMOTE + Cross-Validation
Run once:  python train_model.py
"""

import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("parkinsons_dataset.csv")

FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

X = df[FEATURE_COLS].values
y = df["status"].values

print(f"Dataset  : {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes  : Healthy={sum(y==0)}, Parkinson's={sum(y==1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Scale
# ─────────────────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SMOTE — fix class imbalance on training data only
# ─────────────────────────────────────────────────────────────────────────────

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

print(f"\nAfter SMOTE:")
print(f"  Healthy     : {sum(y_train_bal==0)}")
print(f"  Parkinson's : {sum(y_train_bal==1)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. XGBoost model
# ─────────────────────────────────────────────────────────────────────────────

model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = 42,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring="accuracy")

print(f"\n5-Fold Cross-Validation Accuracy:")
print(f"  Scores : {[round(s, 4) for s in cv_scores]}")
print(f"  Mean   : {cv_scores.mean():.4f}")
print(f"  Std    : {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Train final model
# ─────────────────────────────────────────────────────────────────────────────

model.fit(X_train_bal, y_train_bal)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_s)

print(f"\nTest Set Results:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

with open("model.pkl",        "wb") as f: pickle.dump(model,        f)
with open("scaler.pkl",       "wb") as f: pickle.dump(scaler,       f)
with open("feature_cols.pkl", "wb") as f: pickle.dump(FEATURE_COLS, f)

print("\n✅  Saved: model.pkl | scaler.pkl | feature_cols.pkl")
print("🚀  Now run: streamlit run app.py")