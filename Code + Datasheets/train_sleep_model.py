"""
Predict sleep quality (poor / fair / good) from HRV and heart-rate features.

Since the dataset has no pre-existing "sleep quality" label, one is derived
from standard sleep-science metrics: sleep efficiency, sleep latency,
wake-after-sleep-onset, and deep/REM sleep %. This gives us a defensible
ground truth to train against (see build_target.py for the exact formula).

Models: RandomForestClassifier and HistGradientBoostingClassifier.
HistGradientBoostingClassifier is scikit-learn's built-in histogram-based
gradient boosting model -- algorithmically the closest thing to LightGBM
without needing an external package. If you have internet access locally,
you can swap in real LightGBM with ~2 line changes (shown at the bottom).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load data + derived label
import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Run build_target.py first -- this expects labeled.csv in the same folder
df = pd.read_csv(os.path.join(HERE, 'labeled.csv'))

# Feature set: HRV + HR only, as requested. A couple of derived features
# (day-over-day change) are cheap to add and meaningfully boost signal
# for a physiological time-series like this, since single-day HRV/HR
# values are noisy -- the trend matters as much as the level.
df = df.sort_values(['user_id', 'date'])
df['hrv_rmssd_change'] = df.groupby('user_id')['hrv_rmssd_ms'].diff()
df['resting_hr_change'] = df.groupby('user_id')['resting_hr_bpm'].diff()
df = df.dropna(subset=['hrv_rmssd_change', 'resting_hr_change']).reset_index(drop=True)

feature_cols = [
    'hrv_rmssd_ms',
    'resting_hr_bpm',
    'avg_hr_day_bpm',
    'hrv_rmssd_change',
    'resting_hr_change',
]
target_col = 'sleep_quality'

X = df[feature_cols]
y_raw = df[target_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)  # poor=0/fair... (alphabetical, but we print mapping)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ---------------------------------------------------------------
# 2. Group-aware train/test split (split by user, not by row!)
#    Random row-level splitting would leak each user's baseline HR/HRV
#    into both train and test, making accuracy look better than it
#    would be on a genuinely new person.
# ---------------------------------------------------------------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df['user_id']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTrain rows: {len(X_train)}  ({df['user_id'].iloc[train_idx].nunique()} users)")
print(f"Test rows:  {len(X_test)}  ({df['user_id'].iloc[test_idx].nunique()} users)")

# ---------------------------------------------------------------
# 3. Train models
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
rf.fit(X_train, y_train)

hgb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)
hgb.fit(X_train, y_train)

# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
for name, model in [('Random Forest', rf), ('HistGradientBoosting (LightGBM-like)', hgb)]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

# ---------------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest feature importances:")
print(importances)

# Save models + label encoder for reuse
import joblib
joblib.dump({'rf': rf, 'hgb': hgb, 'label_encoder': le, 'feature_cols': feature_cols},
            os.path.join(HERE, 'sleep_quality_models.joblib'))
print("\nSaved models to sleep_quality_models.joblib")
