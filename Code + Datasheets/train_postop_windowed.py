import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Run build_target.py then simulate_postop.py first -- this expects labeled_postop.csv here
df = pd.read_csv(os.path.join(HERE, 'labeled_postop.csv'))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['user_id', 'date'])
df['hrv_rmssd_change'] = df.groupby('user_id')['hrv_rmssd_ms'].diff()
df['resting_hr_change'] = df.groupby('user_id')['resting_hr_bpm'].diff()
df = df.dropna(subset=['hrv_rmssd_change', 'resting_hr_change']).reset_index(drop=True)
df['surgery_type_code'] = LabelEncoder().fit_transform(df['surgery_type'])

# Restrict to a clinically realistic monitoring window: 7 days pre-op
# through 21 days post-op. This is when post-op sleep monitoring actually
# happens in practice -- including 100+ "recovered" days dilutes the signal
# with data that no longer reflects a surgical population.
window = df[(df['days_since_surgery'] >= -7) & (df['days_since_surgery'] <= 21)].copy()
print("Windowed rows:", len(window), " users:", window['user_id'].nunique())

feature_cols = [
    'hrv_rmssd_ms', 'resting_hr_bpm', 'avg_hr_day_bpm',
    'hrv_rmssd_change', 'resting_hr_change',
    'steps', 'stress_score', 'alcohol_units', 'caffeine_mg', 'screen_time_min',
    'days_since_surgery', 'pain_score', 'opioid_mg', 'body_temp_c', 'wbc_count',
    'surgery_type_code',
]
X = window[feature_cols]
le = LabelEncoder()
y = le.fit_transform(window['sleep_quality'])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=window['user_id']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                             random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print("\n=== RF, post-op monitoring window only ===")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=le.classes_))

hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)
hgb.fit(X_train, y_train)
preds2 = hgb.predict(X_test)
print("=== HGB, post-op monitoring window only ===")
print("Accuracy:", accuracy_score(y_test, preds2))
print(classification_report(y_test, preds2, target_names=le.classes_))

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

import joblib
joblib.dump(
    {'rf' : rf, 'hgb' : hgb, 'label_encoder' : le, 'feature_cols': feature_cols},
    os.path.join(HERE, 'postop_sleep_quality_model.joblib')
)
print(f"\nSaved model to: {os.path.join(HERE, 'postop_sleep_qualuty_model.joblib')}")