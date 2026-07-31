"""
Simulation for post-operative patients data

Note: All of the following are synthetic approximations for demonstration purposes only.

Added fields:
- surgery_type: category of procedure
- days_since_surgery: 0 on day of surgery, increasing after
- pain_score: 0-10 self-report scale, decays over recovery
- opioid_mg: daily opioid analgesic dose (morphine-equivalent mg)
- body_temp_c: elevated post-op (inflammatory response), normalizes ~5-7 days
- wbc_count: white blood cell count (10^9/L), elevated post-op (infection/inflammation marker), normalizes over ~7-10 days

Physiological perturbation applied post-surgery:
- hrv_rmssd_ms: suppressed by pain + opioids (autonomic suppression)
- resting/avg HR: elevated by pain + fever
- sleep_efficiency: reduced by pain + hospital/recovery disruption
- sleep_latency: increased by pain
- wake_after_sleep_onset: increased by pain + nocturnal vitals checks
- sleep_stage_rem_pct: suppressed by opioids (well-documented REM suppression)
- sleep_stage_deep_pct: suppressed by pain/inflammation
- sleep_stage_light_pct: increases to compensate (stages must sum to ~1)

sleep quality is recomputed from the perturbed sleep architecture using the same formula as before, 
so the label still reflects genuine sleep quality, not the post-op fields directly.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Run build_target.py first -- this expects labeled.csv in the same folder
df = pd.read_csv(os.path.join(HERE, 'labeled.csv'))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['user_id', 'date']).reset_index(drop=True)

SURGERY_TYPES = ['orthopedic', 'abdominal', 'cardiac', 'general']

# --- Assign one simulated surgery per user, roughly in the middle third of
# --- their observed date range, so we get pre-op and post-op days for each.
surgery_info = {}
for uid, g in df.groupby('user_id'):
    dates = g['date'].values
    n = len(dates)
    # surgery happens somewhere in the middle third of their date range
    idx = RNG.integers(int(n * 0.3), int(n * 0.6))
    surgery_date = dates[idx]
    surgery_info[uid] = {
        'surgery_date': surgery_date,
        'surgery_type': RNG.choice(SURGERY_TYPES),
        'baseline_pain': RNG.uniform(5, 8),      # initial post-op pain level
        'baseline_opioid': RNG.uniform(20, 60),  # initial daily morphine-equiv mg
        'recovery_rate': RNG.uniform(0.15, 0.35),# how fast pain/opioid/inflammation decay
    }

surgery_df = pd.DataFrame.from_dict(surgery_info, orient='index')
surgery_df.index.name = 'user_id'
df = df.merge(surgery_df, on='user_id', how='left')

df['days_since_surgery'] = (df['date'] - df['surgery_date']).dt.days

# --- Simulate pain, opioid dose, temp, WBC as a function of days since surgery
is_post = df['days_since_surgery'] >= 0
d = df['days_since_surgery'].clip(lower=0)

decay = np.exp(-df['recovery_rate'] * d)
df['pain_score'] = np.where(is_post, (df['baseline_pain'] * decay + RNG.normal(0, 0.6, len(df))).clip(0, 10), 0.0)
df['opioid_mg'] = np.where(is_post, (df['baseline_opioid'] * decay + RNG.normal(0, 3, len(df))).clip(0, None), 0.0)
df['body_temp_c'] = np.where(is_post, 36.6 + 1.1 * np.exp(-0.5 * d) + RNG.normal(0, 0.15, len(df)), 36.6 + RNG.normal(0, 0.15, len(df)))
df['wbc_count'] = np.where(is_post, 6.5 + 5.0 * np.exp(-0.35 * d) + RNG.normal(0, 0.4, len(df)), 6.5 + RNG.normal(0, 0.4, len(df)))
df['wbc_count'] = df['wbc_count'].clip(3.5, None)

# Pre-op rows: no surgery has happened yet from the patient's perspective
df.loc[~is_post, ['pain_score', 'opioid_mg']] = 0.0

# --- Causal perturbation of physiology and sleep architecture post-surgery
pain = df['pain_score']
opioid = df['opioid_mg']
fever = (df['body_temp_c'] - 36.6).clip(lower=0)

df['hrv_rmssd_ms'] = df['hrv_rmssd_ms'] * (1 - 0.035 * pain - 0.004 * opioid).clip(0.25, 1.0)
df['resting_hr_bpm'] = df['resting_hr_bpm'] * (1 + 0.02 * pain + 0.05 * fever)
df['avg_hr_day_bpm'] = df['avg_hr_day_bpm'] * (1 + 0.015 * pain + 0.04 * fever)

df['sleep_efficiency'] = (df['sleep_efficiency'] - 0.02 * pain - 0.05 * fever).clip(0.3, 0.99)
df['sleep_latency_min'] = df['sleep_latency_min'] + 2.5 * pain
df['wake_after_sleep_onset_min'] = df['wake_after_sleep_onset_min'] + 3.0 * pain

rem_suppression = (0.012 * opioid).clip(0, 0.15)
deep_suppression = (0.01 * pain).clip(0, 0.12)
df['sleep_stage_rem_pct'] = (df['sleep_stage_rem_pct'] - rem_suppression).clip(0.03, None)
df['sleep_stage_deep_pct'] = (df['sleep_stage_deep_pct'] - deep_suppression).clip(0.03, None)
# renormalize stages to sum to 1, light picks up the difference
stage_sum = df['sleep_stage_rem_pct'] + df['sleep_stage_deep_pct']
df['sleep_stage_light_pct'] = (1 - stage_sum).clip(0.2, None)

# --- Recompute sleep_quality label from the (now perturbed) sleep architecture,
#     using the same formula/weights as before -- unchanged from build_target.py
def z(s):
    return (s - s.mean()) / s.std()

score = (
    z(df['sleep_efficiency']) * 2.0
    - z(df['sleep_latency_min']) * 1.0
    - z(df['wake_after_sleep_onset_min']) * 1.0
    + z(df['sleep_stage_deep_pct']) * 1.0
    + z(df['sleep_stage_rem_pct']) * 0.5
)
df['sleep_quality_score'] = score
q1, q2 = score.quantile([1/3, 2/3])
df['sleep_quality'] = pd.cut(score, bins=[-np.inf, q1, q2, np.inf], labels=['poor', 'fair', 'good'])

df.to_csv(os.path.join(HERE, 'labeled_postop.csv'), index=False)
print(df['sleep_quality'].value_counts())
print("\nPost-op rows:", is_post.sum(), " / Pre-op rows:", (~is_post).sum())
print("\nSample post-op fields:")
print(df.loc[is_post, ['days_since_surgery','pain_score','opioid_mg','body_temp_c','wbc_count']].describe())
