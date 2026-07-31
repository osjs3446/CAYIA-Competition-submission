import pandas as pd
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Expects wearables_health_6mo_daily.csv in the same folder as this script
df = pd.read_csv(os.path.join(HERE, 'wearables_health_6mo_daily.csv'))

needed = ['sleep_efficiency','sleep_latency_min','wake_after_sleep_onset_min',
          'sleep_stage_rem_pct','sleep_stage_deep_pct','hrv_rmssd_ms',
          'resting_hr_bpm','avg_hr_day_bpm']
df = df.dropna(subset=needed).reset_index(drop=True)

def z(s):
    return (s - s.mean()) / s.std()

# Composite sleep-quality score: efficiency & deep/REM% count positively,
# latency & wake-after-sleep-onset count negatively (standard sleep-science convention)
score = (
    z(df['sleep_efficiency']) * 2.0
    - z(df['sleep_latency_min']) * 1.0
    - z(df['wake_after_sleep_onset_min']) * 1.0
    + z(df['sleep_stage_deep_pct']) * 1.0
    + z(df['sleep_stage_rem_pct']) * 0.5
)
df['sleep_quality_score'] = score

q1, q2 = score.quantile([1/3, 2/3])
df['sleep_quality'] = pd.cut(score, bins=[-np.inf, q1, q2, np.inf], labels=['poor','fair','good'])

print(df['sleep_quality'].value_counts())
print(q1, q2)
df.to_csv(os.path.join(HERE, 'labeled.csv'), index=False)
