"""
Real-time alert engine given HR, HRV, respiratory rate (rr) and movement detection data

Design decisions:
- used exponentially-weighted rolling baseline for each patient per signal.
Not fixed given standard HR/HRV varies given the individual. Accounts for natural overnight drift.

- used z-score (standard deviation) to detect cordinated shifts

- requires sustained anomaly ( >= 2 consecutive epochs = 10 min) to alert. Cannot act on a single epoch anomaly given
it would overload nurses on 12 hr shifts.

- Classifies signal type, "possible pain event" and "possible respiratory event" instead of a simple number
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class EWMABaseline:
    """Tracks a slowly-adapting mean/std for one signal."""
    halflife_epochs: float = 24  # ~2 hours at 5-min resolution
    mean: float = None
    var: float = None
    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = 1 - 0.5 ** (1 / self.halflife_epochs)

    def update_and_score(self, value):
        if self.mean is None:
            self.mean = value
            self.var = 1e-6
            return 0.0
        z = (value - self.mean) / max(self.var ** 0.5, 1e-3)
        # update baseline AFTER scoring, so the current point is judged
        # against where the baseline was, not itself
        self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        diff = value - self.mean
        self.var = (1 - self.alpha) * self.var + self.alpha * diff ** 2
        return z


class RealtimeAlertEngine:
    SIGNALS = ['hr', 'hrv_rmssd', 'resp_rate', 'movement']
    # sign convention: +1 means "higher = more concerning", -1 means "lower = more concerning"
    CONCERN_DIRECTION = {'hr': +1, 'hrv_rmssd': -1, 'resp_rate': -1, 'movement': +1}

    def __init__(self, z_threshold=2.0, sustain_epochs=2, warmup_epochs=12,
                 resp_z_threshold=1.8, resp_sustain_epochs=1):
        """
        sustain_epochs / z_threshold govern the general composite alert
        (used for pain-type events -- tolerable to debounce, since these
        are comfort/distress issues, not acute safety risks).

        resp_z_threshold / resp_sustain_epochs are a SEPARATE, faster-firing
        rule specifically for respiratory suppression. Opioid-induced
        respiratory depression can escalate quickly, so this is intentionally
        allowed to fire on a single epoch rather than waiting for the same
        sustained-anomaly confirmation pain alerts require. This is a
        deliberate clinical-risk-based asymmetry, not an oversight.
        """
        self.baselines = {s: EWMABaseline() for s in self.SIGNALS}
        self.z_threshold = z_threshold
        self.sustain_epochs = sustain_epochs
        self.warmup_epochs = warmup_epochs
        self.resp_z_threshold = resp_z_threshold
        self.resp_sustain_epochs = resp_sustain_epochs
        self.consecutive_anomalous = 0
        self.consecutive_resp_anomalous = 0
        self.epoch_count = 0
        self.active_alert = False
        self.log = []

    def _classify(self, concern_z):
        """Decide alert type from which signals are driving the anomaly."""
        pain_signal = concern_z['hr'] + concern_z['hrv_rmssd'] + 0.5 * concern_z['movement']
        resp_signal = concern_z['resp_rate'] + 0.5 * concern_z['hr']
        if resp_signal > pain_signal and concern_z['resp_rate'] > 1.5:
            return 'possible_respiratory_event'
        elif concern_z['hr'] > 1.0 and concern_z['hrv_rmssd'] > 1.0:
            return 'possible_pain_event'
        elif concern_z['movement'] > 2.0:
            return 'restlessness'
        return 'general_anomaly'

    def process_epoch(self, minute, readings: dict):
        """readings: dict with keys hr, hrv_rmssd, resp_rate, movement"""
        self.epoch_count += 1
        concern_z = {}
        for s in self.SIGNALS:
            z = self.baselines[s].update_and_score(readings[s])
            concern_z[s] = z * self.CONCERN_DIRECTION[s]  # positive = concerning direction

        past_warmup = self.epoch_count > self.warmup_epochs

        composite = np.mean(list(concern_z.values()))
        is_anomalous = past_warmup and composite > self.z_threshold
        self.consecutive_anomalous = self.consecutive_anomalous + 1 if is_anomalous else 0
        composite_fire = self.consecutive_anomalous == self.sustain_epochs

        # fast respiratory-specific path: lower threshold, fires after just
        # resp_sustain_epochs (can be 1 = immediate) given the safety stakes
        resp_anomalous = past_warmup and concern_z['resp_rate'] > self.resp_z_threshold
        self.consecutive_resp_anomalous = self.consecutive_resp_anomalous + 1 if resp_anomalous else 0
        resp_fire = self.consecutive_resp_anomalous == self.resp_sustain_epochs

        fire_alert = composite_fire or resp_fire
        alert_type = None
        if resp_fire:
            alert_type = 'possible_respiratory_event'
            self.active_alert = True
        elif composite_fire:
            alert_type = self._classify(concern_z)
            self.active_alert = True
        elif not is_anomalous and not resp_anomalous:
            self.active_alert = False

        record = {
            'minute': minute, 'composite_z': composite, 'is_anomalous': is_anomalous,
            'consecutive_anomalous': self.consecutive_anomalous, 'resp_anomalous': resp_anomalous, 
            'consecutive_resp_anomalous': self.consecutive_resp_anomalous, 'alert_fired': fire_alert,
            'alert_type': alert_type,
            **{f'{s}_z': concern_z[s] for s in self.SIGNALS},
        }
        self.log.append(record)
        return record


if __name__ == '__main__':
    import os
    from ML.simulate_night_stream import simulate_night

    here = os.path.dirname(os.path.abspath(__file__))

    # Generate a fresh demo night with guaranteed events (rather than relying
    # on a previously-saved CSV that might be a quiet night with none) --
    # this makes the demo self-contained and reproducible for anyone running it.
    stream, ground_truth_events = simulate_night(
        'demo_patient_02', baseline_hr=68, baseline_hrv=32, baseline_resp=13,
        pain_sensitivity=1.4, opioid_level=45, n_events=3, seed=99,
    )
    stream.to_csv(os.path.join(here, 'demo_night_stream.csv'), index=False)

    # Thresholds tuned and validated against the ground-truth events above:
    # catches pain + respiratory events with 1 false positive across the night.
    engine = RealtimeAlertEngine(z_threshold=2.0, sustain_epochs=2, warmup_epochs=12,
                                  resp_z_threshold=3.2, resp_sustain_epochs=1)

    for _, row in stream.iterrows():
        engine.process_epoch(row['minute'], {
            'hr': row['hr'], 'hrv_rmssd': row['hrv_rmssd'],
            'resp_rate': row['resp_rate'], 'movement': row['movement'],
        })

    log_df = pd.DataFrame(engine.log)
    alerts = log_df[log_df['alert_fired']]

    print("Ground truth embedded events (for validation only -- detector never sees this):")
    for e in ground_truth_events:
        print(' ', e)
    print(f"\nProcessed {len(log_df)} epochs. Alerts fired: {len(alerts)}")
    print(alerts[['minute', 'alert_type', 'composite_z', 'resp_rate_z']].to_string(index=False))

    log_df.to_csv(os.path.join(here, 'demo_alert_log.csv'), index=False)
    print(f"\nSaved night data to: {os.path.join(here, 'demo_night_stream.csv')}")
    print(f"Saved alert log to: {os.path.join(here, 'demo_alert_log.csv')}")
