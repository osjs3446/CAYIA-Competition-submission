"""
Simulation of a single night of sensor data for a single post-op patient

Signals per epoch:
- hr : heart rate (bpm)
- hrv_rmssd : heart rate variation root mean square and sucsessive differnce (ms)
- resp_rate : respiratory rate (breaths/min)
- movement : movement/restlessness magnitude (0-100)

Events (will be randomly placed between the night) (0-3 events per night):
- 'pain' event : hr up, hrv down, movement up, lasts ~15-30 mins
- "resp" event : respiratory rate drops OR irregular (opioid driven apnea-like event), lasts ~5-15 mins
- baseline drift : natural slow decrease of hr and increase of hrv (deeper sleep changes cardiac signals)

"""
import numpy as np
import pandas as pd

def simulate_night(patient_id, baseline_hr=62, baseline_hrv=45, baseline_resp=14,
                    pain_sensitivity=1.0, opioid_level=20, n_events=None, seed=None):
    rng = np.random.default_rng(seed)
    n_epochs = 96  # 8 hours * 12 epochs/hour (5-min resolution)
    minutes = np.arange(n_epochs) * 5

    # natural overnight drift: HR dips and HRV rises as sleep deepens (first
    # half of night), then both drift back up toward morning
    t = np.linspace(0, 1, n_epochs)
    natural_drift = np.sin(t * np.pi)  # peaks mid-night

    hr = baseline_hr - 6 * natural_drift + rng.normal(0, 1.5, n_epochs)
    hrv = baseline_hrv + 10 * natural_drift + rng.normal(0, 2.5, n_epochs)
    resp = baseline_resp - 1.0 * natural_drift + rng.normal(0, 0.4, n_epochs)
    movement = np.clip(8 - 6 * natural_drift + rng.normal(0, 3, n_epochs), 0, None)

    events = []
    if n_events is None:
        # more opioid / more pain sensitivity -> more likely to have events
        expected_events = 0.4 + 0.02 * opioid_level * pain_sensitivity
        n_events = rng.poisson(min(expected_events, 4))

    for _ in range(n_events):
        etype = rng.choice(['pain', 'resp'], p=[0.55, 0.45])
        start = rng.integers(0, n_epochs - 8)
        if etype == 'pain':
            dur = rng.integers(3, 7)  # 15-35 min
            end = min(start + dur, n_epochs)
            severity = pain_sensitivity * rng.uniform(0.7, 1.4)
            hr[start:end] += 14 * severity
            hrv[start:end] -= 12 * severity
            movement[start:end] += 25 * severity
        else:
            dur = rng.integers(1, 4)  # 5-20 min
            end = min(start + dur, n_epochs)
            severity = (opioid_level / 40) * rng.uniform(0.7, 1.4)
            resp[start:end] -= 5 * severity
            hr[start:end] += 3 * severity  # compensatory tachycardia
        events.append({'type': etype, 'start_min': minutes[start], 'end_min': minutes[min(end, n_epochs-1)]})

    hrv = np.clip(hrv, 8, None)
    resp = np.clip(resp, 6, 24)
    movement = np.clip(movement, 0, 100)

    df = pd.DataFrame({
        'patient_id': patient_id,
        'minute': minutes,
        'hr': hr,
        'hrv_rmssd': hrv,
        'resp_rate': resp,
        'movement': movement,
    })
    return df, events

if __name__ == '__main__':
    import os
    # Save alongside this script, wherever it's actually run from --
    # this way the folder works out of the box for anyone, no path editing needed.
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_night_stream.csv')

    df, events = simulate_night('demo_patient_01', baseline_hr=68, baseline_hrv=32,
                                 baseline_resp=13, pain_sensitivity=1.3, opioid_level=45, seed=11)
    print(df.describe())
    print("\nEmbedded events (ground truth, for validation only -- detector doesn't see this):")
    for e in events:
        print(e)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
