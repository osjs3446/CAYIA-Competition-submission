import os
import joblib
import numpy as np
import pandas as pd
from ML.simulate_night_stream import simulate_night
from ML.realtime_alert_engine import RealtimeAlertEngine

HERE = os.path.dirname(os.path.abspath(__file__))

def aggregate_night_to_daily(stream: pd.DataFrame) -> dict:
    """Converting one night's epochs to summary to feed to the ML model"""

    return {
    'hrv_rmssd_ms': stream['hrv_rmssd'].mean(),
    'resting_hr_bpm': stream['hr'].quantile(0.1),
    'avg_hr_day_bpm': stream['hr'].mean()
    }

def run_integrated_demo(patient_context: dict, seed: int = 123):
    """
    Hardcoded patient context given intergration does not yet exist (simulation)
    """

    print(f"=== Patient: {patient_context['patient_id']} | "
          f"Day {patient_context['days_since_surgery']} post-op ({patient_context['surgery_type']}) ===\n")

    # pain + opioid easing given passing day
    prev_ctx = {**patient_context, 'days_since_surgery':patient_context['days_since_surgery'] - 1}
    prev_pain = max(prev_ctx['pain_score'] + 1.0, 0)
    prev_opioid = max(prev_ctx['opioids_mg'] + 5.0, 0)

    prev_stream, _ = simulate_night(
        patient_context['patient_id'], baseline_hr=patient_context['baseline_hr'],
        baseline_hrv=patient_context['baseline_hrv'], baseline_resp=patient_context['baseline_resp'],
        pain_sensitivity=prev_pain / 5.0, opioid_level=prev_opioid, seed=seed,
    )
    tonight_stream, tonight_events = simulate_night(
        patient_context['patient_id'], baseline_hr=patient_context['baseline_hr'],
        baseline_hrv=patient_context['baseline_hrv'], baseline_resp=patient_context['baseline_resp'],
        pain_sensitivity=patient_context['pain_score'] / 5.0, opioid_level=patient_context['opioid_mg'],
        seed=seed + 1,
    )
 
    # --- Run the real-time engine over TONIGHT's stream (this is the part
    #     that would actually run live, epoch by epoch, in a real system) ---
    engine = RealtimeAlertEngine(z_threshold=2.0, sustain_epochs=2, warmup_epochs=12,
                                  resp_z_threshold=3.2, resp_sustain_epochs=1)
    for _, row in tonight_stream.iterrows():
        engine.process_epoch(row['minute'], {
            'hr': row['hr'], 'hrv_rmssd': row['hrv_rmssd'],
            'resp_rate': row['resp_rate'], 'movement': row['movement'],
        })
    alert_log = pd.DataFrame(engine.log)
    alerts = alert_log[alert_log['alert_fired']]
 
    print("--- Real-time monitoring (during the night) ---")
    if len(alerts) == 0:
        print("No alerts fired overnight.")
    else:
        for _, a in alerts.iterrows():
            print(f"  [{int(a['minute'])} min] {a['alert_type']} (composite_z={a['composite_z']:.2f})")
 
    # --- Aggregate both nights to daily features, compute the trend cols ---
    prev_daily = aggregate_night_to_daily(prev_stream)
    tonight_daily = aggregate_night_to_daily(tonight_stream)
 
    feature_row = {
        'hrv_rmssd_ms': tonight_daily['hrv_rmssd_ms'],
        'resting_hr_bpm': tonight_daily['resting_hr_bpm'],
        'avg_hr_day_bpm': tonight_daily['avg_hr_day_bpm'],
        'hrv_rmssd_change': tonight_daily['hrv_rmssd_ms'] - prev_daily['hrv_rmssd_ms'],
        'resting_hr_change': tonight_daily['resting_hr_bpm'] - prev_daily['resting_hr_bpm'],
        'steps': patient_context['steps'],
        'stress_score': patient_context['stress_score'],
        'alcohol_units': patient_context['alcohol_units'],
        'caffeine_mg': patient_context['caffeine_mg'],
        'screen_time_min': patient_context['screen_time_min'],
        'days_since_surgery': patient_context['days_since_surgery'],
        'pain_score': patient_context['pain_score'],
        'opioid_mg': patient_context['opioid_mg'],
        'body_temp_c': patient_context['body_temp_c'],
        'wbc_count': patient_context['wbc_count'],
        'surgery_type_code': patient_context['surgery_type_code'],
    }
 
    # --- Load the trained model and predict tonight's overall quality ---
    bundle = joblib.load(os.path.join(HERE, 'postop_sleep_quality_model.joblib'))
    rf, le, feature_cols = bundle['rf'], bundle['label_encoder'], bundle['feature_cols']
    X_row = pd.DataFrame([feature_row])[feature_cols]
    pred = rf.predict(X_row)[0]
    proba = rf.predict_proba(X_row)[0]
    pred_label = le.inverse_transform([pred])[0]
 
    print("\n--- Daily model prediction (morning-after summary) ---")
    print(f"Predicted sleep quality: {pred_label.upper()}")
    print("Class probabilities:", dict(zip(le.classes_, [f'{p:.2f}' for p in proba])))
 
    print(f"\nGround-truth embedded events tonight (for validation only): {len(tonight_events)}")
    for e in tonight_events:
        print(' ', e)
 
    return alert_log, pred_label, proba


if __name__ == '__main__':
    demo_patient = {
        'patient_id': 'demo_patient_integrated',
        'baseline_hr': 68, 'baseline_hrv': 32, 'baseline_resp': 13,
        'days_since_surgery': 3, 'surgery_type': 'orthopedic', 'surgery_type_code': 3,
        'pain_score': 5.5, 'opioid_mg': 35.0, 'body_temp_c': 37.1, 'wbc_count': 9.8,
        'steps': 1200, 'stress_score': 62, 'alcohol_units': 0,
        'caffeine_mg': 40, 'screen_time_min': 90,
    }
    run_integrated_demo(demo_patient)