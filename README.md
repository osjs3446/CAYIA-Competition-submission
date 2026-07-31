# Postop alert and sleep detection
This is a program designed to lessen the burden of nurses, who often work 12 hour shifts at the hospital. It includes an adaptive, rules based monitoring system and a machine learning (ML) system cordinating together to offer a detailed overview during the night and a overall score after the night.


## Run order (must be in this order to properly run)
1. build_target.py - creates labeled.cvs

2. train_sleep_model.py - needs labled.cvs - creates sleep_quality_models.joblib (Optional, used to create the original 35% result)

3. simulate_postop.py - needs labled.cvs - creates labeled_postop.cvs

4. train_postop_windowed.py - needs labled_postop.cvs - creates postop_sleep_quality_models.joblib

5. integrated_demo.py - needs postop_sleep_quality_models.joblib + simulated_night_stream.py and realtime_alert_engine.py (to import functions) - creates quality prediction based on simulated night

ps.
realtime_alert_engine.py can also be run completely on its own anytime (it has its own built-in demo block), requires only simulate_night_stream.py in the same folder. (Not an ML model)

## Simple explination
Using the realtime_alert_engine.py (the monitoring system) as a overnight alert system to notify nurses of any unsual or anomalous events during epochs with an interval of 5 minutes. At the end of the night, intergrated_demo.py uses postop_sleep_quality_models.joblib (tRandom Forest and HistGradientBoosting trained) and the summarized feed from one night's epochs to predict the final sleep quality of the night.

labeled.cvs is the training dataset with the "sleep_quality" column
labeled_postop.cvs is labeled.cvs with more columns to include postop information, such as "days_since_surgery", "opioid_dosage", or "surgery type"

ps.
The file labeled "Night alert chart.png" is a visual representation of the test case after tuning the single-epoch respiratory threshold to 3.2, where a total of 4 alerts were fired for 3 events.
