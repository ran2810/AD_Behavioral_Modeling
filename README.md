# AD Behavioral Modeling: Lane Change Intent Prediction

This project implements a machine learning pipeline to predict vehicle lane change intent using the NGSIM (US-101 and I-80) trajectory datasets. The objective is to predict lane change intent as "Left" and "Right" before they occur using a 5-second prediction horizon.
Data can be downloaded from here https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data

##  Project Structure

- **`data/`**: Raw NGSIM trajectory data (US-101 and I-80).
- **`preprocess.py`**: Loads raw data, adds lane change intent labels with a 5s horizon, and applies physical lane-boundary constraints.
- **`models/`**:
    - `rfclassifier.py`: OOP-based Random Forest implementation including threshold optimization and model persistence.
    - `ltsm.py`: Sequential trajectory modeling (Next Steps).
- **`notebooks/`**: Playground area for jupyter notebooks and data exploration.
- **`results/`**: Logged runs containing Feature Importance plots, Confusion Matrices, and `metrics_report.png`.
- **`utils/`**:
    - `data_prep.py`: Handles vehicle-based splitting and `sampling_keep_factor` to prevent data leakage.
    - `visualize.py`: BEV (Bird’s Eye View) trajectory visualization.
- **`main.py`**: Orchestration script for the end-to-end pipeline.

## Experimental Trials & Best Results

The model underwent iterative tuning to address the high noise floor in the NGSIM dataset and significant class imbalance.

### Top Performance Metrics (Random Forest)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **None (0)** | 0.98 | 0.99 | 0.98 | 1,698,261 |
| **Left (1)** | 0.49 | 0.43 | 0.46 | 38,805 |
| **Right (2)** | 0.35 | 0.26 | 0.30 | 14,462 |

* **Optimal Thresholds**: Left: 0.8052 | Right: 0.8442.
* **Best Weights**: `{0: 1.0, 1: 3.0, 2: 10.0}`.

### Why "Left" outperforms "Right"?
In highway trajectory data, Left lane changes are typically more aggressive (overtaking), resulting in higher lateral velocity ($v_{lat}$) and clearer closing gaps. Right lane changes are often "lazier" (yielding or exiting), making them harder to distinguish from lane-drifting noise.

##  Key Insights & Feature Evolution

1. **The Sensor Paradox**: Raw, unclipped `v_lat` yielded better precision than filtered data, as the Random Forest utilized sensor "spikes" as early indicators of boundary crossing.
2. **Contextual Logic**: Adding `Lane_ID` and binary flags like `can_go_right` significantly improved precision by acting as physical logic gates.
3. **Temporal Features**: The addition of `v_lat_lag

## Next Steps: LSTM Transition
To move beyond the current **0.46 (Left)** and **0.30 (Right)** F1-scores, the project is transitioning to a Long Short-Term Memory (LSTM) model to:
- Process trajectories as continuous temporal sequences rather than independent frames.
- Utilize hidden states to maintain long-term driving context.
- Improve "Interaction Awareness" by modeling how surrounding vehicle gaps influence intent over time.