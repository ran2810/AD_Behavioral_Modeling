# Lane Change Intent Prediction

Supervised learning models for predicting vehicle lane-change intent from trajectory data. Covers the full ML pipeline: data loading, feature engineering, model training, threshold optimisation, and cross-dataset evaluation. Built as a practical study of how model performance changes when training and testing on data collected in different countries with different sensors.

---

## Datasets

**NGSIM** (US-101 and I-80, United States)
Ground-mounted cameras recording at 10 Hz. Trajectories are noisy due to occlusion, camera handoff zones, and position-stitching artefacts. Six recording files covering freeway sections with ramps and auxiliary lanes.
Download: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data

**HighD** (60 recordings, Germany)
Drone-based aerial video recorded at 25 Hz. Trajectories are clean and precise with no stitching artefacts. German Autobahn segments with 2-3 lanes per direction. Left and right lane changes are roughly balanced (~138k vs ~142k in the test set).
Access: https://www.highd-dataset.com

---

## Models

**Random Forest** (`models/rfclassifier.py`)
Trained on individual frames. Class imbalance handled by undersampling the None class (keep_factor = 3) and raising class weights for lane-change classes. Thresholds for Left and Right are optimised on the validation set using per-class precision-recall curves.

**LSTM** (`models/lstm.py`)
Two-layer LSTM with hidden size 64. Processes a sliding window of consecutive frames per vehicle (30 frames = 3 s at 10 Hz for NGSIM; 75 frames = 3 s at 25 Hz for HighD). Class imbalance handled by WeightedRandomSampler at the sequence level. Epoch selection based on validation Right F1 to target the rarer class.

---

## Feature Engineering

Features are computed in `extract_features.py`. All values are in SI units (metres, m/s, m/s²).

Lateral motion:
- v_lat — lateral velocity (derived from Local_X diff for NGSIM; yVelocity for HighD)
- lat_displacement_1s — cumulative lateral displacement over 1 second
- lat_dist_moved_15f — cumulative lateral displacement over 1.5 seconds
- v_lat_accel — rate of change of lateral velocity
- v_lat_lag_5, v_lat_lag_10, v_lat_lag_20 — lateral velocity at 0.5s, 1s, 2s prior

Longitudinal dynamics:
- v_vel — longitudinal speed
- a_long — longitudinal acceleration
- a_long_std_1s — acceleration variance over 1 second (jerky driving signal)

Lane context:
- Lane_ID — current lane number
- can_go_left, can_go_right — physical boundary flags (0 at road edges and ramps)

Lead vehicle interaction:
- actual_gap — bumper-to-bumper gap to the vehicle ahead
- gap_rate_trend_1s — 1-second rolling mean of gap rate of change
- rel_speed — speed difference to the vehicle ahead (positive = lead pulling away)

TTC was dropped after feature importance analysis showed near-zero contribution (< 0.005), as it is already implicit in actual_gap and rel_speed.

---

## Experiment Setup

All experiments use a vehicle-grouped 70/15/15 train/val/test split. Vehicles are assigned entirely to one partition to prevent temporal leakage between a vehicle's frames appearing in both training and test.

Thresholds are optimised on the validation set and applied unchanged to the test set. The test set is only touched once, at the end.

| Experiment           | Train       | Val        | Test       |
|---------------------------------|-------------|------------|------------|
| RF In-domain NGSIM              | NGSIM Train | NGSIM Val  | NGSIM Test |
| RF In-domain HighD              | HighD Train | HighD Val  | HighD Test |
| RF Cross-domain (NGSIM->HighD)  | NGSIM Train | NGSIM Val  | HighD Test |
| RF Cross-domain (HighD->NGSIM)  | HighD Train | HighD Val  | NGSIM Test |
| LSTM In-domain NGSIM            | NGSIM Train | NGSIM Val  | NGSIM Test |
| LSTM In-domain HighD            | HighD Train | HighD Val  | HighD Test |
| LSTM Cross-domain (NGSIM->HighD)| NGSIM Train | NGSIM Val  | HighD Test |
| LSTM Cross-domain (HighD->NGSIM)| HighD Train | HighD Val  | NGSIM Test |

---

## Results

RF results are complete. LSTM experiments are pending.

### RF In-domain NGSIM

| Class     | Precision | Recall | F1   | Support   |
|-----------|-----------|--------|------|-----------|
| None      | 0.982     | 0.985  | 0.984| 1,281,866 |
| Left      | 0.491     | 0.443  | 0.466| 28,376    |
| Right     | 0.331     | 0.281  | 0.304| 10,348    |
| macro avg | 0.601     | 0.570  | 0.585| 1,320,590 |

**Interpretation:**

The RF model achieves a macro F1-score of **0.585** when trained and evaluated on NGSIM. While the model accurately identifies non-lane-change behavior (F1 = 0.984), performance on lane-change classes is considerably lower.

Several factors contribute to this result:

- NGSIM trajectories contain higher levels of measurement noise and trajectory reconstruction artifacts compared to modern drone-based datasets.
- Right lane-change events are significantly less frequent than Left lane changes (10k vs 28k samples in the test set), resulting in a more challenging classification problem.
- Feature importance analysis indicates that the model relies heavily on lateral motion features, suggesting that lane changes are often detected only after lateral movement becomes visible rather than through earlier intent cues.

Overall, NGSIM represents a challenging benchmark for frame-level lane-change intent prediction due to both data quality limitations and class imbalance.

---

### RF In-domain HighD

| Class     | Precision | Recall | F1   | Support   |
|-----------|-----------|--------|------|-----------|
| None      | 0.987     | 0.995  | 0.991| 5,747,599 |
| Left      | 0.814     | 0.631  | 0.711| 102,093   |
| Right     | 0.833     | 0.642  | 0.725| 105,833   |
| macro avg | 0.878     | 0.756  | 0.809| 5,955,525 |

**Interpretation:**

Performance improves substantially on HighD, with the macro F1-score increasing from **0.585 (NGSIM)** to **0.809 (HighD)**.

**Key observations:**

- Both Left and Right lane-change classes achieve strong and nearly symmetric performance **(F1 ≈ 0.71–0.73)**.
- The balanced distribution of lane-change directions in HighD reduces class imbalance effects.
- HighD's drone-based trajectory acquisition provides smoother and more accurate vehicle motion measurements, enabling the RF model to learn more reliable lane-change patterns.

These results demonstrate that the selected feature set is capable of supporting lane-change prediction when trajectory quality is high and behavioral patterns are consistently represented.

---

### RF Cross-domain (Train NGSIM, Test HighD)

| Class     | Precision | Recall | F1   | Support   |
|-----------|-----------|--------|------|-----------|
| None      | 0.970     | 0.993  | 0.981| 5,747,599 |
| Left      | 0.280     | 0.139  | 0.186| 102,093   |
| Right     | 0.744     | 0.171  | 0.279| 105,833   |
| macro avg | 0.665     | 0.434  | 0.482| 5,955,525 |

**Interpretation:**

When trained on NGSIM and evaluated on HighD, the macro F1-score decreases from **0.809** to **0.482**, indicating a significant domain shift between the datasets.

**Key observations:**

- Performance on the None class remains strong (F1 = 0.981), suggesting that non-lane-change driving behavior is relatively consistent across datasets.
- Lane-change recall drops sharply for both Left and Right maneuvers, indicating that many true lane changes are not detected.
- The model remains reasonably precise when it predicts a maneuver (particularly Right lane changes), but it becomes highly conservative and misses a large proportion of events.

These results suggest that behavioral patterns learned from NGSIM do not transfer effectively to HighD. Differences in trajectory quality, traffic dynamics, vehicle interactions, and lane-change execution characteristics likely contribute to the observed performance degradation.

---

### RF Cross-domain (Train HighD, Test NGSIM)
| Class     | Precision | Recall | F1   | Support   |
|-----------|-----------|--------|------|-----------|
| None      | 0.974     | 0.982  | 0.978| 1,281,866 |
| Left      | 0.312     | 0.158  | 0.209| 28,376    |
| Right     | 0.056     | 0.171  | 0.065| 10,348    |
| macro avg | 0.448     | 0.406  | 0.418| 1,320,590 |

**Interpretation:**

This experiment produces the lowest overall performance, with the macro F1-score decreasing to 0.418.

Compared to the NGSIM → HighD transfer direction, the degradation is even more severe:

- Left lane-change performance drops substantially.
- Right lane-change detection nearly collapses (F1 = 0.065).
- The model frequently classifies lane-change events as non-maneuver behavior.

This asymmetric transfer result suggests that models trained on HighD do not generalize well to the more complex and noisier traffic conditions represented in NGSIM. In contrast, models trained on NGSIM demonstrate better transfer to HighD, likely because they are exposed to a wider range of trajectory variability during training.

---

### Summary

| Model | Train   | Test  | Left F1 | Right F1 | Macro F1 |
|-------|---------|-------|---------|----------|----------|
| RF    | NGSIM   | NGSIM | 0.47    | 0.30     | 0.58     |
| RF    | HighD   | HighD | 0.71    | 0.73     | 0.81     |
| RF    | NGSIM   | HighD | 0.19    | 0.28     | 0.48     |
| RF    | HighD   | NGSIM | 0.21    | 0.066    | 0.42     |
| LSTM  | NGSIM   | NGSIM | pending | pending  | pending  |
| LSTM  | HighD   | HighD | pending | pending  | pending  |
| LSTM  | NGSIM   | HighD | pending | pending  | pending  |
| LSTM  | HighD   | NGSIM | pending | pending  | pending  |

**Key Findings:**

1. **HighD is significantly easier to model than NGSIM**, achieving a macro F1-score of 0.809 compared to 0.585 for the same RF architecture and feature set.
2. **Cross-domain generalization remains challenging**, with substantial performance degradation observed in both transfer directions.
3. **NGSIM-trained models generalize better to HighD than vice versa**, suggesting that exposure to noisier and more diverse traffic behavior may improve robustness under domain shift.
4. **Lane-change recall is the primary limitation in cross-domain settings**, indicating that maneuver intent cues learned from one dataset do not directly transfer to another.


---

## Project Status

Done:
- NGSIM data loading (US-101 + I-80, 6 recording files) with global ID offsets
- HighD data loading (60 recordings) with driving direction normalisation
- Lane-change intent labelling with 5-second forward-looking horizon
- Physical lane boundary flags (can_go_left, can_go_right) for both datasets
- Feature engineering pipeline with dataset-specific frame-rate handling
- Random Forest with undersampling, class weights, and per-class threshold optimisation
- LSTM with lazy sequence dataset (index-based, not pre-materialised), WeightedRandomSampler, and proper val/test separation
- 8-experiment runner with per-experiment CLI arguments and staged dataset loading to avoid out-of-memory (OOM) on Colab
- Benchmark CSV with per-class metrics across all experiments
- RF experiments complete for all 3 scenarios
- RF cross-domain reverse (HighD Train, NGSIM Test) results

Pending:
- LSTM in-domain NGSIM results
- LSTM in-domain HighD results
- LSTM cross-domain (NGSIM Train, HighD Test) results
- LSTM cross-domain reverse (HighD Train, NGSIM Test) results
- Comparison of LSTM vs RF cross-domain generalisation gap

---

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run a single experiment:
```
python main.py --experiment rf_ngsim --data_path data/
python main.py --experiment rf_highd --data_path data/
python main.py --experiment rf_ngsim_highd --data_path data/
python main.py --experiment rf_highd_ngsim --data_path data/
python main.py --experiment lstm_ngsim --data_path data/
python main.py --experiment lstm_highd --data_path data/
python main.py --experiment lstm_ngsim_highd --data_path data/
python main.py --experiment lstm_highd_ngsim --data_path data/
```

Run all 8 experiments (staged loading, one dataset in RAM at a time):
```
python main.py --experiment all --data_path data/
```

RF tuning (grid search over keep_factor and w_right on a 30% vehicle sample):
```
python tune_rf.py --sample 0.30
```

Data layout expected under data_path:
```
data/
  I-80_VehTrajectoryData/
  US-101_VehTrajectoryData/
  highD-dataset-v1.0/data/          (01_tracks.csv, 01_tracksMeta.csv, ...)
```
