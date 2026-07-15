# NYC Taxi Demand & Revenue Prediction

Machine learning system that forecasts hourly taxi demand and revenue for every pickup zone in Manhattan, and translates those forecasts into a practical driver allocation strategy for a ride hailing platform.

## Problem

Taxi drivers and fleet operators constantly face the same two questions: **where** should drivers be, and **when** should they work? Poor positioning means idle drivers in quiet zones and missed trips in busy ones.

This project answers both questions by predicting, for each of Manhattan's taxi zones and each hour of the day:

* **Demand**: total number of trips per zone per hour
* **Revenue**: total fare amount per zone per hour

The forecasts feed a proposed staffing policy that repositions drivers dynamically while preserving driver autonomy.

## Results

Six models were trained (three architectures, each fitted separately for demand and revenue). The fine tuned TabularResNet was selected as the final model for both targets.

| Target  | Model                    | Val RMSE | Val MAE | Val R²    |
| ------- | ------------------------ | -------- | ------- | --------- |
| Demand  | Random Forest            | 29.80    | 15.70   | 0.853     |
| Demand  | XGBoost                  | 20.03    | 11.06   | 0.934     |
| Demand  | TabularResNet            | 17.28    | 9.60    | 0.953     |
| Demand  | **TabularResNet (fine tuned)** | **17.08** | **9.49** | **0.955** |
| Revenue | Random Forest            | 705.48   | 387.51  | 0.850     |
| Revenue | XGBoost                  | 486.54   | 277.64  | 0.929     |
| Revenue | TabularResNet            | 428.32   | 247.83  | 0.949     |
| Revenue | **TabularResNet (fine tuned)** | **424.06** | **245.79** | **0.950** |

On the held out test period the final demand model achieved **R² of 0.955** with a typical error of roughly 9 rides per zone per hour against a mean of 54. Crucially, relative error is lowest exactly where accuracy matters most: high volume central zones such as Upper East Side South (11.8%), Union Square (12.8%) and Midtown Center (15.0%).

## Key Findings

* **Time of day dominates.** Cyclical hour encodings (`hour_sin`, `hour_cos`) are the strongest predictors for the neural network, which learns daily demand cycles directly rather than leaning solely on lagged values as the tree models do.
* **Recent history matters.** Two hour demand lags and 24 hour rolling averages are the next most important signals.
* **Demand and revenue are tightly coupled.** Central Manhattan zones dominate both, with peak intensity between 10:00 and 21:00.
* **The model slightly underpredicts extreme peaks.** The proposed allocation policy compensates with a roughly 15% over allocation buffer in peak zones during high demand periods.

## Driver Allocation Policy

The report proposes an hourly staffing policy built on two hour ahead forecasts:

1. Maintain a minimum driver baseline in every zone for city wide coverage.
2. Allocate remaining drivers by a priority score that weights each zone's unmet predicted demand by its average revenue per trip.
3. Apply a peak zone safety buffer (~15%) to protect against underprediction during the busiest hours.
4. Surface ranked zone recommendations through a driver app, so drivers choose from suggested options rather than receiving forced assignments.

## Methodology

### Data

* NYC Yellow Taxi trip records for 2024, filtered to Manhattan pickup zones
* Aggregated from trip level to a zone by hour panel
* Invalid records removed (non positive fares or distances, impossible passenger counts, extreme outliers trimmed at the 99.9th percentile)

### Features

* Calendar features: hour, day of week, month, weekend indicator
* Cyclical encodings of hour (sine and cosine)
* Lagged demand and revenue at 2 hours, 24 hours and 4 weeks (past values only, no leakage)
* 24 hour rolling averages
* Zone identity via entity embeddings in the neural network

### Train / Validation / Test Split

A month based temporal split preserves seasonal, weekly and daily structure:

* **Train**: first three weeks of every month
* **Validation**: final week of January through June
* **Test**: final week of August through December (July is consumed by lag warm up and excluded)

### Models

* **Random Forest**: baseline
* **XGBoost**: gradient boosted trees, GPU accelerated where available
* **TabularResNet**: ResNet style MLP for tabular data with entity embeddings for pickup zone, BatchNorm, SiLU activations, dropout and residual skip connections, trained on log transformed targets. Selected and fine tuned as the final model.

Feature importance for the neural network was computed via permutation importance (increase in RMSE when a feature is shuffled).

## Repository Structure

```
.
├── Final_Notebook.ipynb     # Full pipeline: loading, cleaning, EDA, feature engineering, all six models, evaluation, export
├── BusinessMetrics.ipynb    # Business focused analysis of predictions: zone accuracy, revenue heatmaps, spatial maps, error bias
├── predictions.csv          # Final model predictions vs actuals per zone and hour
```

## Data Setup

The raw data is not committed to the repository due to size. To reproduce:

1. Download the 2024 Yellow Taxi trip records from the [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and place them in `data/` matching the pattern `nyc_taxi_2024-*.csv`.
2. Download the taxi zone lookup table and taxi zone shapefile from the same page into `data/taxi_zone_lookup.csv` and `data/taxi_zones/`.

## Running the Project

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch geopandas statsmodels joblib
```

1. Run `Final_Notebook.ipynb` end to end. This builds the datasets, trains all six models, evaluates them, exports `predictions.csv` and saves trained models to `saved_models/`.
2. Run `BusinessMetrics.ipynb` to generate the business analysis: zone accuracy maps, revenue heatmaps by zone and hour, day of week revenue comparisons, and prediction bias plots.

A CUDA capable GPU is recommended for the XGBoost and PyTorch models; both fall back to CPU automatically.

## Limitations

* External factors (weather, public events, disruptions) are not modelled, so sudden demand spikes are not captured.
* Observed trips are used as a proxy for true demand; unmet demand in undersupplied zones is invisible in the training data, which can cause systematic underprediction there.
* The allocation policy assumes drivers can reposition efficiently and comply voluntarily; travel time and congestion are not modelled.
* Relative errors in near zero demand zones (Randalls Island, Roosevelt Island) are extremely high in percentage terms but operationally insignificant.
