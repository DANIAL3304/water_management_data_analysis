# AIoT for Smart Water Management: Water Quality Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DANIAL3304/water_management_data_analysis/blob/main/WATER_MANAGEMENT.ipynb)

Machine learning were on IoT sensor data to predict water quality (pH) at different monitoring locations. The goal is to move from slow manual testing to automatic monitoring that can flag problems early.

## Problem statement

- **Slow detection:** manual water testing takes time, so contamination is often found too late.
- **Too much data:** sensors at many locations produce more data than people can check by hand.
- **No early warning:** current systems only show the present state. They don't predict when quality will drop.

## Dataset

**Water Quality Prediction** dataset from the UCI Machine Learning Repository (`water_dataset.mat`)

- 37 monitoring locations in a water network
- 11 sensor features per location, covering physical and chemical properties (407 features in total)
- Target: the pH value at each location (already scaled, so values are around 0.6–0.7)
- Time series: the training set has 423 time steps, and the file also includes a separate test set of 282 steps

`X_tr` stores each time step as a 37 × 11 matrix (locations × features), and `Y_tr` is 37 × 423 (pH per location per time step). For a single location, I took that location's row at every time step, which gives a feature matrix of shape (423, 11) and a target of shape (423,).

In the notebook, location index 0 corresponds to L1 in the slides, index 1 to L2, and so on.

The `.mat` file is not included in this repo. Download it from the UCI repository and upload it when Colab asks for it.

## Approach

| Task | Location(s) | Goal |
|---|---|---|
| Regression | L1–L4 | Predict the exact pH value |
| Classification | L1 | Classify pH as low, medium or high |
| Time series forecasting | L1 | Forecast pH over time with SARIMAX |

We trained separate models for each location instead of mixing locations together. Each location is a different physical environment, so combining them would blur location-specific trends. It would also mean the class boundaries for classification aren't consistent.

**Models used**
- Regression: Linear Regression, Ridge, Lasso, Random Forest Regressor, SVR
- Classification: Logistic Regression, Decision Tree, Random Forest, SVC, KNN
- Forecasting: SARIMAX with the 11 sensor features as exogenous inputs

## Results

### Regression (L1–L4)

80/20 split (338 train, 85 test), default model settings.

**R² on the test set**

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Random Forest | **0.9535** | **0.8863** | **0.9067** | **0.9482** |
| Linear Regression | 0.9409 | 0.8168 | 0.7927 | 0.8970 |
| Ridge | 0.6443 | 0.5659 | 0.5769 | 0.5749 |
| Lasso | -0.0001 | -0.0011 | -0.0125 | -0.0023 |
| SVR | -0.4310 | 0.0000 | -0.0026 | -0.0044 |

Random Forest MAE: 0.0024 (L1), 0.0056 (L2), 0.0052 (L3), 0.0042 (L4).

Random Forest was the best model at every location, and Linear Regression came second. Lasso and SVR did no better than predicting the mean. The pH values are scaled into a very small range, and the defaults don't suit that range: Lasso's `alpha=1.0` shrinks every coefficient to zero, and SVR's `epsilon=0.1` is wider than most of the variation in the target. Tuning these parameters or rescaling the target should help.

### Classification (L1)

pH was split into three equal-width bins: low (41), medium (196) and high (186). The data was then split 80/20.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | **0.9529** | **0.9536** | **0.9529** | **0.9527** |
| Decision Tree | 0.9412 | 0.9420 | 0.9412 | 0.9410 |
| KNN | 0.8706 | 0.8812 | 0.8706 | 0.8689 |
| SVC | 0.6706 | 0.6271 | 0.6706 | 0.6425 |
| Logistic Regression | 0.6588 | 0.6133 | 0.6588 | 0.6299 |

*Precision, recall and F1 are weighted averages.*

Random Forest confusion matrix (85 test samples):

| Actual \ Predicted | medium | high | low |
|---|---|---|---|
| medium | 42 | 2 | 0 |
| high | 1 | 34 | 0 |
| low | 1 | 0 | 5 |

Random Forest got 81 of 85 test samples right. The "low" class is the weakest, with recall of 0.83, because it only has 6 test samples. Logistic Regression and SVC never predicted "low" at all.

### Time series forecasting (L1)

The data was split in time order: the first 80% for training and the last 20% for testing.

- Model: SARIMAX `order=(1,1,1)`, `seasonal_order=(1,1,1,7)`, with the 11 sensor features as exogenous variables
- I raised `maxiter` to 1000 because the default gave a convergence warning

| Metric | Value |
|---|---|
| R² | 0.9474 |
| MAE | 0.0031 |
| RMSE | 0.0039 |

## Key findings

- Random Forest was the most reliable model for both regression (R² 0.89–0.95) and classification (95% accuracy).
- SARIMAX with sensor inputs forecast L1's pH well (R² 0.947).
- Location L10 is much noisier and more volatile than L1, so some sites will be harder to predict than others.
- The models rely heavily on one sensor, `L1_F6`. If that sensor fails or drifts, predictions could suffer.

## Possible applications

- **Pollution detection:** unusual predictions could point to a contamination event.
- **Leak and pressure problems:** sudden changes in sensor trends could signal a leak.
- **Decision support:** model outputs could trigger alerts for the authorities.

## Challenges and limitations

- The raw sensor data is noisy, and there are fluctuations in the time series plots.
- Model performance depends on data quality.
- The regression and classification splits are random, but the data is a time series. The model can see time steps on both sides of a test point, which may make the scores look better than real forecasts would be. The SARIMAX test uses a time-based split, so it doesn't have this problem.
- The notebook only uses the training part of the `.mat` file (423 steps). The separate 282-step test set isn't used yet.
- There was no hyperparameter tuning, and only L1–L4 were modelled.

## Future work

- Improve the regression models by scaling, tuning Lasso, Ridge and SVR, and trying gradient boosting
- Fix the weak "low" class (e.g. class weights or oversampling)
- Reduce sensor noise during preprocessing
- Make the models depend less on a single sensor (`L1_F6`)
- Drop models that consistently underperform
- Use time-based splits for every task and evaluate on the 282-step test set

## How to run

1. Open the notebook in Colab using the badge at the top.
2. Run the upload cell and choose `water_dataset.mat`.
3. Run the remaining cells in order.

Libraries: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn
