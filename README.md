# MLCreditModel

A machine learning-based credit risk pipeline that combines XGBoost probability estimation with a client-specific, PnL-driven lending decision framework. The model moves beyond classification accuracy as the primary objective, orienting every design decision — threshold derivation, interest rate pricing, and performance evaluation — toward portfolio return on capital.

---

## Overview

Traditional credit scoring models optimize for classification metrics (accuracy, F1, AUC) using a fixed decision threshold. This pipeline reframes the lending decision as a financial optimization problem: a client is approved if and only if the expected PnL of lending to them is positive, given their individual predicted default probability, revolving balance, exposure at default, and assigned interest rate.

---

## Pipeline Architecture

The pipeline is implemented as four OOP classes with a clean orchestrating `main.py`:

- **`DataPipeline`** — Stratified train / val / test splitting with automatic EAD and revolving balance derivation.
- **`ModelFitter`** — XGBoost training and isotonic probability calibration on a dedicated held-out calibration subset.
- **`RatePricer`** — Client-level interest rate decomposition and k-means bucket discretization.
- **`CreditPortfolio`** — Break-even threshold computation, lending decisions, realized PnL, and benchmark comparison.
- **`visualization.py`** — Stateless plotting functions (unchanged from original procedural implementation).

---

## Dataset

**UCI Default of Credit Card Clients**
- 30,000 clients, Taiwan 2005
- 23 features including demographic data, payment history, bill amounts, and payment amounts
- Target variable: `default.payment.next.month` (binary, 22.1% default rate)
- No missing values or data cleaning required

Two financial variables are derived directly from raw features:

| Variable | Definition | Role |
|---|---|---|
| EAD | `BILL_AMT1` | Loss exposure on default |
| RB | `max(BILL_AMT1 - PAY_AMT1, 0)` | Interest-generating balance |

---

## Methodology

### Data partitioning

| Split | Size | Purpose |
|---|---|---|
| Train | 21,000 (70%) | Model training |
| Val eval | 5,250 (17.5%) | Model evaluation |
| Val cal | 2,250 (7.5%) | Probability calibration |
| Test | 1,500 (5%) | Out-of-sample evaluation |

All splits are stratified by the target variable to preserve the 22.1% default rate across partitions.

### Model

XGBoost classifier with the following key hyperparameters:

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 200 | Ensemble size |
| `max_depth` | 4 | Tree complexity control |
| `learning_rate` | 0.05 | Shrinkage per tree |
| `scale_pos_weight` | 3.5 | Class imbalance correction |
| `reg_lambda` | 2.0 | L2 regularization |
| `eval_metric` | AUC | Internal evaluation |

### Probability calibration

Isotonic calibration via `CalibratedClassifierCV` (cv=5) fitted on the dedicated calibration subset. Corrects the systematic PD overestimation introduced by `scale_pos_weight` without affecting the model's ranking ability (AUC).

### Interest rate model

Client-level rates are decomposed additively:

$$r_i = r_r + \pi + LP + AC + MG + PD_i \times LGD$$

| Component | Value |
|---|---|
| Real rate | 2.5% |
| Inflation | 4.5% |
| Liquidity premium | 0.3% |
| Admin costs | 5.0% |
| Profit margin | 3.5% |
| Risk premium | $PD_i \times LGD$ (variable) |

Continuous rates are discretized into 5 pricing buckets via 1D k-means clustering.

### Lending decision rule

A client is approved if their predicted PD falls below their individual break-even threshold:

$$PD_i^* = \frac{RB_i \cdot r_i^{\text{bucket}}}{RB_i \cdot r_i^{\text{bucket}} + EAD_i \cdot LGD}$$

derived by setting expected PnL = 0. The threshold is heterogeneous — it varies by client as a function of their revolving balance, exposure, and assigned bucket rate.

### PnL framework

$$\text{PnL}_i = \begin{cases} 0 & \text{if rejected} \\ RB_i \cdot r_i^{\text{bucket}} & \text{if approved and no default} \\ -EAD_i \cdot LGD & \text{if approved and default} \end{cases}$$

Performance is benchmarked against a lend-all strategy where every client is approved regardless of their predicted PD.

---

## Results

| Metric | Train | Validation | Test |
|---|---|---|---|
| AUC | 0.7639 | 0.7775 | 0.7698 |
| Accuracy | 64.92% | 65.52% | 64.80% |
| Approval rate | 58.90% | 58.65% | 59.60% |
| Default rate (approved) | 13.67% | 12.99% | 14.21% |
| Capital deployed | 82.36% | 82.98% | 80.64% |
| Benchmark RoC | 4.55% | 4.97% | 0.11% |
| Realized RoC | 10.77% | 11.35% | 7.89% |
| PnL uplift vs benchmark | +$46.1M | +$12.3M | +$4.94M |

The model maintains a spread of approximately 630 basis points over the benchmark return on capital across train and validation. On the test set, where the benchmark collapses to 0.11%, the model generates a realized RoC of 7.89% and an absolute PnL differential of $4.94M over 1,500 clients.

---

## Project Structure

```
MLCreditModel/
│
├── Data/
│   └── UCI_Credit_Card.csv
│
├── Model/
│   ├── xgb_model.pkl
│   └── xgb_model_calibrated.pkl
│
├── DataSplitter.py
├── fitter.py
├── InterestRate.py
├── CreditModel.py
├── visualization.py
└── main.py
```

---

## Requirements

pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```python
# Train and calibrate from scratch
RETRAIN = True

# Load persisted model
RETRAIN = False
```

Run the full pipeline:

```bash
python main.py
```

---

## Key Design Decisions

- **No StandardScaler or PCA.** Tree-based models do not require feature scaling. Removing these transformations improves interpretability without affecting model performance.
- **EAD/RB asymmetry.** Loss calculations use full balance (`BILL_AMT1`); revenue calculations use revolving balance only (`BILL_AMT1 - PAY_AMT1`), reflecting that clients who pay in full generate no interest income.
- **Calibration on dedicated subset.** Isotonic calibration is fitted on a held-out 30% of the validation set, with the remaining 70% reserved for clean evaluation, preserving the integrity of performance metrics.
- **Client-specific threshold.** The break-even PD threshold varies by client, tolerating higher default probability for clients with high revolving balances or high assigned rates where the expected interest income justifies the risk.
- **LGD = 0.75** assumed constant across the portfolio, consistent with industry benchmarks for unsecured consumer credit.

---

## Limitations

The model excludes interchange fee income, which affects the evaluation of clients who pay their balance in full every month (*deadbeats*). These clients produce zero revolving balance, collapsing their break-even threshold to zero and causing the model to reject them regardless of their predicted PD. Incorporating interchange fees would introduce a non-interest revenue source for these clients, producing a positive threshold and allowing the model to evaluate them under the same expected PnL criterion applied to the rest of the portfolio.
