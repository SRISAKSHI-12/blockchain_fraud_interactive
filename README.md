# Blockchain-Enabled AI-Based Fraud Detection and Risk Management System

Interactive final-year / IEEE research prototype.

## Implemented flow

**Data Sources → Preprocessing → Isolation Forest → Fraud Classification → Ensemble Prediction → Adaptive Risk Score → SHAP Explainability → Alert Prioritization → Blockchain Audit → Analyst Dashboard → Feedback Loop**

## Interactive UI pages

1. **Executive Dashboard**
   - system KPIs
   - architecture view
   - model comparison
   - runtime risk distribution

2. **Analyze Transaction**
   - interactive transaction form
   - fraud probability
   - anomaly score
   - 0–100 risk gauge
   - LOW / MEDIUM / HIGH / CRITICAL level
   - explanation chart
   - risk drivers
   - blockchain receipt

3. **Batch Analyzer**
   - CSV upload
   - bulk scoring
   - risk distribution
   - downloadable scored CSV

4. **Live Monitor**
   - risk timeline
   - probability-vs-risk chart
   - alert queue

5. **Blockchain Explorer**
   - chain verification
   - block browser
   - hash inspection

6. **Model Lab**
   - accuracy, precision, recall, F1, ROC-AUC
   - model comparison
   - research experiment notes

7. **Feedback & Retraining**
   - analyst review labels
   - notes
   - exportable feedback dataset

## ML stack

- Logistic Regression
- Random Forest
- Gradient Boosting
- Soft Voting Ensemble
- Isolation Forest
- SMOTE on training data only
- SHAP explainability with safe fallback

## Risk bands

| Score | Level | Action |
|---|---|---|
| 0–30 | LOW | Approve / Monitor |
| 31–60 | MEDIUM | Manual Review |
| 61–80 | HIGH | Hold + Alert |
| 81–100 | CRITICAL | Reject / Urgent Investigation |

## Run in VS Code

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Optional explicit training

```bash
python train.py
```

## Dataset schema

The app creates a synthetic research dataset automatically. To use your own CSV, provide:

```text
amount
transaction_hour
failed_attempts
account_age_days
velocity_1h
location_change
new_device
device_trust
merchant_risk
channel
is_fraud
```

`is_fraud`: 0 = legitimate, 1 = fraud.

## Project structure

```text
blockchain_fraud_interactive/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── contracts/
│   └── FraudAudit.sol
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── blockchain.py
│   ├── feedback.py
│   └── utils.py
├── data/
├── models/
└── audit/
```

## Research note

The default blockchain implementation is a local SHA-256 tamper-evident chain so the project works immediately without MetaMask/Ganache. `contracts/FraudAudit.sol` is included for later Ethereum integration.

Do not claim example/demo metrics as final IEEE results. Run controlled experiments and report the actual outputs.
