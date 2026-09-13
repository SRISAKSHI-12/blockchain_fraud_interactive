from pathlib import Path
APP_TITLE = "Blockchain-Enabled AI-Based Fraud Detection & Risk Management System"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
AUDIT_DIR = BASE_DIR / "audit"
for p in [DATA_DIR, MODEL_DIR, AUDIT_DIR]:
    p.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "transactions.csv"
AUDIT_PATH = AUDIT_DIR / "blockchain.json"
FEEDBACK_PATH = DATA_DIR / "analyst_feedback.csv"
RANDOM_STATE = 42
FEATURES = [
    "amount","transaction_hour","failed_attempts","account_age_days","velocity_1h",
    "location_change","new_device","device_trust","merchant_risk","channel"
]
NUMERIC_FEATURES = FEATURES[:-1]
CATEGORICAL_FEATURES = ["channel"]
