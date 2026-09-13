from src.config import DATA_PATH,MODEL_DIR
from src.data import ensure_demo_dataset,load_dataset
from src.modeling import train_or_load_models
ensure_demo_dataset(DATA_PATH)
bundle=train_or_load_models(load_dataset(DATA_PATH),MODEL_DIR)
print("Training complete")
print(bundle["metrics"])
