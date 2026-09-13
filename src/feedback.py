from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
def save_feedback(path:Path,transaction_hash,analyst_label,notes):
    row=pd.DataFrame([{"timestamp":datetime.now(timezone.utc).isoformat(),"transaction_hash":transaction_hash,
                       "analyst_label":analyst_label,"notes":notes}])
    if path.exists():row=pd.concat([pd.read_csv(path),row],ignore_index=True)
    path.parent.mkdir(parents=True,exist_ok=True);row.to_csv(path,index=False)
def load_feedback(path:Path):
    if not path.exists():return pd.DataFrame(columns=["timestamp","transaction_hash","analyst_label","notes"])
    return pd.read_csv(path)
