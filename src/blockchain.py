import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

class BlockchainAudit:
    def __init__(self,path:Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.chain=self._load()
    def canonical(self,d):return json.dumps(d,sort_keys=True,separators=(",",":"),default=str)
    def h(self,d):return hashlib.sha256(self.canonical(d).encode()).hexdigest()
    def _load(self):
        if self.path.exists():
            try:return json.loads(self.path.read_text())
            except:pass
        g={"index":0,"timestamp":datetime.now(timezone.utc).isoformat(),"transaction_hash":"GENESIS",
           "prediction":"GENESIS","fraud_probability":0,"risk_score":0,"risk_level":"GENESIS",
           "explanation_hash":"GENESIS","action":"GENESIS","model_version":"GENESIS","previous_hash":"0"*64}
        g["hash"]=self.h(g);chain=[g];self.path.write_text(json.dumps(chain,indent=2));return chain
    def add_record(self,transaction,prediction,fraud_probability,risk_score,risk_level,explanation,action,model_version):
        b={"index":len(self.chain),"timestamp":datetime.now(timezone.utc).isoformat(),
           "transaction_hash":self.h(transaction),"prediction":prediction,
           "fraud_probability":round(float(fraud_probability),6),"risk_score":round(float(risk_score),2),
           "risk_level":risk_level,"explanation_hash":self.h(explanation),"action":action,
           "model_version":model_version,"previous_hash":self.chain[-1]["hash"]}
        b["hash"]=self.h(b);self.chain.append(b);self.path.write_text(json.dumps(self.chain,indent=2));return b
    def verify_chain(self):
        for i,b in enumerate(self.chain):
            saved=b.get("hash");body=dict(b);body.pop("hash",None)
            if saved!=self.h(body):return False,f"Block {i} hash mismatch"
            if i>0 and b["previous_hash"]!=self.chain[i-1]["hash"]:return False,f"Block {i} link mismatch"
        return True,"Ledger integrity verified: all hashes and links are valid."
    def records_dataframe(self):
        return pd.DataFrame([b for b in self.chain if b.get("index",0)>0])
