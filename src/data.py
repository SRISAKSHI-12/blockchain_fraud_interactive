from pathlib import Path
import numpy as np
import pandas as pd
from .config import RANDOM_STATE, FEATURES

def ensure_demo_dataset(path: Path, n_rows: int = 10000):
    if path.exists():
        return
    rng = np.random.default_rng(RANDOM_STATE)
    channels = np.array(["UPI","Card","NetBanking","Wallet","Ecommerce"])
    amount = np.clip(rng.lognormal(7.0,1.05,n_rows),20,300000)
    hour = rng.integers(0,24,n_rows)
    fails = rng.poisson(.35,n_rows).clip(0,10)
    age = rng.integers(7,3500,n_rows)
    velocity = rng.poisson(2.1,n_rows).clip(0,30)
    loc = rng.binomial(1,.11,n_rows)
    newdev = rng.binomial(1,.13,n_rows)
    trust = np.clip(rng.beta(5,2,n_rows),0,1)
    merchant = np.clip(rng.beta(2,6,n_rows),0,1)
    channel = rng.choice(channels,n_rows,p=[.36,.27,.14,.08,.15])

    night=((hour<=4)|(hour>=23)).astype(int)
    high=(amount>np.quantile(amount,.92)).astype(int)
    ecom=np.isin(channel,["Ecommerce","Wallet"]).astype(int)
    logit=(-5.1 + 1.35*loc + 1.5*newdev + .52*fails + .17*velocity +
           2.15*merchant - 1.25*trust + .85*night + .9*high + .3*ecom - .00022*age)
    p=1/(1+np.exp(-logit))
    y=rng.binomial(1,np.clip(p,0,.98))

    df=pd.DataFrame({
        "amount":amount.round(2),"transaction_hour":hour,"failed_attempts":fails,
        "account_age_days":age,"velocity_1h":velocity,"location_change":loc,
        "new_device":newdev,"device_trust":trust.round(4),"merchant_risk":merchant.round(4),
        "channel":channel,"is_fraud":y
    })
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)

def load_dataset(path: Path):
    return pd.read_csv(path)

def validate_external_dataset(df: pd.DataFrame):
    missing=[c for c in FEATURES if c not in df.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)
    return True, "OK"
