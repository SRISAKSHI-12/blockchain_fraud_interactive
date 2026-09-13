from __future__ import annotations
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from .config import FEATURES,NUMERIC_FEATURES,CATEGORICAL_FEATURES,RANDOM_STATE

def make_preprocessor():
    return ColumnTransformer([
        ("num",StandardScaler(),NUMERIC_FEATURES),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CATEGORICAL_FEATURES),
    ])

def estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1400,class_weight="balanced",random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=220,max_depth=12,min_samples_leaf=2,
                                                class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,learning_rate=.05,max_depth=3,random_state=RANDOM_STATE),
    }

def metrics(y,p,pr):
    return dict(
        accuracy=round(float(accuracy_score(y,pr)),4),
        precision=round(float(precision_score(y,pr,zero_division=0)),4),
        recall=round(float(recall_score(y,pr,zero_division=0)),4),
        f1=round(float(f1_score(y,pr,zero_division=0)),4),
        roc_auc=round(float(roc_auc_score(y,p)),4),
    )

def train_or_load_models(df: pd.DataFrame, model_dir: Path):
    model_dir.mkdir(exist_ok=True)
    bp=model_dir/"bundle.joblib"; mp=model_dir/"metrics.json"
    if bp.exists() and mp.exists():
        b=joblib.load(bp); b["metrics"]=json.loads(mp.read_text()); return b

    X=df[FEATURES].copy(); y=df["is_fraud"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_STATE)
    pre=make_preprocessor()
    Xtr_t=pre.fit_transform(Xtr); Xte_t=pre.transform(Xte)

    imbalance_method="Class-weighted training"
    try:
        from imblearn.over_sampling import SMOTE
        sm=SMOTE(random_state=RANDOM_STATE)
        Xfit,yfit=sm.fit_resample(Xtr_t,ytr)
        imbalance_method="SMOTE on training split only"
    except Exception:
        Xfit,yfit=Xtr_t,ytr

    fitted={}; rows=[]; best_name=None; best_auc=-1
    for name,m in estimators().items():
        m.fit(Xfit,yfit)
        p=m.predict_proba(Xte_t)[:,1]; pr=(p>=.5).astype(int)
        mm=metrics(yte,p,pr); rows.append({"model":name,**mm}); fitted[name]=m
        if mm["roc_auc"]>best_auc: best_auc=mm["roc_auc"]; best_name=name

    e=estimators()
    ensemble=VotingClassifier(
        estimators=[("lr",e["Logistic Regression"]),("rf",e["Random Forest"]),("gb",e["Gradient Boosting"])],
        voting="soft",weights=[1,2,2]
    )
    ensemble.fit(Xfit,yfit)
    ep=ensemble.predict_proba(Xte_t)[:,1]; epr=(ep>=.5).astype(int)
    em=metrics(yte,ep,epr); rows.append({"model":"Soft Voting Ensemble",**em})
    if em["roc_auc"]>=best_auc: best_auc=em["roc_auc"]; best_name="Soft Voting Ensemble"

    legitimate=Xtr_t[np.asarray(ytr)==0]
    anomaly=IsolationForest(n_estimators=200,contamination="auto",random_state=RANDOM_STATE,n_jobs=-1)
    anomaly.fit(legitimate)

    meta={
        "best_model":best_name,"roc_auc":float(best_auc),"imbalance_method":imbalance_method,
        "model_version":"v2.0-interactive","model_comparison":rows
    }
    bundle={"preprocessor":pre,"models":fitted,"ensemble":ensemble,"anomaly":anomaly,"metrics":meta}
    joblib.dump({k:v for k,v in bundle.items() if k!="metrics"},bp)
    mp.write_text(json.dumps(meta,indent=2))
    return bundle

def risk_level(score):
    if score<=30:return "LOW"
    if score<=60:return "MEDIUM"
    if score<=80:return "HIGH"
    return "CRITICAL"

def explain(bundle, tx_df):
    try:
        import shap
        pre=bundle["preprocessor"]; model=bundle["models"]["Random Forest"]
        xt=pre.transform(tx_df); fn=list(pre.get_feature_names_out())
        sv=shap.TreeExplainer(model).shap_values(xt)
        if isinstance(sv,list): vals=np.asarray(sv[1])[0]
        else:
            arr=np.asarray(sv)
            vals=arr[0,:,1] if arr.ndim==3 else arr[0]
        pairs=sorted(zip(fn,vals),key=lambda z:abs(float(z[1])),reverse=True)[:7]
        return [{"feature":str(f).replace("num__","").replace("cat__",""),"impact":round(float(v),4)} for f,v in pairs], "SHAP TreeExplainer"
    except Exception:
        r=tx_df.iloc[0]
        f={
            "Transaction Amount":min(float(r["amount"])/100000,1)*.30,
            "Location Change":float(r["location_change"])*.31,
            "New Device":float(r["new_device"])*.21,
            "Failed Attempts":min(float(r["failed_attempts"])/5,1)*.17,
            "Unusual Time":(1 if int(r["transaction_hour"])<=4 or int(r["transaction_hour"])>=23 else 0)*.12,
            "Account Age":-min(float(r["account_age_days"])/3000,1)*.08,
            "Merchant Risk":float(r["merchant_risk"])*.25,
            "Device Trust":-float(r["device_trust"])*.18,
        }
        p=sorted(f.items(),key=lambda z:abs(z[1]),reverse=True)[:7]
        return [{"feature":k,"impact":round(float(v),4)} for k,v in p], "Contribution fallback (SHAP unavailable/incompatible)"

def predict_transaction(bundle, tx: dict):
    txdf=pd.DataFrame([tx],columns=FEATURES); xt=bundle["preprocessor"].transform(txdf)
    probs=[float(m.predict_proba(xt)[0,1]) for m in bundle["models"].values()]
    probs.append(float(bundle["ensemble"].predict_proba(xt)[0,1]))
    fp=float(np.average(probs,weights=[1,2,2,3]))

    raw=float(-bundle["anomaly"].decision_function(xt)[0])
    an=float(1/(1+np.exp(-4*raw)))
    r=txdf.iloc[0]
    behavior=(.20*int(r["location_change"])+.15*int(r["new_device"])+
              .15*min(float(r["failed_attempts"])/5,1)+.15*min(float(r["velocity_1h"])/15,1)+
              .10*(1 if int(r["transaction_hour"])<=4 or int(r["transaction_hour"])>=23 else 0)+
              .15*float(r["merchant_risk"])+.10*(1-float(r["device_trust"])))
    rs=float(np.clip(100*(.68*fp+.17*an+.15*behavior),0,100))
    level=risk_level(rs); pred="FRAUD" if fp>=.5 else "LEGITIMATE"

    exp,method=explain(bundle,txdf)
    drivers=[]
    if r["location_change"]:drivers.append("Location change detected")
    if r["new_device"]:drivers.append("New device used")
    if r["failed_attempts"]>=2:drivers.append("Multiple failed attempts")
    if r["velocity_1h"]>=8:drivers.append("High transaction velocity")
    if r["merchant_risk"]>=.6:drivers.append("High merchant risk")
    if r["device_trust"]<=.35:drivers.append("Low device trust")
    if int(r["transaction_hour"])<=4 or int(r["transaction_hour"])>=23:drivers.append("Unusual transaction time")
    if not drivers:drivers=["No major rule-based driver detected"]
    return dict(prediction=pred,fraud_probability=fp,risk_score=rs,risk_level=level,
                anomaly_score=an,top_factors=exp,explanation_method=method,risk_drivers=drivers)
