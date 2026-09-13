import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_TITLE, DATA_PATH, MODEL_DIR, AUDIT_PATH, FEEDBACK_PATH
from src.data import ensure_demo_dataset, load_dataset, validate_external_dataset
from src.modeling import train_or_load_models, predict_transaction
from src.blockchain import BlockchainAudit
from src.feedback import save_feedback, load_feedback
from src.utils import risk_action, risk_color, format_currency

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --card:#ffffff;
  --border:#e6eaf0;
  --muted:#667085;
}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg,#f8fbff 0%,#ffffff 50%);}
.block-container {padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1450px;}
.hero {
  border:1px solid var(--border); border-radius:20px; padding:24px 28px;
  background:linear-gradient(135deg,#0b1f4d 0%,#153a7a 52%,#365fa6 100%);
  color:white; margin-bottom:18px;
}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:6px 0 0; opacity:.92}
.card {
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:16px; box-shadow:0 2px 10px rgba(16,24,40,.04);
}
.section-title {font-weight:800; font-size:1.08rem; margin:0 0 8px;}
.muted {color:var(--muted);}
.pill {display:inline-block;padding:6px 12px;border-radius:999px;font-weight:700;font-size:.85rem}
.low {background:#e8f8ee;color:#167a45}
.medium {background:#fff6d8;color:#9a6b00}
.high {background:#fff0e4;color:#c15d00}
.critical {background:#ffe7e7;color:#b42318}
[data-testid="stMetric"] {background:#fff;border:1px solid var(--border);padding:12px;border-radius:15px;}
div[data-baseweb="tab-list"] {gap:10px;}
button[data-baseweb="tab"] {border-radius:10px;padding:8px 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def boot():
    ensure_demo_dataset(DATA_PATH)
    df = load_dataset(DATA_PATH)
    bundle = train_or_load_models(df, MODEL_DIR)
    return df, bundle

df, bundle = boot()
audit = BlockchainAudit(AUDIT_PATH)

with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.caption("Blockchain-Enabled Fraud Detection & Risk Management")
    page = st.radio(
        "Navigation",
        ["Executive Dashboard","Analyze Transaction","Batch Analyzer","Live Monitor","Blockchain Explorer","Model Lab","Feedback & Retraining"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("### System Health")
    st.success("Models online")
    st.info(f"Blocks: {len(audit.chain)}")
    st.caption(f"Model version: {bundle['metrics']['model_version']}")
    st.caption(f"Best ROC-AUC: {bundle['metrics']['roc_auc']:.3f}")

st.markdown(f"""
<div class="hero">
  <h1>{APP_TITLE}</h1>
  <p>Interactive fraud intelligence platform: anomaly detection, ensemble ML, explainable risk scoring, alerts, blockchain auditability and analyst feedback.</p>
</div>
""", unsafe_allow_html=True)

def gauge(score, title="Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix":"/100"},
        title={"text": title},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#163a7a"},
            "steps":[
                {"range":[0,30],"color":"#dff3e8"},
                {"range":[30,60],"color":"#fff0b8"},
                {"range":[60,80],"color":"#ffd7b5"},
                {"range":[80,100],"color":"#ffc7c7"},
            ],
            "threshold":{"line":{"color":"#7f1d1d","width":4},"thickness":0.8,"value":float(score)}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=15,r=15,t=45,b=10))
    return fig

def risk_pill(level):
    cls = {"LOW":"low","MEDIUM":"medium","HIGH":"high","CRITICAL":"critical"}[level]
    return f'<span class="pill {cls}">{level}</span>'

def recent_records():
    x = audit.records_dataframe()
    if not x.empty and "timestamp" in x:
        x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    return x

if page == "Executive Dashboard":
    rec = recent_records()
    fraud_rate = 0 if rec.empty else 100*(rec["prediction"]=="FRAUD").mean()
    avg_risk = 0 if rec.empty else rec["risk_score"].mean()
    alert_count = 0 if rec.empty else rec["risk_level"].isin(["HIGH","CRITICAL"]).sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Dataset Transactions", f"{len(df):,}")
    c2.metric("Historical Fraud Rate", f"{100*df['is_fraud'].mean():.2f}%")
    c3.metric("Analyzed Transactions", f"{len(rec):,}")
    c4.metric("Current Fraud Predictions", f"{fraud_rate:.1f}%")
    c5.metric("High/Critical Alerts", int(alert_count))

    left,right = st.columns([1.35,1])
    with left:
        st.markdown('<div class="card"><div class="section-title">System Workflow</div>', unsafe_allow_html=True)
        st.markdown("""
**1. Data Sources** → Bank/Card/UPI/E-commerce/Device data  
**2. Preprocessing** → cleaning, encoding, scaling, feature engineering  
**3. Anomaly Detection** → Isolation Forest  
**4. Fraud Classification** → Logistic Regression + Random Forest + Gradient Boosting  
**5. Ensemble Decision** → soft voting + anomaly evidence  
**6. Risk Scoring** → 0–100 adaptive score  
**7. Explainability** → SHAP / contribution fallback  
**8. Alert Prioritization** → LOW / MEDIUM / HIGH / CRITICAL  
**9. Blockchain Audit** → SHA-256 tamper-evident record  
**10. Analyst Feedback** → reviewed labels for next retraining cycle
""")
        st.markdown("</div>", unsafe_allow_html=True)

        model_df = pd.DataFrame(bundle["metrics"]["model_comparison"])
        st.plotly_chart(px.bar(model_df, x="model", y=["precision","recall","f1"], barmode="group",
                               title="Model Quality Comparison"), use_container_width=True)

    with right:
        st.plotly_chart(gauge(avg_risk, "Average Runtime Risk"), use_container_width=True)
        if rec.empty:
            st.info("No runtime transactions yet. Analyze a transaction to populate monitoring charts.")
        else:
            counts = rec["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"], fill_value=0).reset_index()
            counts.columns=["Risk","Count"]
            st.plotly_chart(px.pie(counts, names="Risk", values="Count", hole=.55, title="Runtime Risk Distribution"), use_container_width=True)

    st.markdown("### Architecture Layers")
    arch = pd.DataFrame([
        ["Data Sources","Bank, Credit Card, UPI, E-commerce, Device & Customer signals"],
        ["Data Layer","Ingestion → preprocessing → feature engineering"],
        ["ML Layer","Isolation Forest + classification ensemble + risk engine + explainability"],
        ["Prediction Layer","Fraud/legitimate + probability + risk + explanation + alert"],
        ["Blockchain Layer","Transaction hash + prediction + risk + timestamp + model version"],
        ["Application Layer","Analyst dashboard + alerts + reports + feedback"],
    ], columns=["Layer","Implementation"])
    st.dataframe(arch, use_container_width=True, hide_index=True)

elif page == "Analyze Transaction":
    st.subheader("Single Transaction Risk Analysis")
    st.caption("Change the values interactively and analyze. The decision is written to the audit ledger.")

    with st.form("tx_form"):
        a,b,c = st.columns(3)
        with a:
            amount = st.number_input("Transaction Amount (₹)", 0.0, 1_000_000.0, 3500.0, 100.0)
            channel = st.selectbox("Channel", ["UPI","Card","NetBanking","Wallet","Ecommerce"])
            transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
        with b:
            failed_attempts = st.slider("Failed Attempts",0,10,0)
            velocity_1h = st.slider("Transactions in Previous Hour",0,30,2)
            account_age_days = st.number_input("Account Age (days)",1,6000,420)
        with c:
            location_change = st.toggle("Location Changed")
            new_device = st.toggle("New Device")
            device_trust = st.slider("Device Trust",0.0,1.0,0.76,0.01)
            merchant_risk = st.slider("Merchant Risk",0.0,1.0,0.22,0.01)
        submitted = st.form_submit_button("Run Fraud Analysis", type="primary", use_container_width=True)

    if submitted:
        tx = dict(
            amount=amount, transaction_hour=transaction_hour, failed_attempts=failed_attempts,
            account_age_days=account_age_days, velocity_1h=velocity_1h,
            location_change=int(location_change), new_device=int(new_device),
            device_trust=device_trust, merchant_risk=merchant_risk, channel=channel
        )
        with st.spinner("Running ensemble inference, anomaly analysis and explanation..."):
            result = predict_transaction(bundle, tx)
            action = risk_action(result["risk_level"])
            block = audit.add_record(
                transaction=tx, prediction=result["prediction"],
                fraud_probability=result["fraud_probability"], risk_score=result["risk_score"],
                risk_level=result["risk_level"], explanation=result["top_factors"],
                action=action, model_version=bundle["metrics"]["model_version"]
            )
            time.sleep(.2)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Prediction", result["prediction"])
        m2.metric("Fraud Probability", f"{100*result['fraud_probability']:.1f}%")
        m3.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
        m4.metric("Recommended Action", action)

        l,r = st.columns([1,1.4])
        with l:
            st.plotly_chart(gauge(result["risk_score"]), use_container_width=True)
            st.markdown(f"### Risk Level {risk_pill(result['risk_level'])}", unsafe_allow_html=True)
            st.markdown("**Risk Drivers**")
            for d in result["risk_drivers"]:
                st.write("•", d)
        with r:
            exp = pd.DataFrame(result["top_factors"]).sort_values("impact")
            fig = px.bar(exp, x="impact", y="feature", orientation="h", title="Explanation / SHAP Contribution")
            fig.add_vline(x=0, line_width=1)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(result["explanation_method"])
            st.markdown("**Blockchain Audit Receipt**")
            st.code(
                f"Block: {block['index']}\n"
                f"Transaction Hash: {block['transaction_hash']}\n"
                f"Block Hash: {block['hash']}\n"
                f"Previous Hash: {block['previous_hash']}",
                language=None
            )

elif page == "Batch Analyzer":
    st.subheader("Batch Fraud Analyzer")
    st.caption("Upload a CSV using the project feature schema. The app scores every row and produces an interactive risk report.")

    template = df.drop(columns=["is_fraud"]).head(25)
    st.download_button("Download CSV Template", template.to_csv(index=False).encode(),
                       file_name="fraud_batch_template.csv", mime="text/csv")

    up = st.file_uploader("Upload transactions CSV", type=["csv"])
    if up is not None:
        incoming = pd.read_csv(up)
        ok, msg = validate_external_dataset(incoming)
        if not ok:
            st.error(msg)
        else:
            if st.button("Analyze Batch", type="primary"):
                rows = []
                prog = st.progress(0)
                for i, row in incoming.iterrows():
                    result = predict_transaction(bundle, row.to_dict())
                    rows.append({
                        **row.to_dict(),
                        "prediction": result["prediction"],
                        "fraud_probability": result["fraud_probability"],
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                        "action": risk_action(result["risk_level"]),
                    })
                    prog.progress((i+1)/len(incoming))
                out = pd.DataFrame(rows)
                st.session_state["batch_results"] = out

    if "batch_results" in st.session_state:
        out = st.session_state["batch_results"]
        x1,x2,x3,x4 = st.columns(4)
        x1.metric("Rows",len(out))
        x2.metric("Predicted Fraud",int((out["prediction"]=="FRAUD").sum()))
        x3.metric("High/Critical",int(out["risk_level"].isin(["HIGH","CRITICAL"]).sum()))
        x4.metric("Mean Risk",f"{out['risk_score'].mean():.1f}")
        st.plotly_chart(px.histogram(out,x="risk_score",nbins=20,color="risk_level",title="Batch Risk Distribution"),use_container_width=True)
        st.dataframe(out,use_container_width=True,hide_index=True)
        st.download_button("Download Scored CSV", out.to_csv(index=False).encode(),
                           file_name="scored_transactions.csv", mime="text/csv")

elif page == "Live Monitor":
    st.subheader("Live Monitoring & Alert Dashboard")
    rec = recent_records()
    if rec.empty:
        st.info("No records yet. Analyze transactions first.")
    else:
        a,b,c,d = st.columns(4)
        a.metric("Transactions",len(rec))
        b.metric("Fraud",int((rec["prediction"]=="FRAUD").sum()))
        c.metric("Critical",int((rec["risk_level"]=="CRITICAL").sum()))
        d.metric("Avg Risk",f"{rec['risk_score'].mean():.1f}")

        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(rec,x="timestamp",y="risk_score",color="risk_level",markers=True,title="Risk Timeline"),use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(rec,x="fraud_probability",y="risk_score",color="risk_level",
                                       size="risk_score",hover_data=["prediction","action"],
                                       title="Probability vs Risk"),use_container_width=True)
        st.markdown("### Alert Queue")
        alerts = rec[rec["risk_level"].isin(["HIGH","CRITICAL"])].sort_values("timestamp",ascending=False)
        st.dataframe(alerts,use_container_width=True,hide_index=True)

elif page == "Blockchain Explorer":
    st.subheader("Blockchain Audit Explorer")
    ok,msg = audit.verify_chain()
    st.success(msg) if ok else st.error(msg)
    blocks = pd.DataFrame(audit.chain)
    st.dataframe(blocks,use_container_width=True,hide_index=True)
    if len(blocks)>1:
        idx = st.selectbox("Inspect Block", blocks["index"].tolist()[::-1])
        block = blocks[blocks["index"]==idx].iloc[0].to_dict()
        st.json(block)
    st.caption("Prototype ledger stores hashes and risk metadata rather than raw sensitive transaction values.")

elif page == "Model Lab":
    st.subheader("Model Lab")
    md = pd.DataFrame(bundle["metrics"]["model_comparison"])
    st.dataframe(md,use_container_width=True,hide_index=True)

    t1,t2,t3 = st.tabs(["Comparison","Evaluation Notes","Research Experiments"])
    with t1:
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.bar(md,x="model",y="roc_auc",title="ROC-AUC"),use_container_width=True)
        c2.plotly_chart(px.bar(md,x="model",y="f1",title="F1 Score"),use_container_width=True)
    with t2:
        st.markdown(f"""
- **Best current model:** {bundle['metrics']['best_model']}
- **ROC-AUC:** {bundle['metrics']['roc_auc']:.4f}
- **Imbalance handling:** {bundle['metrics']['imbalance_method']}
- Split is performed **before** oversampling to avoid data leakage.
- Isolation Forest is fitted on legitimate training samples.
- Ensemble decision combines supervised classifiers with anomaly and behavioral evidence.
""")
    with t3:
        st.markdown("""
Recommended IEEE experiments:
1. Model comparison: Logistic Regression vs Random Forest vs Gradient Boosting vs Ensemble.
2. Original imbalanced training vs SMOTE.
3. Binary classification vs adaptive risk scoring.
4. Explainability examples using SHAP.
5. Blockchain write/verification overhead.
6. Ablation: ML → ML+SMOTE → +Explainability → +Risk → Complete System.
""")

elif page == "Feedback & Retraining":
    st.subheader("Analyst Feedback & Retraining Loop")
    rec = recent_records()
    if rec.empty:
        st.info("Analyze at least one transaction first.")
    else:
        labels = rec["transaction_hash"].tolist()[::-1]
        with st.form("fb"):
            tx_hash = st.selectbox("Transaction Hash", labels)
            analyst_label = st.selectbox("Analyst Decision", ["LEGITIMATE","FRAUD"])
            notes = st.text_area("Investigation Notes")
            save = st.form_submit_button("Save Analyst Feedback", type="primary")
        if save:
            save_feedback(FEEDBACK_PATH, tx_hash, analyst_label, notes)
            st.success("Feedback saved for the next retraining cycle.")

        feedback = load_feedback(FEEDBACK_PATH)
        if not feedback.empty:
            st.dataframe(feedback.sort_values("timestamp",ascending=False),use_container_width=True,hide_index=True)
            st.download_button("Export Feedback Dataset", feedback.to_csv(index=False).encode(),
                               file_name="analyst_feedback.csv", mime="text/csv")
        st.info("For a production system, retraining should be performed as a governed offline pipeline after label verification, not automatically after each analyst click.")
