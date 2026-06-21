import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9ff; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #4f46e5;
        margin-bottom: 0.8rem;
    }
    .risk-high   { background:#fff0f0; border-left:4px solid #e53e3e; border-radius:10px; padding:1rem; }
    .risk-low    { background:#f0fff4; border-left:4px solid #38a169; border-radius:10px; padding:1rem; }
    .risk-medium { background:#fffbf0; border-left:4px solid #d69e2e; border-radius:10px; padding:1rem; }
    .section-header { font-size:1.1rem; font-weight:600; color:#1a1a2e; margin-top:1.5rem; margin-bottom:0.5rem; }
    div[data-testid="stMetric"] { background:#f8f9ff; border-radius:10px; padding:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── Feature Engineering (exactly as notebook) ───────────────────────────────
def engineer_features(df):
    df = df.copy()
    services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['TotalServices'] = df[services].apply(
        lambda r: sum(1 for v in r if v not in ['No','No internet service','No phone service']), axis=1)
    df['AvgMonthlySpend'] = df.apply(
        lambda r: r['TotalCharges']/r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
    df['ContractTypeScore'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
    df['Fiber_NoSecurity']  = ((df['InternetService']=='Fiber optic') & (df['OnlineSecurity']=='No')).astype(int)
    df['Fiber_NoTech']      = ((df['InternetService']=='Fiber optic') & (df['TechSupport']=='No')).astype(int)
    df['MtM_LowTenure']     = ((df['Contract']=='Month-to-month') & (df['tenure']<=12)).astype(int)
    df['MtM_HighCharges']   = ((df['Contract']=='Month-to-month') & (df['MonthlyCharges']>70)).astype(int)
    df['Senior_NoSupport']  = ((df['SeniorCitizen']==1) & (df['TechSupport']=='No')).astype(int)
    df['ElecCheck_MtM']     = ((df['PaymentMethod']=='Electronic check') & (df['Contract']=='Month-to-month')).astype(int)
    df['NoFamily_MtM']      = ((df['Partner']=='No') & (df['Dependents']=='No') & (df['Contract']=='Month-to-month')).astype(int)
    df['ChargesPerService']  = df['MonthlyCharges'] / (df['TotalServices'] + 1)
    df['TenureChargeRatio']  = df['tenure'] / (df['MonthlyCharges'] + 1)
    df['ChargesTenureIndex'] = df['MonthlyCharges'] * np.log1p(df['tenure'])
    def risk(row):
        s = 0
        if row['Contract']       == 'Month-to-month':  s += 3
        if row['tenure']         <= 12:                 s += 2
        if row['InternetService']== 'Fiber optic':      s += 2
        if row['TechSupport']    == 'No':               s += 1
        if row['OnlineSecurity'] == 'No':               s += 1
        if row['PaymentMethod']  == 'Electronic check': s += 1
        if row['MonthlyCharges'] > 80:                  s += 1
        if row['SeniorCitizen']  == 1:                  s += 1
        return s
    df['RiskScore']   = df.apply(risk, axis=1)
    df['LoyaltyScore']= ((df['tenure']>24).astype(int)
                        + (df['Contract']=='Two year').astype(int)*2
                        + (df['TotalServices']>=4).astype(int)
                        + (df['Partner']=='Yes').astype(int)
                        + (df['Dependents']=='Yes').astype(int))
    df['RiskLoyaltyRatio']      = df['RiskScore'] / (df['LoyaltyScore'] + 1)
    df['tenure_sq']              = df['tenure'] ** 2
    df['tenure_monthly']         = df['tenure'] * df['MonthlyCharges']
    df['log_total']              = np.log1p(df['TotalCharges'])
    df['is_new']                 = (df['tenure'] <= 6).astype(int)
    df['is_veteran']             = (df['tenure'] >= 48).astype(int)
    df['no_protection']          = ((df['OnlineSecurity']=='No') & (df['TechSupport']=='No') & (df['OnlineBackup']=='No')).astype(int)
    df['full_protection']        = ((df['OnlineSecurity']=='Yes') & (df['TechSupport']=='Yes')).astype(int)
    df['low_tenure_high_charge'] = ((df['tenure']<=12) & (df['MonthlyCharges']>70)).astype(int)
    return df

def encode_features(df):
    df = df.copy()
    binary_maps = {
        'gender'          : {'Female': 0, 'Male': 1},
        'Partner'         : {'No': 0, 'Yes': 1},
        'Dependents'      : {'No': 0, 'Yes': 1},
        'PhoneService'    : {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
    }
    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    if 'TenureGroup' in df.columns:
        df.drop('TenureGroup', axis=1, inplace=True)
    return df

# ─── Load Trained Model (from notebook export — no retraining here) ─────────
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """
    Loads the model artifact produced by the notebook's
    '4.5 Export Best Model for Deployment' cell. That cell trains the
    Voting Ensemble (LightGBM + CatBoost + XGBoost) — the best-performing
    model in the notebook's comparison table — on the full training set
    and pickles it together with its scaler, feature list and metrics.

    The app NEVER retrains a model; it only loads what the notebook produced,
    so the numbers shown here always match the notebook's results exactly.
    """
    pkl_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact

artifact = load_model()

if artifact is not None:
    model          = artifact['model']
    scaler         = artifact['scaler']
    feature_names  = artifact['feature_names']
    metrics        = artifact['metrics']
    feat_importance = artifact.get('feature_importance')
    model_name     = artifact.get('model_name', 'Voting Ensemble')
    threshold      = artifact.get('threshold', 0.5)
else:
    model = scaler = feature_names = metrics = feat_importance = None
    model_name, threshold = 'Voting Ensemble', 0.5

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard AI")
    st.markdown("**ML CEP Project**")
    st.markdown("Course: SE-CD-638")
    st.markdown("Instructor: Dr. Aamir Arsalan")
    st.divider()
    page = st.radio("Navigate", ["🏠 Home", "🔮 Predict Churn", "📊 Model Performance", "📁 Batch Prediction"])
    st.divider()
    st.caption(f"Model: {model_name} | Dataset: Telco Customer Churn")

# ════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🛡️ ChurnGuard AI v5.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Customer Churn Prediction & Retention Intelligence System</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ `churn_model.pkl` not found! Run the notebook's "
                 "**'4.5 Export Best Model for Deployment'** cell first, then "
                 "copy `churn_model.pkl` into this app's folder.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("F1 Score",       f"{metrics['f1']:.4f}")
        c3.metric("ROC-AUC",        f"{metrics['roc_auc']:.4f}")

    st.divider()
    st.markdown("### 📋 Project Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Phase 1 — Preprocessing**
        - Missing value treatment (TotalCharges)
        - Winsorization (IQR-based outlier handling)
        - Feature Engineering (30+ features)
        - SMOTE for class imbalance

        **Phase 2 — Models**
        - 10 Supervised models compared
        - Stacking + Soft Voting Ensemble
        - K-Means, Agglomerative, DBSCAN clustering
        """)
    with col2:
        st.markdown("""
        **Phase 3 — Evaluation**
        - Confusion Matrix, ROC-AUC
        - Precision, Recall, F1 per class
        - Feature Importance (averaged across base models)

        **Phase 4 — Tuning**
        - GridSearchCV (Random Forest)
        - RandomizedSearchCV (LightGBM)
        - 5-Fold Stratified Cross-Validation
        - Deployed model: **Voting Ensemble** (best test accuracy)
        """)

    st.divider()
    st.info("👈 Use the sidebar to select **Predict Churn** for a single customer prediction!")

# ════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.markdown("## 🔮 Single Customer Churn Prediction")
    st.markdown("Fill in the customer details below to view the churn probability.")

    if model is None:
        st.error("⚠️ Model not loaded — `churn_model.pkl` is missing.")
        st.stop()

    with st.form("predict_form"):
        st.markdown('<div class="section-header">👤 Customer Info</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender        = st.selectbox("Gender", ["Male","Female"])
            senior        = st.selectbox("Senior Citizen", ["No","Yes"])
            partner       = st.selectbox("Partner", ["Yes","No"])
            dependents    = st.selectbox("Dependents", ["Yes","No"])
        with c2:
            tenure        = st.slider("Tenure (months)", 0, 72, 12)
            monthly       = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total         = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure*monthly))
        with c3:
            contract      = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            payment       = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            paperless     = st.selectbox("Paperless Billing", ["Yes","No"])

        st.markdown('<div class="section-header">📡 Services</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            phone         = st.selectbox("Phone Service", ["Yes","No"])
            multiple      = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        with c2:
            internet      = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
            online_sec    = st.selectbox("Online Security", ["No","Yes","No internet service"])
        with c3:
            online_bkp    = st.selectbox("Online Backup", ["No","Yes","No internet service"])
            device        = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        with c4:
            tech          = st.selectbox("Tech Support", ["No","Yes","No internet service"])
            tv            = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
            movies        = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        input_dict = {
            'gender': gender, 'SeniorCitizen': 1 if senior=="Yes" else 0,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple,
            'InternetService': internet, 'OnlineSecurity': online_sec,
            'OnlineBackup': online_bkp, 'DeviceProtection': device,
            'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
        }
        row = pd.DataFrame([input_dict])

        # Winsorize (same bounds as training — approximate)
        row['tenure']         = row['tenure'].clip(0, 72)
        row['MonthlyCharges'] = row['MonthlyCharges'].clip(18, 120)
        row['TotalCharges']   = row['TotalCharges'].clip(0, 8685)

        row = engineer_features(row)
        row = encode_features(row)

        # Align columns to training features
        for col in feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_names]

        row_s   = scaler.transform(row)
        prob    = model.predict_proba(row_s)[0][1]
        pred    = int(prob >= threshold)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Prediction", "🔴 Will Churn" if pred else "🟢 Will Stay")
        c3.metric("Risk Level", "HIGH" if prob>0.65 else ("MEDIUM" if prob>0.35 else "LOW"))

        if prob > 0.65:
            st.markdown(f"""<div class="risk-high">
            <b>🔴 HIGH CHURN RISK ({prob*100:.1f}%)</b><br>
            This customer has a high risk of churning. Immediate retention action is recommended.
            <br><br><b>Suggested Actions:</b> Offer a discount • Propose a contract upgrade • Personal outreach
            </div>""", unsafe_allow_html=True)
        elif prob > 0.35:
            st.markdown(f"""<div class="risk-medium">
            <b>🟡 MEDIUM CHURN RISK ({prob*100:.1f}%)</b><br>
            This customer is in the at-risk zone. Proactive engagement can help retain them.
            <br><br><b>Suggested Actions:</b> Loyalty reward • Service upgrade offer • Satisfaction survey
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
            <b>🟢 LOW CHURN RISK ({prob*100:.1f}%)</b><br>
            This customer is loyal — a great opportunity for upselling!
            <br><br><b>Suggested Actions:</b> Premium features offer • Referral program • Long-term contract
            </div>""", unsafe_allow_html=True)

        # Risk factors breakdown
        st.markdown("#### 📊 Key Risk Factors")
        factors = []
        if contract == "Month-to-month":    factors.append(("Contract Type", "Month-to-month — highest churn risk", "🔴"))
        if tenure <= 12:                    factors.append(("Tenure",        f"Only {tenure} months — new customer", "🔴"))
        if internet == "Fiber optic":       factors.append(("Internet",      "Fiber optic users churn more", "🟡"))
        if tech == "No":                    factors.append(("Tech Support",  "No tech support — dissatisfaction risk", "🟡"))
        if online_sec == "No":              factors.append(("Security",      "No online security subscribed", "🟡"))
        if payment == "Electronic check":   factors.append(("Payment",       "Electronic check — correlated with churn", "🟡"))
        if monthly > 80:                    factors.append(("Monthly Bill",  f"High charges ${monthly:.0f}/mo", "🔴"))
        if senior == "Yes" and tech == "No":factors.append(("Senior+NoSupport","Senior citizen without tech support","🔴"))
        if not factors:
            factors.append(("Profile", "No major risk factors detected", "🟢"))
        for name, reason, icon in factors:
            st.markdown(f"{icon} **{name}:** {reason}")

# ════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown(f"## 📊 Model Performance — {model_name}")

    if model is None:
        st.error("⚠️ Model not loaded — `churn_model.pkl` is missing.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("F1 Score",      f"{metrics['f1']:.4f}")
    c3.metric("ROC-AUC",       f"{metrics['roc_auc']:.4f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn','Churn'],
                    yticklabels=['No Churn','Churn'], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Feature Importance (Top 15)")
        if feat_importance is not None:
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feat_importance})
            fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.barh(fi_df['Feature'], fi_df['Importance'], color='#4f46e5')
            ax.set_xlabel("Importance (averaged across base models)")
            ax.set_title("Top 15 Features")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info("Feature importance was not found in the saved model artifact.")

    st.markdown("#### Classification Report")
    st.code(metrics['report'])

    st.markdown("#### Model Configuration")
    st.json({
        "model"            : model_name,
        "base_estimators"  : ["LightGBM", "CatBoost", "XGBoost"],
        "voting"           : "soft",
        "weights"          : [2, 2, 1],
        "decision_threshold": threshold,
        "class_imbalance"  : "SMOTE (k=5)",
        "train_test_split" : "80/20",
        "trained_on"       : "Full training set (post-SMOTE)"
    })

# ════════════════════════════════════════════════════════════════════
# PAGE 4: BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Churn Prediction")
    st.markdown("Upload a CSV file to predict churn for multiple customers at once.")

    if model is None:
        st.error("⚠️ Model not loaded — `churn_model.pkl` is missing.")
        st.stop()

    st.info("📌 The CSV file must contain the following columns: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)

        if 'customerID' in df_up.columns:
            cust_ids = df_up['customerID'].copy()
            df_up.drop('customerID', axis=1, inplace=True)
        else:
            cust_ids = pd.Series([f"Customer_{i+1}" for i in range(len(df_up))])

        if 'Churn' in df_up.columns:
            df_up.drop('Churn', axis=1, inplace=True)

        df_up['TotalCharges'] = pd.to_numeric(df_up['TotalCharges'], errors='coerce').fillna(0)
        df_up['tenure']         = df_up['tenure'].clip(0, 72)
        df_up['MonthlyCharges'] = df_up['MonthlyCharges'].clip(18, 120)
        df_up['TotalCharges']   = df_up['TotalCharges'].clip(0, 8685)

        df_feat = engineer_features(df_up)
        df_enc  = encode_features(df_feat)

        for col in feature_names:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[feature_names]

        X_s   = scaler.transform(df_enc)
        probs = model.predict_proba(X_s)[:,1]
        preds = (probs >= threshold).astype(int)

        result = pd.DataFrame({
            'CustomerID'        : cust_ids.values,
            'Churn_Probability' : (probs*100).round(1),
            'Prediction'        : ['Churn' if p else 'No Churn' for p in preds],
            'Risk_Level'        : ['HIGH' if p>0.65 else ('MEDIUM' if p>0.35 else 'LOW') for p in probs]
        })

        st.success(f"✅ {len(result)} customers processed successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", len(result))
        c2.metric("Predicted Churn", int(preds.sum()))
        c3.metric("Churn Rate",      f"{preds.mean()*100:.1f}%")

        st.dataframe(result, use_container_width=True)

        csv_out = result.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv_out,
                           "churn_predictions.csv", "text/csv")
        
        
