"""
╔══════════════════════════════════════════════════════════════╗
║       ChurnGuard AI — Customer Churn Prediction System       ║
║   Course: Machine Learning (SE-CD-638) | Dr. Aamir Arsalan   ║
║   Run:  streamlit run streamlit_app.py                       ║
║   Needs: WA_Fn-UseC_-Telco-Customer-Churn.csv same folder    ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink:      #0B1120;
    --ink2:     #1A2540;
    --card:     #111827;
    --card2:    #192035;
    --border:   #1E2D4A;
    --border2:  #253557;
    --accent:   #3B82F6;
    --cyan:     #06B6D4;
    --green:    #10B981;
    --amber:    #F59E0B;
    --red:      #EF4444;
    --text:     #E2E8F0;
    --muted:    #64748B;
    --dim:      #334155;
    --r:        14px;
}

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background-color: var(--ink) !important;
    color: var(--text) !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--ink2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem; }
[data-testid="stSidebar"] label { color: var(--muted) !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider > div { color: var(--text); }

/* ─── Main area ─── */
.main .block-container { padding: 1.5rem 2rem; max-width: 100%; }

/* ─── Streamlit widget overrides ─── */
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: var(--accent) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--cyan) 100%) !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(59,130,246,0.5) !important;
    opacity: 0.95 !important;
}

/* ─── Hero ─── */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #0F172A 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 30%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.06em;
}

/* ─── Stat cards ─── */
.stat-row { display: flex; gap: 14px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 110px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.blue::before  { background: linear-gradient(90deg, var(--accent), var(--cyan)); }
.stat-card.green::before { background: linear-gradient(90deg, var(--green), #34D399); }
.stat-card.amber::before { background: linear-gradient(90deg, var(--amber), #FCD34D); }
.stat-card.cyan::before  { background: linear-gradient(90deg, var(--cyan), var(--accent)); }
.stat-card.red::before   { background: linear-gradient(90deg, var(--red), #F87171); }
.stat-lbl { font-size: 0.7rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text); line-height: 1; }
.stat-val.blue  { color: #60A5FA; }
.stat-val.green { color: #34D399; }
.stat-val.amber { color: #FCD34D; }
.stat-val.cyan  { color: #22D3EE; }

/* ─── Section headings ─── */
.sec-head {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
}

/* ─── Result cards ─── */
.res-card {
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.res-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.05) 100%);
    border: 1.5px solid rgba(239,68,68,0.45);
}
.res-medium {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(245,158,11,0.05) 100%);
    border: 1.5px solid rgba(245,158,11,0.45);
}
.res-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.05) 100%);
    border: 1.5px solid rgba(16,185,129,0.45);
}
.res-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}
.badge-high   { background: rgba(239,68,68,0.2);   color: #FCA5A5; border: 1px solid rgba(239,68,68,0.4); }
.badge-medium { background: rgba(245,158,11,0.2);  color: #FDE68A; border: 1px solid rgba(245,158,11,0.4); }
.badge-low    { background: rgba(16,185,129,0.2);  color: #6EE7B7; border: 1px solid rgba(16,185,129,0.4); }

.res-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.pct-high   { color: #F87171; }
.pct-medium { color: #FCD34D; }
.pct-low    { color: #34D399; }

.res-urgency {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ─── Progress bar ─── */
.pb-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    margin: 1rem 0 0.3rem 0;
    overflow: hidden;
}
.pb-fill { height: 100%; border-radius: 999px; }
.pb-high   { background: linear-gradient(90deg, #F87171, #EF4444); }
.pb-medium { background: linear-gradient(90deg, #FCD34D, #F59E0B); }
.pb-low    { background: linear-gradient(90deg, #6EE7B7, #10B981); }

/* ─── Score tiles ─── */
.score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 1rem 0; }
.score-tile {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
}
.score-num.s-risk   { color: #F87171; }
.score-num.s-loyal  { color: #34D399; }
.score-num.s-svc    { color: #60A5FA; }
.score-num.s-ratio  { color: #FCD34D; }
.score-lbl { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

/* ─── Risk factor chips ─── */
.chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 0.6rem 0 1rem 0; }
.chip {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    border: 1px solid;
}
.chip-red    { background: rgba(239,68,68,0.12);   color: #FCA5A5; border-color: rgba(239,68,68,0.35); }
.chip-amber  { background: rgba(245,158,11,0.12);  color: #FDE68A; border-color: rgba(245,158,11,0.35); }
.chip-blue   { background: rgba(59,130,246,0.12);  color: #93C5FD; border-color: rgba(59,130,246,0.35); }

/* ─── Strategy items ─── */
.strat-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.strat-item:hover { border-color: var(--border2); }
.strat-item.s-high   { border-left: 3px solid #EF4444; }
.strat-item.s-medium { border-left: 3px solid #F59E0B; }
.strat-item.s-low    { border-left: 3px solid #10B981; }
.strat-icon { font-size: 1.15rem; flex-shrink: 0; margin-top: 1px; }
.strat-body { flex: 1; }
.strat-priority {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 4px;
}
.p-high   { background: rgba(239,68,68,0.2);  color: #FCA5A5; }
.p-medium { background: rgba(245,158,11,0.2); color: #FDE68A; }
.p-low    { background: rgba(16,185,129,0.2); color: #6EE7B7; }
.strat-text { font-size: 0.875rem; color: var(--text); line-height: 1.5; font-weight: 400; }

/* ─── Guide table ─── */
.guide-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
    margin-top: 0.8rem;
}
.guide-wrap table { width: 100%; border-collapse: collapse; }
.guide-wrap th {
    background: var(--card2);
    color: var(--muted);
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.guide-wrap td {
    padding: 10px 14px;
    font-size: 0.82rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
}
.guide-wrap tr:last-child td { border-bottom: none; }
.guide-wrap tr:hover td { background: var(--card2); }
.t-high   { color: #FCA5A5; font-weight: 700; }
.t-medium { color: #FDE68A; font-weight: 700; }
.t-low    { color: #6EE7B7; font-weight: 700; }

/* ─── Sidebar label fix ─── */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    color: #3B82F6;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ─── Info box ─── */
.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    color: #93C5FD;
    font-size: 0.875rem;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}

/* ─── Bottom bar ─── */
.bottom-bar {
    margin-top: 2rem;
    padding: 1rem 1.5rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.bb-item { text-align: center; }
.bb-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.bb-val   { font-size: 0.85rem; color: var(--text); font-weight: 600; margin-top: 2px; }

/* ─── Matplotlib charts ─── */
.stPlotlyChart, .stPyplot { background: transparent !important; }

/* ─── Hide defaults ─── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODEL (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df['TenureGroup'] = df['tenure'].apply(lambda t:
        '0-12' if t<=12 else '12-24' if t<=24 else '24-36' if t<=36 else
        '36-48' if t<=48 else '48-60' if t<=60 else '60+')
    svcs = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['TotalServices'] = df[svcs].apply(
        lambda r: sum(1 for v in r if v not in ['No','No internet service','No phone service']), axis=1)
    df['AvgMonthlySpend'] = df.apply(
        lambda r: r['TotalCharges']/r['tenure'] if r['tenure']>0 else r['MonthlyCharges'], axis=1)
    df['ContractTypeScore'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})

    for col in ['MonthlyCharges','TotalCharges']:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(lower=Q1-1.5*(Q3-Q1), upper=Q3+1.5*(Q3-Q1))

    le = LabelEncoder()
    for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
        df[col] = le.fit_transform(df[col])

    mc = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
          'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod','TenureGroup']
    df_enc = pd.get_dummies(df, columns=mc, drop_first=True)

    X = df_enc.drop('Churn', axis=1)
    y = df_enc['Churn']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    mdl = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        random_state=42, eval_metric='logloss')
    mdl.fit(Xtr_s, ytr)
    preds = mdl.predict(Xte_s)
    probs = mdl.predict_proba(Xte_s)[:,1]

    return (mdl, sc, list(X.columns),
            accuracy_score(yte,preds), f1_score(yte,preds), roc_auc_score(yte,probs),
            yte.values, probs)

with st.spinner("🔄 Training XGBoost model on Telco dataset..."):
    model, scaler, feature_cols, acc, f1, auc, y_test, test_probs = load_model()


# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDER (bug-free exact matching)
# ══════════════════════════════════════════════════════════════════
def build_features(gender, senior, partner, dependents, tenure,
                   phone, multi_lines, internet,
                   security, backup, device, tech, stv, smov,
                   contract, billing, payment, monthly):
    total = monthly * tenure
    inp = {col: 0 for col in feature_cols}
    inp['gender']           = 1 if gender=='Male' else 0
    inp['SeniorCitizen']    = 1 if senior=='Yes' else 0
    inp['Partner']          = 1 if partner=='Yes' else 0
    inp['Dependents']       = 1 if dependents=='Yes' else 0
    inp['tenure']           = tenure
    inp['PhoneService']     = 1 if phone=='Yes' else 0
    inp['PaperlessBilling'] = 1 if billing=='Yes' else 0
    inp['MonthlyCharges']   = monthly
    inp['TotalCharges']     = total
    all_svcs = [phone, multi_lines, internet, security, backup, device, tech, stv, smov]
    inp['TotalServices']     = sum(1 for v in all_svcs if v not in ['No','No internet service','No phone service'])
    inp['AvgMonthlySpend']   = total/tenure if tenure>0 else monthly
    inp['ContractTypeScore'] = {'Month-to-month':0,'One year':1,'Two year':2}[contract]
    def oh(col):
        if col in inp: inp[col] = 1
    oh(f'MultipleLines_{multi_lines}')
    oh(f'InternetService_{internet}')
    oh(f'OnlineSecurity_{security}')
    oh(f'OnlineBackup_{backup}')
    oh(f'DeviceProtection_{device}')
    oh(f'TechSupport_{tech}')
    oh(f'StreamingTV_{stv}')
    oh(f'StreamingMovies_{smov}')
    oh(f'Contract_{contract}')
    oh(f'PaymentMethod_{payment}')
    tg = ('0-12' if tenure<=12 else '12-24' if tenure<=24 else '24-36' if tenure<=36
          else '36-48' if tenure<=48 else '48-60' if tenure<=60 else '60+')
    oh(f'TenureGroup_{tg}')
    return pd.DataFrame([inp])[feature_cols]


# ══════════════════════════════════════════════════════════════════
# SCORES
# ══════════════════════════════════════════════════════════════════
def compute_scores(contract, tenure, internet, tech, security, monthly, payment, senior, backup, device):
    risk = 0
    if contract=='Month-to-month': risk += 3
    if tenure<=12: risk += 2
    if internet=='Fiber optic': risk += 2
    if tech=='No' and internet!='No': risk += 1
    if security=='No' and internet!='No': risk += 1
    if payment=='Electronic check': risk += 1
    if monthly>80: risk += 1
    if senior=='Yes': risk += 1

    loyal = 0
    if tenure>24: loyal += 1
    if contract=='Two year': loyal += 2
    if backup=='Yes': loyal += 1
    if device=='Yes': loyal += 1

    return risk, loyal, round(risk/(loyal+1), 2)


# ══════════════════════════════════════════════════════════════════
# RETENTION STRATEGIES (fully differentiated)
# ══════════════════════════════════════════════════════════════════
def get_strategies(prob, contract, tenure, tech, security, monthly, internet,
                   payment, backup, device, stv, smov, multi_lines, senior, partner, dependents):
    s = []
    cl = 'high' if prob>0.6 else 'medium' if prob>0.35 else 'low'

    if prob > 0.6:
        if contract=='Month-to-month':
            s.append(('🚨','high','URGENT: Offer 25% discount to switch to annual contract — contract type is #1 churn driver'))
        if tenure<6:
            s.append(('👤','high','Assign dedicated Customer Success Manager — first 90 days are critical onboarding window'))
        if internet=='Fiber optic' and tech=='No':
            s.append(('🔧','high','Dispatch fiber quality check + activate FREE Tech Support for 6 months — fiber + no support = 55% churn'))
        if monthly>80:
            s.append(('💰','high','Apply immediate 20% loyalty discount — high charges + no commitment is the highest-risk combination'))
        if payment=='Electronic check':
            s.append(('💳','high','Offer $10/month credit for switching to auto-pay — electronic check users churn 2× more'))
        if security=='No' and internet!='No':
            s.append(('🔐','high','Bundle Online Security FREE for 3 months — removes a key churn trigger immediately'))

    elif prob > 0.35:
        if contract=='Month-to-month':
            s.append(('📋','medium','Offer 15% loyalty discount for upgrading to 1-year contract plan'))
        if 6<=tenure<=24:
            s.append(('🎁','medium','Send a loyalty appreciation gift — customers in months 6-24 are at elevated risk of leaving'))
        if security=='No' and internet!='No':
            s.append(('🔐','medium','Offer Online Security bundle at 50% off for 6 months'))
        if backup=='No' and internet!='No':
            s.append(('☁️','medium','Introduce Online Backup at introductory price for first 3 months'))
        if stv!='Yes' and smov!='Yes' and internet!='No':
            s.append(('📺','medium','Promote Streaming bundle (TV + Movies) with 30-day free trial'))
        if payment in ['Electronic check','Mailed check']:
            s.append(('📱','medium','Offer $5/month savings for switching to digital auto-payment method'))

    else:
        s.append(('⭐','low','Enroll in VIP Loyalty Rewards Program — this customer is highly satisfied'))
        if tenure>=36:
            s.append(('🏆','low','Send personalized anniversary reward to celebrate customer loyalty milestone'))
        if multi_lines=='Yes' or (stv=='Yes' and smov=='Yes'):
            s.append(('💎','low','Offer exclusive multi-service discount bundle to reinforce long-term value'))
        if partner=='Yes' or dependents=='Yes':
            s.append(('👨‍👩‍👧','low','Promote Family Plan upgrade — household customers have lower churn and higher LTV'))
        s.append(('📊','low','Schedule quarterly satisfaction check-in call to proactively address any concerns'))
        if contract=='Two year':
            s.append(('🎖️','low','Recognize contract commitment with a surprise loyalty bonus credit this billing cycle'))

    if senior=='Yes' and len(s)<6:
        s.append(('👴', cl,'Assign Senior Care specialist — personalized support significantly reduces senior churn'))

    return s[:6]


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-size:1.6rem;font-weight:800;color:#E2E8F0;margin-bottom:2px;">🛡️ ChurnGuard</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#64748B;letter-spacing:0.08em;margin-bottom:1rem;">AI CUSTOMER RETENTION PLATFORM</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">👤 Demographics</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",         ["Male","Female"], key="g")
    senior     = st.selectbox("Senior Citizen", ["No","Yes"],      key="sc")
    partner    = st.selectbox("Partner",        ["Yes","No"],      key="p")
    dependents = st.selectbox("Dependents",     ["Yes","No"],      key="d")

    st.markdown('<div class="sidebar-section">📄 Account</div>', unsafe_allow_html=True)
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract",  ["Month-to-month","One year","Two year"])
    billing  = st.selectbox("Paperless Billing", ["Yes","No"])
    payment  = st.selectbox("Payment Method", [
        "Electronic check","Mailed check",
        "Bank transfer (automatic)","Credit card (automatic)"])
    monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)

    st.markdown('<div class="sidebar-section">📞 Phone</div>', unsafe_allow_html=True)
    phone      = st.selectbox("Phone Service",  ["Yes","No"])
    multi_lines= st.selectbox("Multiple Lines", ["No","Yes","No phone service"])

    st.markdown('<div class="sidebar-section">🌐 Internet & Add-ons</div>', unsafe_allow_html=True)
    internet = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
    if internet != 'No':
        security = st.selectbox("Online Security",   ["No","Yes"])
        backup   = st.selectbox("Online Backup",     ["No","Yes"])
        device   = st.selectbox("Device Protection", ["No","Yes"])
        tech     = st.selectbox("Tech Support",      ["No","Yes"])
        stv      = st.selectbox("Streaming TV",      ["No","Yes"])
        smov     = st.selectbox("Streaming Movies",  ["No","Yes"])
    else:
        security=backup=device=tech=stv=smov="No internet service"

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Churn Risk", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════

# Hero
st.markdown(f"""
<div class="hero">
    <div class="hero-title">🛡️ ChurnGuard AI</div>
    <div class="hero-sub">AI-Powered Customer Churn Prediction &amp; Retention Intelligence &nbsp;·&nbsp; SE-CD-638 &nbsp;·&nbsp; Dr. Aamir Arsalan</div>
</div>
""", unsafe_allow_html=True)

# Model metric row
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card blue">
        <div class="stat-lbl">Model</div>
        <div class="stat-val" style="font-size:1rem;color:#93C5FD;margin-top:4px;">XGBoost Tuned</div>
    </div>
    <div class="stat-card blue">
        <div class="stat-lbl">Accuracy</div>
        <div class="stat-val blue">{acc:.1%}</div>
    </div>
    <div class="stat-card green">
        <div class="stat-lbl">F1-Score</div>
        <div class="stat-val green">{f1:.1%}</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-lbl">ROC-AUC</div>
        <div class="stat-val amber">{auc:.1%}</div>
    </div>
    <div class="stat-card cyan">
        <div class="stat-lbl">Dataset</div>
        <div class="stat-val cyan">7,043</div>
    </div>
    <div class="stat-card blue">
        <div class="stat-lbl">Features</div>
        <div class="stat-val blue">38</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── PREDICTION ─────────────────────────────────────────────────────
if predict_btn:
    df_inp = build_features(gender, senior, partner, dependents, tenure,
                            phone, multi_lines, internet,
                            security, backup, device, tech, stv, smov,
                            contract, billing, payment, monthly)
    scaled = scaler.transform(df_inp)
    prob   = model.predict_proba(scaled)[0][1]
    pct    = prob * 100
    cl     = 'high' if prob>0.6 else 'medium' if prob>0.35 else 'low'
    risk_label = {'high':'⚠ HIGH RISK','medium':'◐ MODERATE RISK','low':'✓ LOW RISK'}[cl]
    urgency    = {'high':'🚨 IMMEDIATE ACTION REQUIRED','medium':'⚠ FOLLOW-UP RECOMMENDED','low':'✅ MONITOR — CUSTOMER IS STABLE'}[cl]

    risk_score, loyal_score, rl_ratio = compute_scores(
        contract, tenure, internet, tech, security, monthly, payment, senior, backup, device)
    strategies = get_strategies(prob, contract, tenure, tech, security, monthly, internet,
                                payment, backup, device, stv, smov, multi_lines, senior, partner, dependents)

    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        # Result card
        st.markdown(f"""
        <div class="res-card res-{cl}">
            <div class="res-badge badge-{cl}">{risk_label}</div>
            <div class="res-pct pct-{cl}">{pct:.1f}%</div>
            <div style="font-size:0.82rem;color:#64748B;margin-top:4px;">Churn Probability Score</div>
            <div class="pb-wrap">
                <div class="pb-fill pb-{cl}" style="width:{min(pct,100):.1f}%"></div>
            </div>
            <div class="res-urgency">{urgency}</div>
            <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);">
                <span style="font-size:0.75rem;color:#64748B;">Segment &nbsp;→&nbsp;</span>
                <span style="font-size:0.85rem;font-weight:700;color:{'#F87171' if cl=='high' else '#FCD34D' if cl=='medium' else '#34D399'};">
                    {'🔴 High-Risk' if cl=='high' else '🟡 Moderate-Risk' if cl=='medium' else '🟢 Loyal / Low-Risk'}
                </span>
                &nbsp;&nbsp;
                <span style="font-size:0.75rem;color:#64748B;">Revenue &nbsp;→&nbsp;</span>
                <span style="font-size:0.85rem;font-weight:600;color:#93C5FD;">${monthly:.0f}/mo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score tiles
        st.markdown('<div class="sec-head">📊 Customer Scores</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-grid">
            <div class="score-tile">
                <div class="score-num s-risk">{risk_score}</div>
                <div class="score-lbl">Risk Score</div>
            </div>
            <div class="score-tile">
                <div class="score-num s-loyal">{loyal_score}</div>
                <div class="score-lbl">Loyalty Score</div>
            </div>
            <div class="score-tile">
                <div class="score-num s-svc">{df_inp['TotalServices'].iloc[0]:.0f}</div>
                <div class="score-lbl">Services Used</div>
            </div>
            <div class="score-tile">
                <div class="score-num s-ratio">{rl_ratio}</div>
                <div class="score-lbl">Risk/Loyalty Ratio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk factor chips
        factors = []
        if contract=='Month-to-month': factors.append(('chip-red','Month-to-month'))
        if tenure<=12:                 factors.append(('chip-red',f'New ({tenure}mo)'))
        if internet=='Fiber optic':    factors.append(('chip-red','Fiber Optic'))
        if tech=='No' and internet!='No':     factors.append(('chip-amber','No Tech Support'))
        if security=='No' and internet!='No': factors.append(('chip-amber','No Security'))
        if payment=='Electronic check': factors.append(('chip-amber','E-Check'))
        if monthly>80:                 factors.append(('chip-amber',f'${monthly:.0f}/mo'))
        if senior=='Yes':              factors.append(('chip-blue','Senior'))
        if partner!='Yes' and dependents!='Yes': factors.append(('chip-blue','No Family Plan'))

        if factors:
            st.markdown('<div class="sec-head" style="margin-top:1rem;">🔍 Risk Factors</div>', unsafe_allow_html=True)
            chips = " ".join([f'<span class="chip {c}">{t}</span>' for c,t in factors])
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

        # Donut chart
        st.markdown('<div class="sec-head" style="margin-top:1rem;">📈 Probability Gauge</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.5), subplot_kw=dict(aspect='equal'))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        clr = {'high':'#EF4444','medium':'#F59E0B','low':'#10B981'}[cl]
        ax.pie([pct,100-pct], startangle=90,
               colors=[clr,'#1E2D4A'],
               wedgeprops=dict(width=0.42, edgecolor='#0B1120', linewidth=2))
        ax.text(0, 0.08, f"{pct:.1f}%", ha='center', va='center',
                fontsize=21, fontweight='bold', color=clr,
                fontfamily='monospace')
        ax.text(0, -0.28, {'high':'HIGH RISK','medium':'MOD. RISK','low':'LOW RISK'}[cl],
                ha='center', va='center', fontsize=9, color='#64748B', fontweight='600')
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        # Strategies
        st.markdown('<div class="sec-head">💡 Personalized Retention Strategies</div>', unsafe_allow_html=True)
        for icon, priority, text in strategies:
            st.markdown(f"""
            <div class="strat-item s-{priority}">
                <div class="strat-icon">{icon}</div>
                <div class="strat-body">
                    <span class="strat-priority p-{priority}">{priority.upper()}</span>
                    <div class="strat-text">{text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Population chart
        st.markdown('<div class="sec-head" style="margin-top:1.5rem;">📊 Where This Customer Falls</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')
        ax2.hist(test_probs[y_test==0], bins=40, alpha=0.65, color='#10B981', label='Non-Churners', density=True)
        ax2.hist(test_probs[y_test==1], bins=40, alpha=0.65, color='#EF4444', label='Churners', density=True)
        ax2.axvline(prob, color=clr, linewidth=2.5, linestyle='--', label=f'This Customer ({pct:.1f}%)', zorder=5)
        ax2.axvline(0.60, color='#EF4444', linewidth=1, linestyle=':', alpha=0.4)
        ax2.axvline(0.35, color='#F59E0B', linewidth=1, linestyle=':', alpha=0.4)
        for spine in ax2.spines.values(): spine.set_color('#1E2D4A')
        ax2.tick_params(colors='#64748B', labelsize=8)
        ax2.set_xlabel('Churn Probability Score', color='#64748B', fontsize=9)
        ax2.set_ylabel('Density', color='#64748B', fontsize=9)
        ax2.legend(fontsize=8, facecolor='#192035', edgecolor='#1E2D4A',
                   labelcolor='#E2E8F0', loc='upper right')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

# ── DEFAULT VIEW ─────────────────────────────────────────────────
else:
    st.markdown('<div class="info-box">👈 Fill in the customer profile in the sidebar and click <strong>🔮 Predict Churn Risk</strong> to get the full AI analysis.</div>', unsafe_allow_html=True)

    col_g, col_t = st.columns([1.1, 1], gap="large")

    with col_g:
        st.markdown('<div class="sec-head">🎯 Test Scenario Guide — Inputs to Try</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="guide-wrap">
        <table>
        <thead><tr><th>Risk Level</th><th>Contract</th><th>Tenure</th><th>Internet</th><th>Tech</th><th>Monthly $</th><th>Payment</th></tr></thead>
        <tbody>
        <tr><td class="t-high">🔴 HIGH (&gt;60%)</td><td>Month-to-month</td><td>1–5 mo</td><td>Fiber optic</td><td>No</td><td>$90–$120</td><td>Electronic check</td></tr>
        <tr><td class="t-high">🔴 HIGH (&gt;60%)</td><td>Month-to-month</td><td>6–12 mo</td><td>Fiber optic</td><td>No</td><td>$75–$100</td><td>Mailed check</td></tr>
        <tr><td class="t-medium">🟡 MEDIUM (35–60%)</td><td>Month-to-month</td><td>12–24 mo</td><td>DSL</td><td>No</td><td>$55–$75</td><td>Mailed check</td></tr>
        <tr><td class="t-medium">🟡 MEDIUM (35–60%)</td><td>One year</td><td>6–18 mo</td><td>Fiber optic</td><td>No</td><td>$70–$90</td><td>Electronic check</td></tr>
        <tr><td class="t-low">🟢 LOW (&lt;35%)</td><td>Two year</td><td>36+ mo</td><td>DSL</td><td>Yes</td><td>$55–$70</td><td>Bank transfer</td></tr>
        <tr><td class="t-low">🟢 LOW (&lt;35%)</td><td>Two year</td><td>60+ mo</td><td>DSL</td><td>Yes</td><td>$60–$80</td><td>Credit card</td></tr>
        </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Score distribution
        st.markdown('<div class="sec-head" style="margin-top:1.5rem;">📊 Model Score Distribution (Test Set)</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        fig3.patch.set_facecolor('#111827')
        ax3.set_facecolor('#111827')
        ax3.hist(test_probs[y_test==0], bins=40, alpha=0.65, color='#10B981', label='Non-Churners', density=True)
        ax3.hist(test_probs[y_test==1], bins=40, alpha=0.65, color='#EF4444', label='Churners', density=True)
        ax3.axvline(0.60, color='#EF4444', lw=1.5, ls=':', alpha=0.7, label='High threshold (0.60)')
        ax3.axvline(0.35, color='#F59E0B', lw=1.5, ls=':', alpha=0.7, label='Med threshold (0.35)')
        for spine in ax3.spines.values(): spine.set_color('#1E2D4A')
        ax3.tick_params(colors='#64748B', labelsize=8)
        ax3.set_xlabel('Churn Probability Score', color='#64748B', fontsize=9)
        ax3.set_ylabel('Density', color='#64748B', fontsize=9)
        ax3.legend(fontsize=8, facecolor='#192035', edgecolor='#1E2D4A', labelcolor='#E2E8F0')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with col_t:
        # Risk breakdown donut
        st.markdown('<div class="sec-head">📈 Test Set Risk Distribution</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(4.5, 3.8), subplot_kw=dict(aspect='equal'))
        fig4.patch.set_facecolor('#111827')
        ax4.set_facecolor('#111827')
        low_n  = (test_probs<0.35).sum()
        med_n  = ((test_probs>=0.35)&(test_probs<0.60)).sum()
        high_n = (test_probs>=0.60).sum()
        n = len(test_probs)
        ax4.pie([low_n,med_n,high_n],
                labels=[f'Low Risk\n{low_n/n*100:.1f}%', f'Moderate\n{med_n/n*100:.1f}%', f'High Risk\n{high_n/n*100:.1f}%'],
                colors=['#10B981','#F59E0B','#EF4444'],
                wedgeprops=dict(width=0.48, edgecolor='#0B1120', linewidth=2),
                textprops=dict(color='#94A3B8', fontsize=8.5))
        ax4.set_title('Risk Level Breakdown', color='#64748B', fontsize=9, pad=10)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        # Feature importance (top 10)
        st.markdown('<div class="sec-head" style="margin-top:1.2rem;">🔑 Top Churn Predictors</div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=feature_cols).nlargest(10)
        fig5, ax5 = plt.subplots(figsize=(4.5, 4))
        fig5.patch.set_facecolor('#111827')
        ax5.set_facecolor('#111827')
        colors_fi = ['#3B82F6','#06B6D4','#10B981','#F59E0B','#EF4444',
                     '#8B5CF6','#EC4899','#14B8A6','#F97316','#6366F1']
        bars = ax5.barh(range(len(fi)), fi.values, color=colors_fi[:len(fi)], alpha=0.85)
        ax5.set_yticks(range(len(fi)))
        ax5.set_yticklabels([c.replace('_',' ').replace(' No internet service','').replace('Contract ','')
                             .replace('PaymentMethod ','').replace('TenureGroup ','')
                             for c in fi.index], color='#94A3B8', fontsize=7.5)
        ax5.set_xlabel('Importance', color='#64748B', fontsize=8)
        for spine in ax5.spines.values(): spine.set_color('#1E2D4A')
        ax5.tick_params(colors='#64748B', labelsize=7.5)
        ax5.invert_yaxis()
        plt.tight_layout(pad=0.5)
        st.pyplot(fig5, use_container_width=True)
        plt.close()

# Bottom bar
st.markdown(f"""
<div class="bottom-bar">
    <div class="bb-item"><div class="bb-label">Model</div><div class="bb-val">XGBoost (Tuned)</div></div>
    <div class="bb-item"><div class="bb-label">Accuracy</div><div class="bb-val" style="color:#60A5FA;">{acc:.1%}</div></div>
    <div class="bb-item"><div class="bb-label">F1-Score</div><div class="bb-val" style="color:#34D399;">{f1:.1%}</div></div>
    <div class="bb-item"><div class="bb-label">ROC-AUC</div><div class="bb-val" style="color:#FCD34D;">{auc:.1%}</div></div>
    <div class="bb-item"><div class="bb-label">Dataset</div><div class="bb-val">Telco · 7,043 rows</div></div>
    <div class="bb-item"><div class="bb-label">Features</div><div class="bb-val">38 engineered</div></div>
    <div class="bb-item"><div class="bb-label">Course</div><div class="bb-val">SE-CD-638 · Dr. Aamir Arsalan</div></div>
</div>
""", unsafe_allow_html=True)
