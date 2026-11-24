"""Student Performance Prediction System
Decision Tree (CART/Gini) vs Gaussian Naive Bayes
Prototype for: Calaca City Global College thesis.
Run:  streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Analytics Dashboard", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# --- DATA & LOGIC ---
FEATURES = ["attendance", "quiz_avg", "exam_avg", "prev_grade"]
TARGET = "status"
CLASSES = ["Fail", "At-Risk", "Pass"]

def make_sample(n=800, seed=42):
    r = np.random.default_rng(seed)
    ability = r.normal(0, 1, n)
    df = pd.DataFrame({
        "attendance": np.clip(82 + 9 * ability + r.normal(0, 7, n), 30, 100),
        "quiz_avg": np.clip(76 + 10 * ability + r.normal(0, 6, n), 30, 100),
        "exam_avg": np.clip(75 + 11 * ability + r.normal(0, 7, n), 30, 100),
        "prev_grade": np.clip(80 + 8 * ability + r.normal(0, 5, n), 55, 100),
    })
    score = (0.2 * df.attendance + 0.25 * df.quiz_avg + 0.35 * df.exam_avg
             + 0.2 * df.prev_grade + r.normal(0, 2.5, n))
    df[TARGET] = np.where(score < 68, "Fail", np.where(score < 76, "At-Risk", "Pass"))
    return df.round(1)

def preprocess(df):
    log = {"rows_in": len(df)}
    df = df[FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title().replace({"At-risk": "At-Risk"})
    df = df[df[TARGET].isin(CLASSES)]
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log["missing_filled"] = int(df[FEATURES].isna().sum().sum())
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    
    # --- BINAGO: TINANGGAL ANG DROP_DUPLICATES PARA MAPANATILI ANG LAHAT NG RECORDS ---
    log["duplicates_removed"] = 0 
    
    clipped = 0
    for c in FEATURES:
        q1, q3 = df[c].quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        clipped += int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lo, hi)
    log["outliers_capped"] = clipped
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log

def build(name, scale, depth, leaf, alpha, seed):
    if name == "Decision Tree":
        model = DecisionTreeClassifier(criterion="gini", max_depth=depth,
                                       min_samples_leaf=leaf, ccp_alpha=alpha,
                                       random_state=seed)
    else:
        model = GaussianNB()
    return make_pipeline(StandardScaler(), model) if scale else make_pipeline(model)

# --- SIDEBAR TOP (Theme Toggle) ---
with st.sidebar:
    st.markdown("### ⚙️ Settings & Controls")
    
    # TOGGLE PARA SA DARK/LIGHT MODE
    dark_mode = st.toggle("🌙 Enable Dark Mode", value=False)
    
    st.divider()
    up = st.file_uploader("Upload Student Records (CSV)", type="csv")
    st.caption("Required columns: " + ", ".join(FEATURES + [TARGET]))
    st.divider()
    
    st.markdown("**Data Split & Scaling**")
    train_pct = st.slider("Training split (%)", 60, 90, 80, 5) 
    seed = st.number_input("Random seed", 0, 9999, 42)
    scale = st.checkbox("Standardize features (Z-score)", True,
                        help="Applied as per methodology to bring all variables to a comparable scale.")
    
    st.divider()
    st.markdown("**Decision Tree Pruning**")
    depth = st.slider("max_depth", 2, 12, 4)
    leaf = st.slider("min_samples_leaf", 1, 30, 5)
    alpha = st.select_slider("ccp_alpha", [0.0, 0.001, 0.005, 0.01, 0.02], 0.0)
    
    st.divider()
    st.markdown("**Validation**")
    k_folds = st.slider("Cross-validation folds", 3, 10, 10)
    
    st.caption("Prototype for research/demonstration only. Predictions support, not replace, teacher judgment.")

# --- DYNAMIC CSS BASED ON THEME ---
if dark_mode:
    bg_color = "#0F172A"        
    text_color = "#F8FAFC"      
    card_bg = "#1E293B"         
    border_color = "#334155"    
    tab_bg = "#1E293B"
    tab_selected = "#3B82F6"
    sidebar_bg = "#020617"      
else:
    bg_color = "#F8F9FA"        
    text_color = "#111827"      
    card_bg = "#FFFFFF"         
    border_color = "#E5E7EB"    
    tab_bg = "#FFFFFF"
    tab_selected = "#EFF6FF"
    sidebar_bg = "#1E293B"      

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: {text_color} !important;
    }}
    .main-header {{
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white !important;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .main-header h1, .main-header p {{ color: white !important; margin: 0; }}
    .main-header h1 {{ font-size: 28px; font-weight: 600; }}
    .main-header p {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #3B82F6;
        border-top: 1px solid {border_color};
        border-right: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
    }}
    div[data-testid="stMetric"] label {{ color: #94A3B8 !important; font-weight: 600; font-size: 14px; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: {text_color} !important; font-weight: 700; font-size: 24px; }}
    [data-testid="stSidebar"] {{ 
        background-color: {sidebar_bg};
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}
    [data-testid="stSidebar"] h3 {{ color: #60A5FA !important; }}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] > div {{
        background-color: #334155 !important;
        border: 1px dashed #64748B !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
        background-color: #475569 !important;
        color: #F8FAFC !important;
        border: 1px solid #64748B !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {{
        background-color: #3B82F6 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: {tab_bg};
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid {border_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #6B7280;
        font-weight: 600;
        padding: 0 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_selected};
        color: {"#FFFFFF" if dark_mode else "#3B82F6"} !important;
    }}
    div[data-testid="stDataFrame"] {{
        background-color: {card_bg};
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid {border_color};
    }}
    .stAlert {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# --- MAIN UI HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Analytics</h1>
    <p>Decision Tree vs Naive Bayes Classification | Calaca City Global College</p>
</div>
""", unsafe_allow_html=True)

# --- PROCESS DATA ---
if up is not None:
    raw, source = pd.read_csv(up), "Uploaded CSV File"
else:
    raw, source = make_sample(), "SYNTHETIC Demo Data"

missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df, log = preprocess(raw)
if df[TARGET].value_counts().min() < 5:
    st.error("Each class needs at least 5 records.")
    st.stop()

# --- TRAIN MODEL ---
X, y = df[FEATURES], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=train_pct / 100, stratify=y, random_state=int(seed))
results, models, preds = {}, {}, {}
for name in ["Decision Tree", "Naive Bayes"]:
    pipe = build(name, scale, depth, leaf, alpha, int(seed))
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); t1 = time.perf_counter()
    yp = pipe.predict(Xte); t2 = time.perf_counter()
    p, r, f, _ = precision_recall_fscore_support(yte, yp, average="weighted", zero_division=0)
    _, rc, _, _ = precision_recall_fscore_support(yte, yp, labels=["At-Risk"], zero_division=0)
    results[name] = {"Accuracy": accuracy_score(yte, yp), "Precision": p, "Recall": r,
                     "F1-score": f, "At-Risk Recall": rc[0],
                     "Train time (ms)": (t1 - t0) * 1000,
                     "Predict time (ms)": (t2 - t1) * 1000,
                     "Total time (ms)": (t2 - t0) * 1000}
    models[name], preds[name] = pipe, yp
res = pd.DataFrame(results).T

# --- TABS LAYOUT ---
t_data, t_res, t_cv, t_int, t_pred = st.tabs([
    "📊 Data Overview", 
    "📈 Model Results", 
    "🔄 Cross-validation", 
    "🧠 Interpretation", 
    "🔮 Live Prediction"
])

with t_data:
    st.markdown("### Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Uploaded", log["rows_in"])
    c2.metric("Valid Rows Used", log["rows_out"])
    c3.metric("Missing Values Filled", log["missing_filled"])
    c4.metric("Outliers Capped", log["outliers_capped"])
    
    st.info(f"**Duplicates Removed:** {log['duplicates_removed']} &nbsp;|&nbsp; **Data Split:** {len(Xtr)} Train / {len(Xte)} Test (Stratified) &nbsp;|&nbsp; **Source:** {source}")
    
    st.divider()
    a, b = st.columns([2, 1])
    with a:
        st.markdown(f"**Processed Dataset (All {len(df)} rows)**")
        st.dataframe(df, use_container_width=True, height=600)
    with b:
        st.markdown("**Class Distribution**")
        st.bar_chart(y.value_counts().reindex(CLASSES), height=600)
    st.download_button("📥 Download Sample CSV Format", make_sample().to_csv(index=False), "sample_students.csv", "text/csv")

with t_res:
    st.markdown("### Model Performance Comparison")
    st.dataframe(res.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)
    
    st.markdown("**Metrics Visualization**")
    st.bar_chart(res[["Accuracy", "Precision", "Recall", "F1-score", "At-Risk Recall"]].T, height=250)
    
    best = res["F1-score"].idxmax()
    fast = res["Total time (ms)"].idxmin()
    ar = res["At-Risk Recall"].idxmax()
    st.info(f"🏆 **Best Overall (F1):** {best} &nbsp;|&nbsp; 🎯 **Best At-Risk Recall:** {ar} &nbsp;|&nbsp; ⚡ **Fastest Execution:** {fast}")
    
    st.divider()
    st.markdown("### Confusion Matrices")
    cols = st.columns(2)
    for col, name in zip(cols, models):
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix(yte, preds[name], labels=CLASSES),
                                   display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{name}", fontweight="bold", pad=15)
            st.pyplot(fig)

with t_cv:
    st.markdown("### K-Fold Cross Validation")
    n_splits = int(min(k_folds, ytr.value_counts().min()))
    st.write(f"Evaluating model stability using Stratified **{n_splits}-fold** cross-validation.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    scoring = {"Accuracy": "accuracy", "Precision": "precision_weighted",
               "Recall": "recall_weighted", "F1-score": "f1_weighted"}
    rows = {}
    for name in models:
        cv = cross_validate(build(name, scale, depth, leaf, alpha, int(seed)),
                            Xtr, ytr, cv=skf, scoring=scoring)
        rows[name] = {m: f"{cv['test_' + m].mean():.3f} ± {cv['test_' + m].std():.3f}"
                      for m in scoring}
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
    st.caption("Values shown as Mean ± Standard Deviation. A lower standard deviation indicates higher stability across folds.")

with t_int:
    st.markdown("### Model Interpretability")
    tree = models["Decision Tree"][-1]
    a, b = st.columns(2)
    with a:
        st.markdown("**Decision Tree Feature Importance**")
        imp = pd.Series(tree.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(imp, height=300)
    with b:
        st.markdown("**Decision Tree IF-THEN Rules**")
        st.code(export_text(tree, feature_names=FEATURES), language="text")
    
    st.divider()
    st.markdown("**Naive Bayes Statistics (Priors & Feature Means)**")
    nb = models["Naive Bayes"][-1]
    st.dataframe(pd.DataFrame(nb.theta_, index=nb.classes_, columns=FEATURES)
                 .assign(prior=nb.class_prior_).round(3), use_container_width=True)
    st.caption("Thresholds/means are in standardized units because scaling is toggled ON.")

with t_pred:
    st.markdown("### Student Risk Predictor")
    st.write("Adjust the sliders to simulate a student's academic profile and view model predictions.")
    
    # --- BINAGO: DYNAMIC NA ANG SLIDERS PARA UMAYON SA DATASET (Halimbawa: prev_grade 1.0-5.0) ---
    c = st.columns(4)
    vals = {
        "attendance": c[0].slider("Attendance (%)", 0, 100, 80),
        "quiz_avg": c[1].slider("Quiz Average", 0, 100, 75),
        "exam_avg": c[2].slider("Exam Average", 0, 100, 72),
        "prev_grade": c[3].slider("Previous Grade (GPA)", 1.0, 5.0, 2.0, step=0.1) # Binago mula 50-100
    }
    row = pd.DataFrame([vals])
    
    st.divider()
    out = st.columns(2)
    for col, (name, pipe) in zip(out, models.items()):
        with col:
            pred = pipe.predict(row)[0]
            proba = pd.Series(pipe.predict_proba(row)[0], index=pipe.classes_).reindex(CLASSES)
            icon = {"Pass": "🟢", "At-Risk": "🟠", "Fail": "🔴"}[pred]
            
            st.markdown(f"**{name} Prediction**")
            st.metric(label="Result", value=f"{icon} {pred}")
            st.bar_chart(proba, height=200)

    st.divider()
    st.markdown("### 🚨 At-Risk Watchlist (Test Set)")
    best_model = res["At-Risk Recall"].idxmax()
    flag = Xte.assign(actual=yte.values, predicted=preds[best_model])
    flag = flag[flag.predicted.isin(["At-Risk", "Fail"])]
    
    st.warning(f"Model **{best_model}** flagged **{len(flag)}** students as At-Risk or Failing.")
    st.dataframe(flag, use_container_width=True)
    st.download_button("📥 Download At-Risk List (CSV)", flag.to_csv(index=False), "at_risk_students.csv")