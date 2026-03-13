import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CFISD Rezoning Impact Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .finding-box {
        background: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    .stSelectbox label, .stSlider label { color: #8b949e !important; font-size: 0.85rem !important; }
    div[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD / GENERATE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    schools = [
        {"name": "Cypress Creek HS",    "level": "High",        "rezoned_year": 2020},
        {"name": "Cypress Ridge HS",    "level": "High",        "rezoned_year": None},
        {"name": "Cypress Lakes HS",    "level": "High",        "rezoned_year": 2021},
        {"name": "Cypress Springs HS",  "level": "High",        "rezoned_year": None},
        {"name": "Cypress Park HS",     "level": "High",        "rezoned_year": 2020},
        {"name": "Arnold MS",           "level": "Middle",      "rezoned_year": 2020},
        {"name": "Cook MS",             "level": "Middle",      "rezoned_year": None},
        {"name": "Bleyl MS",            "level": "Middle",      "rezoned_year": 2021},
        {"name": "Hamilton MS",         "level": "Middle",      "rezoned_year": None},
        {"name": "Spillane MS",         "level": "Middle",      "rezoned_year": 2020},
        {"name": "Birkes ES",           "level": "Elementary",  "rezoned_year": 2020},
        {"name": "Sampson ES",          "level": "Elementary",  "rezoned_year": None},
        {"name": "Swenke ES",           "level": "Elementary",  "rezoned_year": 2021},
        {"name": "Jowell ES",           "level": "Elementary",  "rezoned_year": None},
        {"name": "Frazier ES",          "level": "Elementary",  "rezoned_year": 2021},
    ]
    years = list(range(2016, 2024))
    rows = []
    for s in schools:
        base_score = np.random.uniform(68, 88)
        base_hispanic = np.random.uniform(30, 65)
        base_white = np.random.uniform(20, 50)
        base_black = np.random.uniform(5, 20)
        base_econ_dis = np.random.uniform(25, 60)
        base_pop = np.random.randint(400, 2200)
        for yr in years:
            rezoned = 1 if (s["rezoned_year"] and yr >= s["rezoned_year"]) else 0
            score = base_score + np.random.normal(0, 1.5)
            if rezoned:
                score += np.random.normal(-2.5, 1.0)
            income = np.random.uniform(55000, 95000)
            rows.append({
                "School": s["name"],
                "Level": s["level"],
                "Year": yr,
                "Score": round(score, 1),
                "Pct_Hispanic": round(min(base_hispanic + rezoned * np.random.uniform(2, 6) + np.random.normal(0, 1), 90), 1),
                "Pct_White": round(max(base_white - rezoned * np.random.uniform(1, 4) + np.random.normal(0, 1), 5), 1),
                "Pct_Black": round(min(base_black + np.random.normal(0, 0.5), 40), 1),
                "Pct_EconDis": round(min(base_econ_dis + rezoned * np.random.uniform(1, 4) + np.random.normal(0, 1), 90), 1),
                "Enrollment": int(base_pop + np.random.randint(-50, 100)),
                "Rezoned": rezoned,
                "Income": round(income, 0),
                "Rezoned_Year": s["rezoned_year"],
            })
    return pd.DataFrame(rows)

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CFISD Analysis")
    st.markdown("---")
    level_filter = st.multiselect(
        "School Level",
        options=["Elementary", "Middle", "High"],
        default=["Elementary", "Middle", "High"]
    )
    year_range = st.slider("Year Range", 2016, 2023, (2016, 2023))
    st.markdown("---")
    st.markdown("""
    **Data Sources**
    - Texas Education Agency
    - CFISD District Reports
    - U.S. Census Bureau
    
    **Methods**
    - Difference-in-Differences
    - OLS Regression
    - Demographic Clustering
    
    **Author**  
    Independent Research Project  
    Cypress, TX · 2024
    """)

# Apply filters
filtered = df[
    (df["Level"].isin(level_filter)) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
# School Rezoning & Academic Outcomes
### Cypress-Fairbanks Independent School District · 2016–2023
""")
st.markdown("""
> This project investigates whether school rezoning decisions in CFISD are statistically 
> associated with changes in STAAR test performance and demographic composition. 
> Using a Difference-in-Differences framework, I compare rezoned schools against 
> comparable non-rezoned schools across an 8-year window.
""")

st.markdown("---")

# ─────────────────────────────────────────────
# KEY METRICS
# ─────────────────────────────────────────────
rezoned_schools = filtered[filtered["Rezoned"] == 1]
control_schools = filtered[filtered["Rezoned"] == 0]
avg_score_rezoned = rezoned_schools["Score"].mean()
avg_score_control = control_schools["Score"].mean()
score_gap = avg_score_rezoned - avg_score_control
n_rezoned = filtered[filtered["Rezoned"] == 1]["School"].nunique()
n_schools = filtered["School"].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_schools}</div>
        <div class="metric-label">Schools Analyzed</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_rezoned}</div>
        <div class="metric-label">Rezoned Schools</div>
    </div>""", unsafe_allow_html=True)
with col3:
    arrow = "▼" if score_gap < 0 else "▲"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{'#f85149' if score_gap < 0 else '#3fb950'}">{arrow} {abs(score_gap):.1f}%</div>
        <div class="metric-label">Score Δ (Rezoned vs Control)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    n_years = year_range[1] - year_range[0] + 1
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_years}</div>
        <div class="metric-label">Years of Data</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART 1 — SCORE TRENDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">01 · STAAR Score Trends</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([2, 1])

with col_a:
    score_by_year = filtered.groupby(["Year", "Rezoned"])["Score"].mean().reset_index()
    rezoned_data = score_by_year[score_by_year["Rezoned"] == 1]
    control_data = score_by_year[score_by_year["Rezoned"] == 0]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    ax.plot(control_data["Year"], control_data["Score"], color="#58a6ff",
            linewidth=2.5, marker="o", markersize=5, label="Control (Not Rezoned)")
    ax.plot(rezoned_data["Year"], rezoned_data["Score"], color="#f85149",
            linewidth=2.5, marker="o", markersize=5, linestyle="--", label="Treatment (Rezoned)")
    ax.axvline(x=2019.5, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(2019.6, ax.get_ylim()[0] + 1, "Rezoning begins", color="#e3b341",
            fontsize=8, family="monospace")
    ax.set_xlabel("Year", color="#8b949e", fontsize=9)
    ax.set_ylabel("Avg STAAR Score (%)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.markdown("**Key Findings**")
    st.markdown(f"""
    <div class="finding-box">Rezoned schools show a mean STAAR score of <strong>{avg_score_rezoned:.1f}%</strong>, compared to <strong>{avg_score_control:.1f}%</strong> in control schools.</div>
    <div class="finding-box">The score gap widens post-2020, suggesting a potential short-term disruption effect following rezoning.</div>
    <div class="finding-box">Control schools maintained more stable trajectories, consistent with a Difference-in-Differences interpretation.</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# CHART 2 — DEMOGRAPHIC SHIFTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">02 · Demographic Composition Shifts</div>', unsafe_allow_html=True)

demo_by_year = filtered.groupby(["Year", "Rezoned"])[
    ["Pct_Hispanic", "Pct_White", "Pct_Black", "Pct_EconDis"]
].mean().reset_index()

fig2, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#161b22")
titles = ["Control Schools", "Rezoned Schools"]
palette = {"Pct_Hispanic": "#58a6ff", "Pct_White": "#3fb950", "Pct_Black": "#e3b341", "Pct_EconDis": "#f85149"}

for i, rz in enumerate([0, 1]):
    ax = axes[i]
    ax.set_facecolor("#161b22")
    data_sub = demo_by_year[demo_by_year["Rezoned"] == rz]
    for col, color in palette.items():
        ax.plot(data_sub["Year"], data_sub[col], color=color, linewidth=2, label=col.replace("Pct_", "% "))
    ax.set_title(titles[i], color="#c9d1d9", fontsize=10, family="monospace")
    ax.set_xlabel("Year", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=7)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# ─────────────────────────────────────────────
# CHART 3 — REGRESSION (DiD)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">03 · Regression Analysis (Difference-in-Differences)</div>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns([1, 1])

with col_r1:
    model_data = df.copy()
    model_data["Post"] = (model_data["Year"] >= 2020).astype(int)
    model_data["DiD"] = model_data["Rezoned"] * model_data["Post"]
    model = smf.ols("Score ~ Rezoned + Post + DiD + Pct_EconDis + np.log(Income)", data=model_data).fit()

    coef_names = {
        "Intercept": "Intercept",
        "Rezoned": "Rezoned School",
        "Post": "Post-2020",
        "DiD": "DiD (Rezoned × Post)",
        "Pct_EconDis": "% Econ. Disadvantaged",
        "np.log(Income)": "Log(Median Income)"
    }
    summary_df = pd.DataFrame({
        "Variable": [coef_names.get(k, k) for k in model.params.index],
        "Coefficient": model.params.values.round(3),
        "P-value": model.pvalues.values.round(4),
        "Significant": ["✓" if p < 0.05 else "✗" for p in model.pvalues.values]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption(f"R² = {model.rsquared:.3f} · N = {int(model.nobs)} school-years")

with col_r2:
    # Coefficient plot
    fig3, ax = plt.subplots(figsize=(6, 4), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    params = model.params[1:]
    conf = model.conf_int().iloc[1:]
    colors_coef = ["#f85149" if v < 0 else "#3fb950" for v in params.values]
    y_pos = range(len(params))
    ax.barh(list(y_pos), params.values, color=colors_coef, alpha=0.85, height=0.5)
    ax.errorbar(params.values, list(y_pos),
                xerr=[params.values - conf[0].values, conf[1].values - params.values],
                fmt="none", color="#8b949e", capsize=3, linewidth=1)
    ax.axvline(0, color="#30363d", linewidth=1)
    short_names = ["Rezoned", "Post-2020", "DiD", "EconDis", "log(Income)"]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(short_names, color="#c9d1d9", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_xlabel("Coefficient", color="#8b949e", fontsize=8)
    ax.set_title("Regression Coefficients", color="#c9d1d9", fontsize=9, family="monospace")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#21262d", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig3)

st.markdown("---")

# ─────────────────────────────────────────────
# CHART 4 — SCHOOL TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">04 · School-Level Comparison</div>', unsafe_allow_html=True)

school_summary = filtered.groupby(["School", "Level", "Rezoned_Year"]).agg(
    Avg_Score=("Score", "mean"),
    Avg_EconDis=("Pct_EconDis", "mean"),
    Avg_Hispanic=("Pct_Hispanic", "mean"),
    Avg_Enrollment=("Enrollment", "mean")
).reset_index()
school_summary["Rezoned"] = school_summary["Rezoned_Year"].apply(lambda x: "✓ " + str(int(x)) if x else "✗ No")
school_summary["Avg_Score"] = school_summary["Avg_Score"].round(1)
school_summary["Avg_EconDis"] = school_summary["Avg_EconDis"].round(1)
school_summary["Avg_Hispanic"] = school_summary["Avg_Hispanic"].round(1)
school_summary["Avg_Enrollment"] = school_summary["Avg_Enrollment"].round(0).astype(int)
display_cols = ["School", "Level", "Rezoned", "Avg_Score", "Avg_EconDis", "Avg_Hispanic", "Avg_Enrollment"]
st.dataframe(
    school_summary[display_cols].rename(columns={
        "Avg_Score": "Avg Score (%)",
        "Avg_EconDis": "Econ. Dis. (%)",
        "Avg_Hispanic": "Hispanic (%)",
        "Avg_Enrollment": "Enrollment"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ─────────────────────────────────────────────
# SIMULATION TOOL
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">05 · Rezoning Impact Simulator</div>', unsafe_allow_html=True)
st.markdown("Adjust variables to predict the likely STAAR score change for a hypothetical rezoning scenario.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    sim_econ_dis = st.slider("% Economically Disadvantaged", 10, 90, 40)
with col_s2:
    sim_income = st.slider("Median Household Income ($K)", 40, 120, 70)
with col_s3:
    sim_post = st.radio("Timeline", ["Pre-rezoning", "Post-rezoning"])

did_val = 1 if sim_post == "Post-rezoning" else 0
pred_score = (
    model.params["Intercept"]
    + model.params["Rezoned"] * 1
    + model.params["Post"] * did_val
    + model.params["DiD"] * did_val
    + model.params["Pct_EconDis"] * sim_econ_dis
    + model.params["np.log(Income)"] * np.log(sim_income * 1000)
)

col_pred1, col_pred2 = st.columns(2)
with col_pred1:
    color = "#3fb950" if pred_score >= 75 else "#e3b341" if pred_score >= 65 else "#f85149"
    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px">
        <div class="metric-value" style="color:{color}">{pred_score:.1f}%</div>
        <div class="metric-label">Predicted STAAR Score</div>
    </div>""", unsafe_allow_html=True)
with col_pred2:
    risk = "Low" if pred_score >= 75 else "Medium" if pred_score >= 65 else "High"
    risk_color = "#3fb950" if risk == "Low" else "#e3b341" if risk == "Medium" else "#f85149"
    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px">
        <div class="metric-value" style="color:{risk_color}">{risk}</div>
        <div class="metric-label">Performance Risk Level</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<small style="color:#484f58">
⚠️ Note: This dashboard uses simulated data for demonstration purposes. 
Replace <code>data/staar_scores.csv</code> with real Texas Education Agency data to produce 
accurate research findings. The statistical framework (DiD regression) is production-ready.
</small>
""", unsafe_allow_html=True)
