"""
GlobalPulse | Socio-Economic Country Clustering Intelligence Platform
Author: Built for Harshil
Model: Agglomerative Hierarchical Clustering (4 clusters) on HELP-style country data
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to this script's own folder, so the app works
# no matter which directory Streamlit/Anaconda is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Country-data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "agglomerative_model_4_clusters__1_.pkl")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="GlobalPulse | Country Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
FEATURES = ["child_mort", "exports", "health", "imports", "income",
            "inflation", "life_expec", "total_fer", "gdpp"]

FEATURE_LABELS = {
    "child_mort": "Child Mortality (per 1000)",
    "exports": "Exports (% of GDP)",
    "health": "Health Spend (% of GDP)",
    "imports": "Imports (% of GDP)",
    "income": "Net Income per Capita ($)",
    "inflation": "Inflation (%)",
    "life_expec": "Life Expectancy (yrs)",
    "total_fer": "Fertility Rate",
    "gdpp": "GDP per Capita ($)",
}

CLUSTER_META = {
    0: {"name": "Developing Economies",   "color": "#5EC8F8", "icon": "🌤️",
        "desc": "Moderate income & health metrics, transitional growth trajectory."},
    1: {"name": "Developed Economies",    "color": "#4ADE80", "icon": "🟢",
        "desc": "High income, low child mortality, strong life expectancy."},
    2: {"name": "High-Need / Fragile",    "color": "#F87171", "icon": "🔴",
        "desc": "High child mortality & fertility, low GDP — priority for aid."},
    3: {"name": "Trade Hub Elite",        "color": "#C084FC", "icon": "💎",
        "desc": "Small population, extreme trade volume & GDP per capita."},
}

# ============================================================================
# THEME / CSS — Dark Glassmorphism
# ============================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
    --bg-0:#070b14; --bg-1:#0c1220; --bg-2:#111a2c;
    --glass:rgba(255,255,255,0.045); --glass-brd:rgba(255,255,255,0.09);
    --accent:#5EC8F8; --accent-2:#7C9BFF; --accent-3:#C084FC;
    --text-hi:#EAF2FF; --text-lo:#8FA3C4;
    --good:#4ADE80; --bad:#F87171; --warn:#FBBF24;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3,h4,.big-title{font-family:'Sora',sans-serif;}

.stApp{
    background:
      radial-gradient(circle at 15% 0%, rgba(94,200,248,0.10), transparent 45%),
      radial-gradient(circle at 85% 15%, rgba(192,132,252,0.09), transparent 40%),
      radial-gradient(circle at 50% 100%, rgba(76,222,128,0.06), transparent 45%),
      linear-gradient(180deg, var(--bg-0), var(--bg-1) 40%, var(--bg-0));
    color:var(--text-hi);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #080d18, #0a1120 60%, #070b14);
    border-right:1px solid var(--glass-brd);
}
#MainMenu, footer{visibility:hidden;}

/* ---------- Hero ---------- */
.hero{
    padding:26px 32px; border-radius:22px; margin-bottom:22px;
    background:linear-gradient(120deg, rgba(94,200,248,0.13), rgba(124,155,255,0.07) 45%, rgba(192,132,252,0.10));
    border:1px solid var(--glass-brd);
    box-shadow:0 8px 32px rgba(0,0,0,0.35);
    position:relative; overflow:hidden;
}
.hero:before{
    content:""; position:absolute; inset:0;
    background:radial-gradient(circle at 90% -10%, rgba(255,255,255,0.10), transparent 55%);
}
.hero h1{font-size:2.05rem; font-weight:800; margin:0 0 4px 0;
    background:linear-gradient(90deg,#EAF2FF,#B9D6FF 60%,#C084FC);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p{color:var(--text-lo); font-size:0.98rem; margin:0; max-width:720px;}
.badge-row{margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;}
.badge{padding:6px 14px; border-radius:999px; font-size:0.76rem; font-weight:600;
    background:rgba(255,255,255,0.06); border:1px solid var(--glass-brd); color:var(--text-hi);}

/* ---------- Glass cards ---------- */
.glass-card{
    background:var(--glass); border:1px solid var(--glass-brd); border-radius:18px;
    padding:20px 22px; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    box-shadow:0 4px 24px rgba(0,0,0,0.28); height:100%;
    transition:transform .18s ease, border-color .18s ease;
}
.glass-card:hover{ transform:translateY(-3px); border-color:rgba(255,255,255,0.18); }

.kpi-label{color:var(--text-lo); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;}
.kpi-value{font-family:'Sora',sans-serif; font-size:1.85rem; font-weight:800; color:var(--text-hi); margin-top:2px;}
.kpi-sub{font-size:0.78rem; color:var(--text-lo); margin-top:4px;}

.section-title{font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; color:var(--text-hi);
    margin:6px 0 14px 0; display:flex; align-items:center; gap:8px;}
.section-title .bar{width:5px; height:18px; border-radius:4px;
    background:linear-gradient(180deg,var(--accent),var(--accent-3));}

.cluster-chip{
    display:inline-flex; align-items:center; gap:8px; padding:9px 16px; border-radius:14px;
    background:rgba(255,255,255,0.05); border:1px solid var(--glass-brd); font-weight:600; font-size:0.85rem;
}
.dot{width:10px; height:10px; border-radius:50%; display:inline-block;}

hr.soft{border:none; border-top:1px solid var(--glass-brd); margin:18px 0;}

/* Streamlit widget restyle */
[data-testid="stMetric"]{
    background:var(--glass); border:1px solid var(--glass-brd); border-radius:16px; padding:14px 16px;
}
div[data-testid="stTable"], .stDataFrame{border-radius:14px; overflow:hidden;}
button[kind="primary"]{
    background:linear-gradient(90deg,var(--accent),var(--accent-2)) !important;
    border:none !important; font-weight:700 !important; border-radius:12px !important;
}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{
    background:rgba(255,255,255,0.04); border-radius:10px 10px 0 0; padding:8px 18px; color:var(--text-lo);
}
.stTabs [aria-selected="true"]{background:rgba(94,200,248,0.14); color:var(--text-hi) !important;}
input, textarea, select{color:var(--text-hi) !important;}
.stSlider [data-baseweb="slider"]{padding-top:6px;}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"


def style_fig(fig, h=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=h,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color="#EAF2FF", size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ============================================================================
# DATA / MODEL LOADING
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def prep(df):
    """Attach cluster labels (from training) + build scaler & centroids for
    nearest-centroid prediction of NEW / unseen countries.
    (AgglomerativeClustering has no native .predict for new data.)"""
    model = load_model()
    data = df.copy()
    data["cluster"] = model.labels_[: len(data)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[FEATURES])
    scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
    scaled_df["cluster"] = data["cluster"].values

    centroids = scaled_df.groupby("cluster")[FEATURES].mean()
    return data, scaler, centroids


def predict_new(sample_dict, scaler, centroids):
    x = pd.DataFrame([sample_dict])[FEATURES]
    x_scaled = scaler.transform(x)[0]
    dists = np.linalg.norm(centroids[FEATURES].values - x_scaled, axis=1)
    cluster = int(centroids.index[np.argmin(dists)])
    conf = 1 - (dists.min() / (dists.sum() + 1e-9))
    return cluster, dists, conf


raw_df = load_data()
model = load_model()
df, scaler, centroids = prep(raw_df)
df["cluster_name"] = df["cluster"].map(lambda c: CLUSTER_META[c]["name"])
df["cluster_color"] = df["cluster"].map(lambda c: CLUSTER_META[c]["color"])

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 18px 0;">
        <div style="font-size:2.1rem;">🌍</div>
        <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.25rem; color:#EAF2FF;">GlobalPulse</div>
        <div style="font-size:0.75rem; color:#8FA3C4; letter-spacing:0.05em;">COUNTRY INTELLIGENCE ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🧬 Cluster Deep-Dive", "🔎 Country Explorer",
         "🎯 Predict New Country", "🗂️ Data Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Model</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8FA3C4; line-height:1.6;">
    Algorithm: <b style="color:#EAF2FF;">Agglomerative (Ward)</b><br>
    Clusters: <b style="color:#EAF2FF;">4</b><br>
    Countries: <b style="color:#EAF2FF;">{len(df)}</b><br>
    Features: <b style="color:#EAF2FF;">{len(FEATURES)}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Cluster Legend</div>", unsafe_allow_html=True)
    for cid, meta in CLUSTER_META.items():
        st.markdown(f"""
        <div class="cluster-chip" style="margin-bottom:8px; width:100%;">
            <span class="dot" style="background:{meta['color']};"></span>
            {meta['icon']} {meta['name']}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.markdown("""
    <div class="hero">
        <h1>Global Socio-Economic Clustering</h1>
        <p>An unsupervised intelligence layer that segments 167 nations into actionable development
        cohorts using hierarchical clustering across health, trade and economic indicators.</p>
        <div class="badge-row">
            <span class="badge">🧬 Agglomerative Ward Linkage</span>
            <span class="badge">📈 9 Socio-Economic Indicators</span>
            <span class="badge">🌐 167 Countries</span>
            <span class="badge">🎯 Live Prediction Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Countries Analyzed", f"{len(df)}", "Across all regions", c1),
        ("Clusters Formed", "4", "Ward linkage, Euclidean", c2),
        ("Avg GDP / Capita", f"${df['gdpp'].mean():,.0f}", f"Range ${df['gdpp'].min():,.0f} – ${df['gdpp'].max():,.0f}", c3),
        ("Avg Child Mortality", f"{df['child_mort'].mean():.1f}", "per 1,000 live births", c4),
    ]
    for label, val, sub, col in kpis:
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("<div class='section-title'><span class='bar'></span>GDP per Capita vs Child Mortality (Cluster View)</div>", unsafe_allow_html=True)
        fig = px.scatter(
            df, x="gdpp", y="child_mort", size="income", color="cluster_name",
            hover_name="country", size_max=32, log_x=True,
            color_discrete_map={CLUSTER_META[c]["name"]: CLUSTER_META[c]["color"] for c in CLUSTER_META},
            labels={"gdpp": "GDP per Capita ($, log)", "child_mort": "Child Mortality", "cluster_name": "Cluster"},
        )
        fig.update_traces(marker=dict(line=dict(width=0.6, color="rgba(255,255,255,0.4)")))
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with colB:
        st.markdown("<div class='section-title'><span class='bar'></span>Cluster Distribution</div>", unsafe_allow_html=True)
        counts = df["cluster_name"].value_counts().reset_index()
        counts.columns = ["cluster_name", "count"]
        fig2 = px.pie(
            counts, names="cluster_name", values="count", hole=0.62,
            color="cluster_name",
            color_discrete_map={CLUSTER_META[c]["name"]: CLUSTER_META[c]["color"] for c in CLUSTER_META},
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=11)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig2, 460), use_container_width=True)

    st.markdown("<div class='section-title'><span class='bar'></span>World Map — Cluster Membership</div>", unsafe_allow_html=True)
    fig3 = px.choropleth(
        df, locations="country", locationmode="country names", color="cluster_name",
        hover_name="country", hover_data={"gdpp": True, "child_mort": True, "cluster_name": False},
        color_discrete_map={CLUSTER_META[c]["name"]: CLUSTER_META[c]["color"] for c in CLUSTER_META},
    )
    fig3.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
                      coastlinecolor="rgba(255,255,255,0.15)", landcolor="rgba(255,255,255,0.04)",
                      oceancolor="rgba(0,0,0,0)", showocean=True)
    st.plotly_chart(style_fig(fig3, 480), use_container_width=True)

# ============================================================================
# PAGE: CLUSTER DEEP-DIVE
# ============================================================================
elif page == "🧬 Cluster Deep-Dive":
    st.markdown("""
    <div class="hero">
        <h1>Cluster Deep-Dive</h1>
        <p>Compare the socio-economic fingerprint of each cluster across every indicator used in training.</p>
    </div>
    """, unsafe_allow_html=True)

    cluster_stats = df.groupby("cluster")[FEATURES].mean()
    norm_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min() + 1e-9)

    cols = st.columns(4)
    for cid, col in zip(sorted(CLUSTER_META.keys()), cols):
        meta = CLUSTER_META[cid]
        n = (df["cluster"] == cid).sum()
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid {meta['color']};">
                <div style="font-size:1.5rem;">{meta['icon']}</div>
                <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:1.02rem; margin-top:4px;">{meta['name']}</div>
                <div class="kpi-sub" style="margin-top:6px;">{meta['desc']}</div>
                <div style="margin-top:10px; font-size:0.8rem; color:#8FA3C4;">Countries: <b style="color:#EAF2FF;">{n}</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><span class='bar'></span>Multi-Dimensional Cluster Fingerprint</div>", unsafe_allow_html=True)

    fig = go.Figure()
    for cid in sorted(CLUSTER_META.keys()):
        meta = CLUSTER_META[cid]
        fig.add_trace(go.Scatterpolar(
            r=norm_stats.loc[cid].values.tolist() + [norm_stats.loc[cid].values[0]],
            theta=[FEATURE_LABELS[f] for f in FEATURES] + [FEATURE_LABELS[FEATURES[0]]],
            fill="toself", name=meta["name"], line=dict(color=meta["color"], width=2),
            opacity=0.55,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.12)"),
                                  angularaxis=dict(gridcolor="rgba(255,255,255,0.12)")),
                       showlegend=True)
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    st.markdown("<div class='section-title'><span class='bar'></span>Indicator Distribution by Cluster</div>", unsafe_allow_html=True)
    feat_pick = st.selectbox("Select indicator", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    fig4 = px.box(
        df, x="cluster_name", y=feat_pick, color="cluster_name", points="all",
        color_discrete_map={CLUSTER_META[c]["name"]: CLUSTER_META[c]["color"] for c in CLUSTER_META},
        labels={"cluster_name": "Cluster", feat_pick: FEATURE_LABELS[feat_pick]},
    )
    fig4.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig4, 440), use_container_width=True)

    st.markdown("<div class='section-title'><span class='bar'></span>Parallel Coordinates — All Indicators</div>", unsafe_allow_html=True)
    pc_df = df.copy()
    pc_df["cluster_code"] = pc_df["cluster"]
    fig5 = px.parallel_coordinates(
        pc_df, dimensions=FEATURES, color="cluster_code",
        color_continuous_scale=[CLUSTER_META[c]["color"] for c in sorted(CLUSTER_META.keys())],
        labels=FEATURE_LABELS,
    )
    st.plotly_chart(style_fig(fig5, 460), use_container_width=True)

# ============================================================================
# PAGE: COUNTRY EXPLORER
# ============================================================================
elif page == "🔎 Country Explorer":
    st.markdown("""
    <div class="hero">
        <h1>Country Explorer</h1>
        <p>Drill into any nation's socio-economic profile against its assigned cluster benchmark.</p>
    </div>
    """, unsafe_allow_html=True)

    country = st.selectbox("Select a country", sorted(df["country"].unique()))
    row = df[df["country"] == country].iloc[0]
    meta = CLUSTER_META[row["cluster"]]

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {meta['color']}; margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.4rem;">{country}</div>
                <div class="kpi-sub">{meta['desc']}</div>
            </div>
            <div class="cluster-chip">
                <span class="dot" style="background:{meta['color']};"></span>
                {meta['icon']} {meta['name']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kc = st.columns(len(FEATURES) // 2 + 1)
    metrics = list(FEATURE_LABELS.items())
    for i, (feat, label) in enumerate(metrics):
        with kc[i % len(kc)]:
            cluster_avg = df[df["cluster"] == row["cluster"]][feat].mean()
            delta = row[feat] - cluster_avg
            st.metric(label, f"{row[feat]:,.2f}", f"{delta:+.2f} vs cluster avg", delta_color="normal")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    colL, colR = st.columns(2)

    with colL:
        st.markdown("<div class='section-title'><span class='bar'></span>Country vs Cluster Average</div>", unsafe_allow_html=True)
        cluster_avg_vals = df[df["cluster"] == row["cluster"]][FEATURES].mean()
        norm_max = df[FEATURES].max()
        country_norm = (row[FEATURES] / norm_max).values
        cluster_norm = (cluster_avg_vals / norm_max).values

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=list(country_norm) + [country_norm[0]],
                                       theta=[FEATURE_LABELS[f] for f in FEATURES] + [FEATURE_LABELS[FEATURES[0]]],
                                       name=country, fill="toself", line=dict(color=meta["color"], width=2)))
        fig.add_trace(go.Scatterpolar(r=list(cluster_norm) + [cluster_norm[0]],
                                       theta=[FEATURE_LABELS[f] for f in FEATURES] + [FEATURE_LABELS[FEATURES[0]]],
                                       name="Cluster Avg", fill="toself",
                                       line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot")))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False,
                                                       gridcolor="rgba(255,255,255,0.12)")))
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with colR:
        st.markdown("<div class='section-title'><span class='bar'></span>Global Percentile Ranking</div>", unsafe_allow_html=True)
        pct = {f: (df[f] < row[f]).mean() * 100 for f in FEATURES}
        fig2 = go.Figure(go.Bar(
            x=list(pct.values()), y=[FEATURE_LABELS[f] for f in pct.keys()],
            orientation="h", marker=dict(color=list(pct.values()), colorscale="Tealgrn"),
            text=[f"{v:.0f}th" for v in pct.values()], textposition="outside",
        ))
        fig2.update_layout(xaxis=dict(range=[0, 105], title="Percentile"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig2, 420), use_container_width=True)

# ============================================================================
# PAGE: PREDICT NEW COUNTRY
# ============================================================================
elif page == "🎯 Predict New Country":
    st.markdown("""
    <div class="hero">
        <h1>Predict New Country Cluster</h1>
        <p>Enter socio-economic indicators for a new or hypothetical country to classify it into one of the
        four development clusters using nearest-centroid assignment against the trained model.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("<div class='section-title'><span class='bar'></span>Input Indicators</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            child_mort = st.number_input("Child Mortality (per 1000)", 0.0, 250.0, 25.0, step=0.5)
            exports = st.number_input("Exports (% of GDP)", 0.0, 250.0, 40.0, step=0.5)
            health = st.number_input("Health Spend (% of GDP)", 0.0, 20.0, 6.5, step=0.1)
        with c2:
            imports = st.number_input("Imports (% of GDP)", 0.0, 250.0, 45.0, step=0.5)
            income = st.number_input("Net Income per Capita ($)", 0, 150000, 12000, step=100)
            inflation = st.number_input("Inflation (%)", -10.0, 100.0, 5.0, step=0.1)
        with c3:
            life_expec = st.number_input("Life Expectancy (yrs)", 20.0, 90.0, 70.0, step=0.5)
            total_fer = st.number_input("Fertility Rate", 0.5, 8.0, 2.5, step=0.1)
            gdpp = st.number_input("GDP per Capita ($)", 0, 150000, 6000, step=100)

        submitted = st.form_submit_button("🔮 Classify Country", use_container_width=True, type="primary")

    if submitted:
        sample = dict(child_mort=child_mort, exports=exports, health=health, imports=imports,
                      income=income, inflation=inflation, life_expec=life_expec,
                      total_fer=total_fer, gdpp=gdpp)
        cluster, dists, conf = predict_new(sample, scaler, centroids)
        meta = CLUSTER_META[cluster]

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="border:1px solid {meta['color']}55; text-align:center; padding:32px;">
            <div style="font-size:2.6rem;">{meta['icon']}</div>
            <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.6rem; color:{meta['color']}; margin-top:6px;">
                {meta['name']}
            </div>
            <div class="kpi-sub" style="font-size:0.92rem; margin-top:6px;">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        colL, colR = st.columns(2)
        with colL:
            st.markdown("<div class='section-title'><span class='bar'></span>Distance to Each Cluster Centroid</div>", unsafe_allow_html=True)
            dist_df = pd.DataFrame({
                "Cluster": [CLUSTER_META[c]["name"] for c in centroids.index],
                "Distance": dists,
                "Color": [CLUSTER_META[c]["color"] for c in centroids.index],
            }).sort_values("Distance")
            fig = go.Figure(go.Bar(x=dist_df["Distance"], y=dist_df["Cluster"], orientation="h",
                                    marker=dict(color=dist_df["Color"])))
            fig.update_layout(xaxis_title="Euclidean Distance (scaled space)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        with colR:
            st.markdown("<div class='section-title'><span class='bar'></span>Input Profile vs Assigned Cluster</div>", unsafe_allow_html=True)
            cluster_avg_vals = df[df["cluster"] == cluster][FEATURES].mean()
            norm_max = df[FEATURES].max()
            input_norm = (pd.Series(sample)[FEATURES] / norm_max).values
            cluster_norm = (cluster_avg_vals / norm_max).values
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=list(input_norm) + [input_norm[0]],
                                            theta=[FEATURE_LABELS[f] for f in FEATURES] + [FEATURE_LABELS[FEATURES[0]]],
                                            name="Your Input", fill="toself", line=dict(color=meta["color"], width=2)))
            fig2.add_trace(go.Scatterpolar(r=list(cluster_norm) + [cluster_norm[0]],
                                            theta=[FEATURE_LABELS[f] for f in FEATURES] + [FEATURE_LABELS[FEATURES[0]]],
                                            name="Cluster Avg", fill="toself",
                                            line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot")))
            fig2.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False,
                                                            gridcolor="rgba(255,255,255,0.12)")))
            st.plotly_chart(style_fig(fig2, 340), use_container_width=True)

        st.info("ℹ️ Agglomerative Clustering has no native predict function for unseen data — "
                "classification is performed via **nearest-centroid matching** in the same scaled "
                "feature space used to train the model.")

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "🗂️ Data Explorer":
    st.markdown("""
    <div class="hero">
        <h1>Raw Data Explorer</h1>
        <p>Filter, search and export the full clustered dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("🔍 Search country", "")
    with c2:
        cluster_filter = st.multiselect(
            "Filter by cluster", options=list(CLUSTER_META.keys()),
            format_func=lambda c: CLUSTER_META[c]["name"], default=list(CLUSTER_META.keys()),
        )

    filtered = df[df["cluster"].isin(cluster_filter)]
    if search:
        filtered = filtered[filtered["country"].str.contains(search, case=False)]

    st.markdown(f"<div class='kpi-sub'>Showing {len(filtered)} of {len(df)} countries</div>", unsafe_allow_html=True)
    st.dataframe(
        filtered[["country", "cluster_name"] + FEATURES].rename(columns={**FEATURE_LABELS, "cluster_name": "Cluster", "country": "Country"}),
        use_container_width=True, height=500,
    )

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="clustered_countries.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<hr class='soft'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8FA3C4; font-size:0.78rem; padding-bottom:10px;">
    GlobalPulse Country Intelligence · Agglomerative Clustering Engine · Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)