# pages/5_🍱_Meal_Clusters.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Page config
st.set_page_config(layout="wide", page_title="SNP — Meal Clusters", page_icon="🍱")

# ---------- Styling to match app (clean / premium) ----------
st.markdown(
    """
<style>
/* keep consistent dark theme + subtle glass cards */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }

.cluster-card {
  background: rgba(255,255,255,0.03);
  border-radius: 14px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 18px;
}

h1,h2,h3 { color: #E6EEF8 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
@st.cache_data
def load_food_df():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))
    path = os.path.join(root, "data", "cleaned_nutrition_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("cleaned_nutrition_dataset.csv not found in /data/")
    df = pd.read_csv(path)
    # ensure numeric columns exist and are numeric
    for c in ["calories", "protein", "carbs", "fat", "fibre", "sugar", "sodium"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    return df

def safe_pca_and_kmeans(df, feature_cols, n_components=2, n_clusters=3, random_state=42):
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(Xs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xs)
    centers_scaled = kmeans.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    centers_pca = pca.transform(centers_scaled)
    return Xp, labels, kmeans, centers_unscaled, centers_pca, scaler, pca

def name_cluster_from_centroid(centroid, feature_names):
    # centroid is in original units (unscaled)
    # decide dominant nutrient and simple name
    vals = {f: centroid[i] for i,f in enumerate(feature_names)}
    sorted_feats = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    top = sorted_feats[0][0]
    if top == "protein":
        return "High - Protein"
    if top == "carbs":
        return "High - Carbs"
    if top == "fat":
        return "High - Fat"
    # fallback balanced detection
    arr = np.array(list(vals.values()))
    if arr.std() < arr.mean() * 0.25:
        return "Balanced"
    return f"Dominant: {top}"

def assignments_df_for_download(df_vis):
    df_out = df_vis[["food_name", "category", "calories", "protein", "carbs", "fat", "cluster"]].copy()
    return df_out

# ---------- Load dataset ----------
try:
    df_food = load_food_df()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

FEATURES = ["protein", "carbs", "fat"]

# ---------- Session check ----------
if "profile" not in st.session_state or "food_log" not in st.session_state:
    st.warning("Fill the User Input page first so I can highlight your foods here.")
    if st.button("Go to User Input"):
        st.switch_page("pages/2_🧍_User_Input.py")
    st.stop()

profile = st.session_state["profile"]
food_log = st.session_state["food_log"]  # dict food -> grams

# ---------- Header ----------
st.title("🍱 Meal Clusters — Nutrient Galaxy (Professional View)")
st.markdown("Interactive clustering of foods by macronutrients. Use controls on the left to explore cluster structure and discover which nutrient-groups your foods belong to.")
st.markdown("---")

# ---------- Controls column ----------
ctrl_col, vis_col = st.columns([1, 2.2])

with ctrl_col:
    st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
    st.subheader("Controls")
    k = st.slider("Clusters (k)", min_value=2, max_value=6, value=3, step=1)
    sample_pct = st.slider("Show % of foods (sampling)", min_value=30, max_value=100, value=100, step=10)
    # choice: color by category or cluster
    color_by = st.selectbox("Color points by", options=["cluster", "category"], index=0)
    highlight_mode = st.selectbox("Highlight", options=["Your logged foods", "A single food", "None"], index=0)
    if highlight_mode == "A single food":
        chosen_food = st.selectbox("Pick a food", options=sorted(df_food["food_name"].tolist()))
    else:
        chosen_food = None
    st.markdown("Tip: reduce sample% if page is slow. Clusters are recomputed instantly for small datasets.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Data for visualization (sampling) ----------
if sample_pct < 100:
    df_vis = df_food.sample(frac=sample_pct/100.0, random_state=42).reset_index(drop=True)
else:
    df_vis = df_food.copy().reset_index(drop=True)

# make sure foods have unique names for matching
df_vis["food_name"] = df_vis["food_name"].astype(str)

# ---------- compute PCA + kmeans safely ----------
try:
    Xp, labels, kmeans, centers_unscaled, centers_pca, scaler, pca = safe_pca_and_kmeans(
        df_vis, FEATURES, n_components=2, n_clusters=k, random_state=42
    )
except Exception as e:
    st.error(f"Clustering failed: {e}")
    st.stop()

df_vis["pc1"] = Xp[:,0]
df_vis["pc2"] = Xp[:,1]
df_vis["cluster"] = labels.astype(int)

# ---------- Build left: PCA 2D cluster map (main) ----------
with vis_col:
    left_col, right_col = st.columns([1.75, 1.25])

    with left_col:
        st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
        st.subheader("PCA 2D — Cluster Map (Primary view)")

        # base scatter: light points
        fig = go.Figure()

        palette = px.colors.qualitative.Plotly

        # if color_by==category, we still need colors; create mapping
        if color_by == "category":
            cats = df_vis["category"].fillna("unknown").astype(str)
            cat_map = {c:i for i,c in enumerate(sorted(cats.unique()))}
            colors = [palette[cat_map[c] % len(palette)] for c in cats]
            fig.add_trace(go.Scatter(
                x=df_vis["pc1"],
                y=df_vis["pc2"],
                mode="markers",
                marker=dict(size=9, color=colors, line=dict(width=0.3, color="rgba(255,255,255,0.04)")),
                text=df_vis["food_name"],
                hovertemplate="<b>%{text}</b><br>Protein: %{customdata[0]} g<br>Carbs: %{customdata[1]} g<br>Fat: %{customdata[2]} g<extra></extra>",
                customdata=df_vis[["protein","carbs","fat"]].values
            ))
        else:
            # color by cluster
            for cl in sorted(df_vis["cluster"].unique()):
                sub = df_vis[df_vis["cluster"]==cl]
                fig.add_trace(go.Scatter(
                    x=sub["pc1"], y=sub["pc2"],
                    mode="markers",
                    marker=dict(size=10, color=palette[cl % len(palette)], line=dict(width=0.4, color="rgba(255,255,255,0.04)")),
                    name=f"Cluster {cl}",
                    text=sub["food_name"],
                    hovertemplate="<b>%{text}</b><br>Protein: %{customdata[0]} g<br>Carbs: %{customdata[1]} g<br>Fat: %{customdata[2]} g<extra></extra>",
                    customdata=sub[["protein","carbs","fat"]].values
                ))

        # cluster centroids
        for i, c in enumerate(centers_pca):
            fig.add_trace(go.Scatter(
                x=[c[0]], y=[c[1]],
                mode="markers+text",
                marker=dict(size=20, symbol="star", color=palette[i % len(palette)], line=dict(width=2, color="white")),
                text=[f"Cluster {i}"],
                textposition="top center",
                hoverinfo="text"
            ))

        # highlight user's foods (diamond white)
        user_foods = list(food_log.keys())
        user_plot = df_vis[df_vis["food_name"].isin(user_foods)]
        if not user_plot.empty:
            fig.add_trace(go.Scatter(
                x=user_plot["pc1"], y=user_plot["pc2"],
                mode="markers+text",
                marker=dict(size=14, symbol="diamond", color="white", line=dict(width=2, color="#111111")),
                text=user_plot["food_name"],
                textposition="bottom right",
                name="Your foods",
                hovertemplate="<b>%{text}</b><br>Protein: %{customdata[0]} g<br>Carbs: %{customdata[1]} g<br>Fat: %{customdata[2]} g<extra></extra>",
                customdata=user_plot[["protein","carbs","fat"]].values
            ))

        # highlight single food if selected
        if chosen_food:
            hf = df_vis[df_vis["food_name"]==chosen_food]
            if not hf.empty:
                fig.add_trace(go.Scatter(
                    x=hf["pc1"], y=hf["pc2"],
                    mode="markers+text",
                    marker=dict(size=16, symbol="star", color="magenta", line=dict(width=2, color="white")),
                    text=hf["food_name"],
                    textposition="top center",
                    name=f"Highlight: {chosen_food}"
                ))

        fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=40, b=10),
            height=700,
        )

        st.plotly_chart(fig, config={"responsive": True}, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Right column: Macro-space scatter & cluster inspector ----------
    with right_col:
        st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
        st.subheader("Macro Space — Protein vs Carbs (Right view)")

        fig2 = px.scatter(
            df_vis,
            x="carbs",
            y="protein",
            color=df_vis["cluster"].astype(str) if color_by=="cluster" else df_vis["category"].astype(str),
            hover_data=["food_name", "fat", "calories"],
            labels={"carbs":"Carbs (g)","protein":"Protein (g)"},
            title="Protein vs Carbs"
        )
        # highlight user's foods visually
        if not user_plot.empty:
            fig2.add_trace(go.Scatter(
                x=user_plot["carbs"], y=user_plot["protein"],
                mode="markers+text",
                marker=dict(size=12, color="white", symbol="diamond", line=dict(width=1, color="#111111")),
                text=user_plot["food_name"],
                textposition="top right",
                name="Your foods"
            ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig2, config={"responsive": True}, width="stretch")

        # cluster inspector summary
        st.markdown("### Cluster Inspector")
        # show simple descriptions based on centroids (unscaled)
        inspector_rows = []
        for i, center in enumerate(centers_unscaled):
            cname = name_cluster_from_centroid(center, FEATURES)
            inspector_rows.append({
                "cluster": i,
                "name": cname,
                "protein": round(center[0],1),
                "carbs": round(center[1],1),
                "fat": round(center[2],1)
            })
        st.table(pd.DataFrame(inspector_rows).set_index("cluster"))

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Bottom: detailed tables, user breakdown, download ----------
st.markdown("---")
st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
st.subheader("Cluster Details & Your Food Breakdown")

# top foods per cluster selector
sel_cluster = st.selectbox("Inspect cluster", options=sorted(df_vis["cluster"].unique().tolist()))
cluster_table = df_vis[df_vis["cluster"]==sel_cluster][["food_name","category","calories","protein","carbs","fat"]].sort_values(by="protein", ascending=False).reset_index(drop=True)
st.dataframe(cluster_table)

# user's cluster breakdown
user_cluster_counts = {}
for f in user_foods:
    m = df_vis[df_vis["food_name"]==f]
    if not m.empty:
        cl = int(m["cluster"].iloc[0])
        user_cluster_counts[cl] = user_cluster_counts.get(cl, 0) + 1

if user_cluster_counts:
    breakdown_df = pd.DataFrame([
        {"cluster": c, "count": user_cluster_counts[c], "cluster_name": name_cluster_from_centroid(centers_unscaled[c], FEATURES)}
        for c in sorted(user_cluster_counts.keys())
    ])
    st.markdown("Your logged foods — cluster counts")
    st.table(breakdown_df.set_index("cluster"))
    most = max(user_cluster_counts.items(), key=lambda x:x[1])[0]
    st.success(f"Your foods are mostly from cluster #{most} — {name_cluster_from_centroid(centers_unscaled[most], FEATURES)}")
else:
    st.info("No logged foods matched (try increasing sample% to 100%).")

# download CSV of assignments
assign_df = assignments_df_for_download(df_vis)
csv = assign_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download cluster assignments (CSV)", data=csv, file_name="snp_cluster_assignments.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- final little UX flourish ----------
if st.session_state.get("_visited_clusters") is None:
    st.session_state["_visited_clusters"] = True
    st.balloons()
