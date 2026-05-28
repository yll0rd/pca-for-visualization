import streamlit as st
import pandas as pd
from data_pipeline.cleaner import DataCleaner
from engine.pca_via_svd import SVDPCA
from visualizations.plot_manager import VizManager
from sklearn.cluster import KMeans
import json

# --- Page Config ---
st.set_page_config(page_title="PCA Cluster Platform", layout="wide", page_icon="🔬")

# --- Session State Management ---
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

# --- Sidebar: Data Loading ---
st.sidebar.title("🔬 PCA & Clustering Engine")
st.sidebar.header("1. Data Ingestion")

upload_file = st.sidebar.file_uploader("Upload CSV/Excel/JSON", type=['csv', 'xlsx', 'json'])

if upload_file:
    # Basic routing based on extension
    if upload_file.name.endswith('.csv'):
        st.session_state['raw_data'] = pd.read_csv(upload_file)
    elif upload_file.name.endswith('.xlsx'):
        st.session_state['raw_data'] = pd.read_excel(upload_file)
    elif upload_file.name.endswith('.json'):
        st.session_state['raw_data'] = pd.read_json(upload_file)

# --- Main Dashboard ---
if st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data']
    
    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Preview", "🧹 Cleaning & Preprocessing", 
        "🧮 PCA Analysis", "📈 Visualizations", "💾 Export Results"
    ])

    # Tab 1: Preview
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(15), use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

    # Tab 2: Cleaning
    with tab2:
        st.subheader("Configurable Cleaning Pipeline")
        target_col = st.selectbox("Select Target/Label Column (Optional - Excluded from PCA)", [None] + list(df.columns))
        imputation = st.radio("Missing Value Handling", ["mean", "median", "drop"])
        
        if st.button("Apply Cleaning & Scaling"):
            with st.spinner("Standardizing data..."):
                scaled_df, targets, features = DataCleaner.preprocess_for_pca(df, target_col, imputation)
                st.session_state['scaled_df'] = scaled_df
                st.session_state['targets'] = targets
                st.session_state['features'] = features
            st.success("Data successfully standardized (μ=0, σ=1) and ready for PCA.")
            st.dataframe(scaled_df.head())

    # Tab 3 & 4: PCA & Visualization
    if 'scaled_df' in st.session_state:
        scaled_df = st.session_state['scaled_df']
        targets = st.session_state['targets']
        
        with tab3:
            st.subheader("Singular Value Decomposition (SVD) Settings")
            n_features = scaled_df.shape[1]
            n_components = st.slider("Select Number of Principal Components", min_value=1, max_value=n_features, value=min(3, n_features))
            
            # Execute Custom SVD PCA
            pca = SVDPCA(n_components=n_components)
            pca_transformed = pca.fit_transform(scaled_df)
            st.session_state['pca_result'] = pca_transformed
            
            st.markdown("### Explained Variance Analysis")
            fig_var = VizManager.plot_variance(pca.explained_variance_ratio_)
            st.plotly_chart(fig_var, use_container_width=True)

            # Mathematical Explainability
            st.info("""
            **Why is this mathematically robust?** 
            Instead of computing the covariance matrix, our engine factored the centered data matrix directly:
            $X = U\\Sigma V^T$. 
            This prevents catastrophic loss of precision. The components shown above correspond to the right singular vectors ($V$).
            """)

        with tab4:
            st.subheader("Dimensional Projection")
            if n_components > 3:
                st.warning("You selected >3 components. The visualizations below default to the first 3 components (highest variance), but your exported data will contain all calculated components.")
            
            # Clustering Bonus
            plot_targets = targets if targets is not None else None
            run_clustering = st.checkbox("Apply K-Means Clustering on PCA Space")
            if run_clustering:
                n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters)
                plot_targets = kmeans.fit_predict(pca_transformed)

            viz_dims = min(n_components, 3)
            fig_scatter = VizManager.plot_pca_scatter(pca_transformed, viz_dims, plot_targets)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 5: Export
    with tab5:
        st.subheader("Download Pipeline Artifacts")
        if 'pca_result' in st.session_state:
            csv = st.session_state['pca_result'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download PCA Transformed Data (CSV)",
                data=csv,
                file_name='pca_transformed.csv',
                mime='text/csv',
            )
else:
    st.info("👈 Please upload a dataset in the sidebar to begin.")