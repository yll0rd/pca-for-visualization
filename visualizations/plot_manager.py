import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class VizManager:
    @staticmethod
    def plot_variance(explained_variance_ratio):
        """Generates Scree Plot and Cumulative Variance."""
        cumulative_variance = np.cumsum(explained_variance_ratio)
        x_labels = [f"PC{i+1}" for i in range(len(explained_variance_ratio))]

        fig = go.Figure()
        # Bar chart for individual variance
        fig.add_trace(go.Bar(
            x=x_labels, y=explained_variance_ratio,
            name='Individual Variance', marker_color='#3498db'
        ))
        # Line chart for cumulative variance
        fig.add_trace(go.Scatter(
            x=x_labels, y=cumulative_variance,
            name='Cumulative Variance', mode='lines+markers', line=dict(color='#e74c3c', width=3)
        ))
        
        fig.update_layout(
            title="Scree Plot & Cumulative Explained Variance",
            xaxis_title="Principal Components",
            yaxis_title="Variance Ratio",
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_pca_scatter(pca_df, n_dims, targets=None):
        """Dynamically routes to 1D, 2D, or 3D scatter plots."""
        color_kwarg = {'color': targets} if targets is not None else {}

        if n_dims == 1:
            # 1D: Strip plot / Histogram hybrid
            pca_df['Y_Zero'] = 0  # Dummy axis
            fig = px.scatter(pca_df, x='PC1', y='Y_Zero', title="1D PCA Distribution", **color_kwarg)
            fig.update_yaxes(showticklabels=False, title="")
            
        elif n_dims == 2:
            fig = px.scatter(pca_df, x='PC1', y='PC2', title="2D PCA Scatter", hover_data=pca_df.columns, **color_kwarg)
            
        else: # n_dims >= 3
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', title="3D PCA Scatter", **color_kwarg)
            
        fig.update_layout(template="plotly_white")
        return fig