import numpy as np
import pandas as pd

class SVDPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None     # Loadings (V)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the model with X and applies the dimensionality reduction using SVD.
        """
        # Convert to numpy array
        X_arr = X.values
        n_samples, n_features = X_arr.shape

        # 1. Mean Centering (Crucial Step)
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # 2. SVD Factorization
        # full_matrices=False ensures we only compute the required singular vectors
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 3. Handle Component Selection
        max_components = min(n_samples, n_features)
        if self.n_components is None:
            self.n_components = max_components
        elif self.n_components > max_components:
            self.n_components = max_components

        # Store Loadings (V transposed to match sklearn format)
        self.components_ = Vt[:self.n_components, :]
        self.singular_values_ = S[:self.n_components]

        # 4. Calculate Explained Variance
        # Variance = (Singular_Values^2) / (n - 1)
        variances = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = variances[:self.n_components]
        total_variance = np.sum(variances)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        # 5. Project Data: Z = U * Sigma
        # We only use the top 'n_components' columns
        U_reduced = U[:, :self.n_components]
        S_reduced = np.diag(S[:self.n_components])
        Z = np.dot(U_reduced, S_reduced)

        # Return as DataFrame for easy downstream visualization
        columns = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(Z, columns=columns, index=X.index)