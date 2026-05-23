import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    @staticmethod
    def preprocess_for_pca(df: pd.DataFrame, target_col=None, imputation_strategy='mean'):
        """
        Cleans data and standardizes it. Returns cleaned features, targets, and scaler.
        """
        # Separate target if provided (we don't run PCA on the target/labels)
        targets = None
        if target_col and target_col in df.columns:
            targets = df[target_col]
            df = df.drop(columns=[target_col])

        # Filter strictly numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        # Drop constant columns (variance = 0)
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

        # Imputation
        if imputation_strategy == 'mean':
            numeric_df = numeric_df.fillna(numeric_df.mean())
        elif imputation_strategy == 'median':
            numeric_df = numeric_df.fillna(numeric_df.median())
        else:
            numeric_df = numeric_df.dropna()

        # Feature Scaling (Crucial for PCA)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=numeric_df.index)

        return scaled_df, targets, numeric_df.columns