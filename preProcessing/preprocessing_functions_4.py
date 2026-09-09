import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

def normalize_data(df, columns, method='standard'):
    """
    Normalizes numerical features.
    method: 'standard' (Z-score) or 'minmax' (0 to 1 scaling)
    """
    df_scaled = df.copy()
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    print(f"Normalized columns {columns} using {method.upper()} Scaling.")
    return df_scaled, scaler

def handle_outliers_iqr(df, columns, action='cap'):
    """
    Detects and handles outliers using the Interquartile Range (IQR) method.
    action: 'cap' (winsorize to bounds) or 'remove' (drop outlier rows)
    """
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()
        print(f"Column '{col}': Found {outliers_count} outliers [Bounds: {lower_bound:.2f}, {upper_bound:.2f}]")
        
        if action == 'cap':
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])
        elif action == 'remove':
            df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
            
    return df_out

def bin_feature(df, column, bins, labels):
    """
    Discretizes a continuous numerical column into categorical bins.
    """
    df_binned = df.copy()
    binned_col_name = f"{column}_Group"
    df_binned[binned_col_name] = pd.cut(df_binned[column], bins=bins, labels=labels, include_lowest=True)
    print(f"Created binned feature '{binned_col_name}'.")
    return df_binned

def encode_categorical(df, categorical_cols, method='onehot'):
    """
    Encodes categorical features.
    method: 'onehot' (pd.get_dummies) or 'label' (LabelEncoder)
    """
    df_encoded = df.copy()
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, dtype=int)
    elif method == 'label':
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    print(f"Encoded {len(categorical_cols)} categorical columns using {method.upper()} method.")
    return df_encoded

def apply_pca(df, numerical_cols, n_components=2):
    """
    Applies Principal Component Analysis (PCA) to reduce dataset dimensionality.
    """
    # Impute missing values (NaN) with column medians directly inside the function
    data_to_scale = df[numerical_cols].fillna(df[numerical_cols].median())

    # Features must be scaled before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=df.index)
    
    print(f"Applied PCA: Reduced {len(numerical_cols)} features down to {n_components} components.")
    print(f"Explained Variance Ratios: {pca.explained_variance_ratio_}")
    print(f"Total Retained Variance: {sum(pca.explained_variance_ratio_):.2%}")
    
    return df_pca, pca