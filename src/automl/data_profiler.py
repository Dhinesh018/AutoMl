import pandas as pd
import numpy as np

def profile_dataset(df, target_column):
    """
    Create a comprehensive dataset profile for LLM
    """
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Check for missing values
    has_missing = df.isnull().any().any()
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
    
    # Build profile with ALL required fields
    profile = {
        # Basic info
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "num_features": len(X.columns),
        "num_numeric_features": len(numeric_cols),
        "num_categorical_features": len(categorical_cols),
        
        # Column lists
        "columns": df.columns.tolist(),
        "feature_names": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        
        # Target info
        "target_column": target_column,
        
        # Data types
        "dtypes": df.dtypes.astype(str).to_dict(),
        
        # Missing values
        "has_missing_values": bool(has_missing),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "total_missing_percentage": float(missing_percentage),
    }
    
    # Target statistics
    if y is not None:
        profile["target_type"] = str(y.dtype)
        if np.issubdtype(y.dtype, np.number):
            profile["target_stats"] = {
                "mean": float(y.mean()) if not pd.isna(y.mean()) else 0,
                "std": float(y.std()) if not pd.isna(y.std()) else 0,
                "min": float(y.min()) if not pd.isna(y.min()) else 0,
                "max": float(y.max()) if not pd.isna(y.max()) else 0,
                "median": float(y.median()) if not pd.isna(y.median()) else 0,
            }
        else:
            profile["target_unique_values"] = int(y.nunique())
    
    # Basic stats for numeric features (limit to 10)
    if len(numeric_cols) > 0:
        sample_cols = numeric_cols[:10]
        profile["numeric_stats"] = {}
        for col in sample_cols:
            profile["numeric_stats"][col] = {
                "mean": float(X[col].mean()) if not pd.isna(X[col].mean()) else 0,
                "std": float(X[col].std()) if not pd.isna(X[col].std()) else 0,
                "min": float(X[col].min()) if not pd.isna(X[col].min()) else 0,
                "max": float(X[col].max()) if not pd.isna(X[col].max()) else 0,
                "median": float(X[col].median()) if not pd.isna(X[col].median()) else 0,
            }
    
    # Categorical feature cardinality (limit to 10)
    if len(categorical_cols) > 0:
        sample_cats = categorical_cols[:10]
        profile["categorical_cardinality"] = {
            col: int(X[col].nunique())
            for col in sample_cats
        }
    
    # Data size estimate
    profile["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    
    print(f"📊 Profiled: {len(df)} rows, {len(X.columns)} features ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical), Missing: {missing_percentage:.1f}%")
    
    return profile