import pandas as pd


def profile_dataset(df: pd.DataFrame, target_column: str) -> dict:
    profile = {}

    profile["num_rows"] = df.shape[0]
    profile["num_columns"] = df.shape[1]

    feature_cols = df.drop(columns=[target_column])

    numeric_features = feature_cols.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = feature_cols.select_dtypes(include=["object", "category"]).columns.tolist()

    profile["num_numeric_features"] = len(numeric_features)
    profile["num_categorical_features"] = len(categorical_features)
    profile["numeric_features"] = numeric_features
    profile["categorical_features"] = categorical_features

    missing = df.isnull().sum()
    missing_features = missing[missing > 0].to_dict()

    profile["missing_features"] = missing_features
    profile["has_missing_values"] = len(missing_features) > 0

    return profile
