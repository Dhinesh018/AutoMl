import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocess data: encode categoricals, split, and scale
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        test_size: Test set proportion (default 0.2)
        random_state: Random seed (default 42)
    
    Returns:
        X_train, X_test, y_train, y_test (all preprocessed)
    """
    df = df.copy()
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Store label encoders
    label_encoders = {}
    
    # Identify column types
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"📊 Preprocessing: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical columns")
    
    # Encode categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        # Fill missing values
        X[col] = X[col].fillna('missing')
        # Encode
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    if numeric_columns:
        X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    print(f"✅ Preprocessed: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test