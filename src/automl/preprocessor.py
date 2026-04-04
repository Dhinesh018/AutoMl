import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df, target_column, test_size=0.2, random_state=42):
    """
    Minimal preprocessing - just split the data
    Pipeline will handle encoding and scaling
    """
    df = df.copy()
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Log column types
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"📊 Preprocessing: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical columns")
    
    # Just split - NO encoding, NO scaling
    # Pipeline will handle all preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Preprocessed: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test