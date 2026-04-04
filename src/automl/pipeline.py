"""
Production ML Pipeline with categorical + numeric support
Uses ColumnTransformer for mixed data types
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


class ProductionPipeline:
    """
    Handles:
    - Numeric features → Imputation + Scaling
    - Categorical features → Imputation + OneHotEncoding
    Returns sklearn Pipeline that can be saved with MLflow
    """
    
    @staticmethod
    def build(X: pd.DataFrame, model):
        """
        Build complete pipeline (preprocessing + model)
        
        Args:
            X: Training features
            model: sklearn-compatible model (RF, XGB, LGB)
        
        Returns:
            sklearn Pipeline
        """
        # Detect feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        # Numeric pipeline
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Categorical pipeline
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        return pipeline
    
    @staticmethod
    def get_feature_metadata(X: pd.DataFrame, target: str):
        """
        Extract metadata for API
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'features': X.columns.tolist(),
            'target': target,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'num_features': len(X.columns)
        }