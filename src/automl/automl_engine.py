import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from .pipeline import ProductionPipeline


MODEL_REGISTRY = {
    "RandomForest": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "ElasticNet": ElasticNet,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor
}

MODEL_NAME = "llm_automl_tabular_model"


def run_automl(models_to_run, X_train, X_test, y_train, y_test):
    """
    Train multiple models with production pipeline
    Returns best model pipeline
    """
    results = {}

    for model_config in models_to_run:
        model_name = model_config['name']

        # Get model class
        model_class = model_config.get('class') or MODEL_REGISTRY.get(model_name)

        if model_class is None:
            raise ValueError(f"Model class missing for {model_name}")

        print(f"🔹 Training {model_name}...")

        # Create model instance
        model = model_class(**model_config.get('params', {}))

        # Build FULL PIPELINE (preprocessing + model)
        pipeline = ProductionPipeline.build(X_train, model)

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[model_name] = {
            'pipeline': pipeline,
            'r2': r2,
            'rmse': rmse
        }

        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # Select best
    best_name = max(results, key=lambda x: results[x]['r2'])
    best_result = results[best_name]

    print(f"✅ Best: {best_name}")

    return best_name, best_result['r2'], best_result['pipeline']