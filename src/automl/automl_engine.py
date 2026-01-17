import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression , ElasticNet

from src.automl.evaluate import evaluate


MODEL_REGISTRY = {
    "RandomForest": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "ElasticNet": ElasticNet,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor
}



def run_automl(models_config, X_train, X_test, y_train, y_test):
    best_model = None
    best_score = float("-inf")
    best_name = None

    for model_cfg in models_config:
        model_name = model_cfg["name"]
        params = model_cfg.get("params", {})

        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(**params)

        with mlflow.start_run(run_name=model_name ,nested=True) :
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = evaluate(y_test, preds)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_params(params)
            mlflow.sklearn.log_model(model, artifact_path="model")

            if metrics["r2"] > best_score:
                best_score = metrics["r2"]
                best_model = model
                best_name = model_name

    return best_name, best_score, best_model
