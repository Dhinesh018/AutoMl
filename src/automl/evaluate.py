from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred)
    }
