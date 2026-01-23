def mock_llm_response(dataset_profile: dict, candidate_models: list):
    response = []

    if dataset_profile["num_rows"] < 200:
        response.append(
            "The dataset is small, so simpler models like LinearRegression and ElasticNet are preferred."
        )
        response.append(
            "Tree-based boosting models may overfit due to limited data."
        )
    else:
        response.append(
            "The dataset is sufficiently large to support complex models like XGBoost and LightGBM."
        )

    response.append(
        "Multiple models will be evaluated, and the best-performing one will be selected using validation metrics."
    )

    return "\n".join(response)
