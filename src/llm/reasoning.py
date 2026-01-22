def reason_about_models(dataset_profile: dict, candidate_models: list):
    reasoning = []
    selected_models = []

    n_rows = dataset_profile["num_rows"]
    has_categorical = dataset_profile["num_categorical_features"] > 0
    has_missing = dataset_profile["has_missing_values"]

    for model in candidate_models:
        if model in ["XGBoost", "LightGBM"] and n_rows < 200:
            reasoning.append(
                f"Skipped {model}: dataset too small ({n_rows} rows)"
            )
            continue

        if model == "LinearRegression" and has_categorical:
            reasoning.append(
                "Skipped LinearRegression: categorical features present"
            )
            continue

        reasoning.append(f"Selected {model}: suitable for dataset characteristics")
        selected_models.append(model)

    return {
        "selected_models": selected_models,
        "reasoning": reasoning
    }
