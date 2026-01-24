def mock_llm_decision(dataset_profile: dict, candidate_models: list):
    selected = []
    skipped = []

    n_rows = dataset_profile["num_rows"]

    for model in candidate_models:
        name = model["name"]

        if n_rows < 200 and name in ["XGBoost", "LightGBM"]:
            skipped.append(
                f"Skipped {name}: dataset too small ({n_rows} rows)"
            )
        else:
            selected.append(model)

    response = {
        "selected_models": selected,
        "skipped_models": skipped,
        "summary": (
            "Model selection was adjusted based on dataset size and complexity."
        )
    }

    return response
