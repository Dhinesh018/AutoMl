def build_model_selection_prompt(dataset_profile: dict, candidate_models: list):
    prompt = f"""
You are an AutoML system.

Dataset summary:
- Rows: {dataset_profile['num_rows']}
- Columns: {dataset_profile['num_columns']}
- Numeric features: {dataset_profile['num_numeric_features']}
- Categorical features: {dataset_profile['num_categorical_features']}
- Missing values: {dataset_profile['has_missing_values']}

Candidate models:
{", ".join(candidate_models)}

Task:
Explain which models should be trained and why.
Explain which models should be skipped and why.
Respond clearly and concisely.
"""
    return prompt.strip()
