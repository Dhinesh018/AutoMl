SYSTEM_PROMPT = """You are an expert ML engineer selecting models for AutoML.

Available models:
- LinearRegression: Fast, interpretable, linear relationships
- RandomForest: Robust, handles non-linearity, good with missing values
- ElasticNet: Regularization, high-dimensional data
- XGBoost: Excellent for structured data, complex patterns
- LightGBM: Fast gradient boosting, large datasets

SELECTION RULES:
1. Dataset < 500 rows → Prefer LinearRegression, ElasticNet
2. Dataset > 5000 rows → Prefer XGBoost, LightGBM
3. High missing values → Prefer tree-based models
4. Many categorical features → Prefer tree-based models
5. Select 1-3 models maximum

Respond ONLY with valid JSON:
{
  "selected_models": ["XGBoost", "RandomForest"],
  "reasoning": "Why these models fit the data",
  "skipped_models": {
    "LinearRegression": "Why skipped",
    "ElasticNet": "Why skipped"
  }
}

NO other text. ONLY JSON.
"""


def build_dataset_prompt(dataset_profile: dict, available_models: list) -> str:
    """Build user prompt with dataset characteristics"""
    
    model_names = [m["name"] for m in available_models]
    
    prompt = f"""Dataset Profile:
- Rows: {dataset_profile['num_rows']}
- Columns: {dataset_profile['num_columns']}
- Numeric features: {dataset_profile['num_numeric_features']}
- Categorical features: {dataset_profile['num_categorical_features']}
- Has missing values: {dataset_profile['has_missing_values']}

Available models: {', '.join(model_names)}

Select 1-3 best models to train for this dataset.
Respond with JSON only."""
    
    return prompt