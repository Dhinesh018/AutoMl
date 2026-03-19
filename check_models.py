from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri='http://mlflow:5000')
versions = client.search_model_versions('name="llm_automl_tabular_model"')

print(f'Found {len(versions)} model versions')
for v in versions:
    print(f'Version {v.version}, Stage: {v.current_stage}')