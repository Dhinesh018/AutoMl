from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri='http://mlflow:5000')

# Get latest version
versions = client.search_model_versions('name="llm_automl_tabular_model"')
if versions:
    latest = max([int(v.version) for v in versions])
    
    # Promote to Production
    client.transition_model_version_stage(
        name='llm_automl_tabular_model',
        version=latest,
        stage='Production'
    )
    print(f'✅ Promoted version {latest} to Production!')
else:
    print('❌ No models found!')