from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

model_name = "diabetes_rf"
model_version = 3

# Transition the model version to a new stage
new_stage = "Production"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to stage {new_stage} successfully.")