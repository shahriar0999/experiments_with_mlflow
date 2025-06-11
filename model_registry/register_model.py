from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

run_id = "7a26da4d5d1d4bf3bc1e646050aead1b"

model_path = "file:///A:/Python%20For%20Accounting/Agent/New%20folder/mlruns/747597075917670187/7a26da4d5d1d4bf3bc1e646050aead1b/artifacts/RandomForestClassifier"

model_uri = f"runs:/{run_id}/{model_path}"

model_name = "diabetes_rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# add description
client.update_model_version(
    name=model_name,
    version=result.version,
    description="Random Forest Classifier for diabetes prediction with v3"

)

# add tags
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="author",
    value="Shahriar Kabir"
)

print(f"Model {model_name} version {result.version} registered successfully.")
