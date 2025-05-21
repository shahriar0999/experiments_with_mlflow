import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv")

# split into features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set the experiment name
mlflow.set_experiment("experiment_with_random_forest")
# experment tracking using mlflow

n_estimators = 50
max_depth = 5


with mlflow.start_run():

    # train model with RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # calculate accuracy
    accuarcy_score = accuracy_score(y_test, y_pred)

    # log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # log metrics
    mlflow.log_metric("accuracy", accuarcy_score)

    # log the model
    mlflow.sklearn.log_model(model, "RandomForestClassifier")

    # log the dataset
    train_df = X_train
    train_df["Outcome"] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "train_data")

    test_df = X_test
    test_df["Outcome"] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "test_data")

    # set tags
    mlflow.set_tag("author", "Shahriar Kabir")
    mlflow.set_tag("role", "Data Scientist")

    # save code
    mlflow.log_artifact(__file__)

    print(f"Accuracy: {accuarcy_score}")