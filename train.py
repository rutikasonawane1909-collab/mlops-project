
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create model folder
os.makedirs("model", exist_ok=True)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Set MLflow experiment
mlflow.set_experiment("Iris-Classification")

# Start MLflow run
with mlflow.start_run():

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Save model locally
    joblib.dump(model, "model/iris_model.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    # Display results
    print("Model training completed successfully!")
    print("Model accuracy:", round(accuracy * 100, 2), "%")
    print("Model saved at: model/iris_model.pkl")
    print("MLflow experiment: Iris-Classification")
