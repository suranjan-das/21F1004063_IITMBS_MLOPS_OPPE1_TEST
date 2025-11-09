import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn

# ==============================
# 🔧 MLflow Setup
# ==============================
mlflow.set_tracking_uri("http://127.0.0.1:8100")  # Change if your MLflow runs elsewhere
client = MlflowClient(mlflow.get_tracking_uri())

# Create or set experiment
mlflow.set_experiment("Mlflow: Stock Movement Prediction")

# ==============================
# 📊 Load and Prepare Data
# ==============================
DATA_PATH = "train.csv"

df = pd.read_csv(DATA_PATH)

# Define features and target
feature_cols = ["rolling_avg_10", "volume_sum_10"]
target_col = "target"

X = df[feature_cols]
y = df[target_col]

# Split train/test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================
# 🤖 Train Model
# ==============================
params = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 500,
    "random_state": 42
}

model = LogisticRegression(**params)
model.fit(X_train, y_train)

# ==============================
# 📈 Evaluate Model
# ==============================
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n✅ Logistic Regression Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save locally
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

# ==============================
# 🧠 Log Run to MLflow
# ==============================
with mlflow.start_run():
    # Log model parameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Set tags for tracking
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("developer", "Suranjan Das")

    # Infer model signature (for model input/output schema)
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="StockMovementModel"
    )

    print("\nModel and metrics successfully logged to MLflow!")

# ==============================
# ✅ Verification
# ==============================
print("\nAll experiments:")
for exp in client.search_experiments():
    print(f"Name: {exp.name}, ID: {exp.experiment_id}")
