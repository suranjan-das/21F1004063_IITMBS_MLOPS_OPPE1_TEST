# scripts/evaluate_stock_model.py
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc

# ==============================
# 🔧 Configurations
# ==============================
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
MODEL_NAME = os.environ.get("MODEL_NAME", "StockMovementModel")

# Default to test.csv for evaluation
DATA_PATH = os.environ.get("DATA_PATH", "test.csv")
REPORT_PATH = os.environ.get("REPORT_PATH", "report.md")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.6"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

# ==============================
# 📊 Load Data
# ==============================
def load_data(path):
    df = pd.read_csv(path)

    if "target" not in df.columns:
        print("ERROR: target column missing in dataset.", file=sys.stderr)
        sys.exit(2)

    X = df[["rolling_avg_10", "volume_sum_10"]]
    y = df["target"]
    return X, y


# ==============================
# 🧠 Fetch and Load Best Model
# ==============================
def load_best_model(model_name):
    print(f"🔍 Searching best version of model '{model_name}' in MLflow registry...")

    try:
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print(f"ERROR: No versions found for model '{model_name}'.", file=sys.stderr)
            sys.exit(2)

        scored_versions = []
        for v in versions:
            run_id = v.run_id
            try:
                metrics = client.get_run(run_id).data.metrics
                acc = metrics.get("accuracy", 0.0)
                scored_versions.append((v.version, acc))
            except Exception as e:
                print(f"⚠️ Skipping version {v.version} due to error: {e}")

        if not scored_versions:
            print("ERROR: No model versions with accuracy metric found.", file=sys.stderr)
            sys.exit(2)

        best_version, best_acc = max(scored_versions, key=lambda x: x[1])
        model_uri = f"models:/{model_name}/{best_version}"

        print(f"🏆 Best model found: version {best_version} (accuracy={best_acc:.4f})")
        print(f"📦 Loading model from URI: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        return model, best_version, best_acc

    except Exception as e:
        print(f"ERROR: Failed to load best model from MLflow registry: {e}", file=sys.stderr)
        sys.exit(2)


# ==============================
# 🧪 Evaluate Model
# ==============================
def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cls_report = classification_report(y, preds, output_dict=False)
    return acc, cls_report


# ==============================
# 📝 Write Markdown Report
# ==============================
def write_report(acc, cls_report, best_version, best_acc):
    md = [
        "# Stock Movement Test Evaluation Report",
        "",
        f"**Evaluated Model Version:** {best_version}",
        f"**Best Logged Accuracy (Training):** {best_acc:.4f}",
        "",
        f"**Accuracy on Test Data:** {acc:.4f}",
        "",
        "## Classification Report",
        "",
        f"```\n{cls_report}\n```",
        "",
        f"**Accuracy Threshold:** {ACCURACY_THRESHOLD}",
        "",
    ]

    if acc >= ACCURACY_THRESHOLD:
        md.append("✅ **Sanity check passed (Test Accuracy above threshold)**")
        exit_code = 0
    else:
        md.append("❌ **Sanity check failed (Test Accuracy below threshold)**")
        exit_code = 3

    report = "\n".join(md)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)
    sys.exit(exit_code)


# ==============================
# 🚀 Main Entry Point
# ==============================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: test data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(2)

    X, y = load_data(DATA_PATH)
    model, best_version, best_acc = load_best_model(MODEL_NAME)

    acc, cls_report = evaluate(model, X, y)
    write_report(acc, cls_report, best_version, best_acc)


if __name__ == "__main__":
    main()
