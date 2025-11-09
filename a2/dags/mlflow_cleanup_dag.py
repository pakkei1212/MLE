# dags/mlflow_cleanup_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit_risk_model")
TARGET_VERSION = int(os.getenv("TARGET_VERSION", "3"))  # <-- Only delete this version

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

def delete_specific_model_version(**context):
    print(f"[CLEANUP] Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # List all models
    all_models = [m.name for m in client.search_registered_models()]
    print(f"[CLEANUP] Registered models: {all_models}")

    if MODEL_NAME not in all_models:
        print(f"[CLEANUP] Model '{MODEL_NAME}' not found.")
        return

    # Find the specific version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    target_versions = [v for v in versions if int(v.version) == TARGET_VERSION]

    if not target_versions:
        print(f"[CLEANUP] No version {TARGET_VERSION} found for '{MODEL_NAME}'.")
        return

    print(f"[CLEANUP] Deleting version {TARGET_VERSION} of '{MODEL_NAME}'...")
    client.delete_model_version(name=MODEL_NAME, version=str(TARGET_VERSION))
    print(f"[CLEANUP] ✅ Version {TARGET_VERSION} deleted successfully.")


with DAG(
    dag_id="mlflow_delete_specific_version_dag",
    description="Delete a specific version of a registered model from MLflow",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlflow", "cleanup"],
) as dag:

    delete_version_task = PythonOperator(
        task_id="delete_specific_model_version_task",
        python_callable=delete_specific_model_version,
        provide_context=True,
    )

    delete_version_task
