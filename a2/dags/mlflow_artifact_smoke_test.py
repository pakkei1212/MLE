# dags/mlflow_cleanup_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit_risk_model")
DELETE_MODEL_VERSIONS = True
DELETE_REGISTERED_MODEL = True

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

def delete_mlflow_model(**context):
    print(f"[CLEANUP] Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # --- FIX: use search_registered_models() for compatibility ---
    all_models = [m.name for m in client.search_registered_models()]
    print(f"[CLEANUP] Registered models in server: {all_models}")

    if MODEL_NAME not in all_models:
        print(f"[CLEANUP] Model '{MODEL_NAME}' not found. Nothing to delete.")
        return

    if DELETE_MODEL_VERSIONS:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            print(f"[CLEANUP] No versions found for {MODEL_NAME}.")
        else:
            print(f"[CLEANUP] Deleting {len(versions)} versions for {MODEL_NAME}...")
            for v in versions:
                print(f"  - Deleting version {v.version}")
                client.delete_model_version(name=v.name, version=v.version)

    if DELETE_REGISTERED_MODEL:
        print(f"[CLEANUP] Deleting registered model '{MODEL_NAME}'")
        client.delete_registered_model(name=MODEL_NAME)
        print(f"[CLEANUP] ✅ Model '{MODEL_NAME}' fully removed from registry.")
    else:
        print(f"[CLEANUP] Skipped deleting registered model entry for '{MODEL_NAME}'")


with DAG(
    dag_id="mlflow_cleanup_dag",
    description="Delete models or versions from MLflow registry",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlflow", "cleanup"],
) as dag:

    cleanup_task = PythonOperator(
        task_id="delete_mlflow_model_task",
        python_callable=delete_mlflow_model,
        provide_context=True,
    )

    cleanup_task
