from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from pathlib import Path
import os, re
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional

# -----------------------------
# Configurable knobs
# -----------------------------
MIN_MONTHS_REQUIRED = 14
GOLD_CLICK_BASE = Path("/opt/airflow/scripts/datamart/gold/features/clickstream")
GOLD_ATTR_BASE  = Path("/opt/airflow/scripts/datamart/gold/features/attributes")
GOLD_FIN_BASE   = Path("/opt/airflow/scripts/datamart/gold/features/financials")
LABEL_BASE      = Path("/opt/airflow/scripts/datamart/gold/labels")
PROD_DIR = "/opt/airflow/scripts/model_bank/production/best"
PREDICTIONS_BASE = Path("/opt/airflow/scripts/datamart/gold/model_predictions")
MONITOR_OUT_DIR  = "datamart/gold/model_monitoring"

# -----------------------------
# Helpers
# -----------------------------
def _parse_train_date_from_version(version: str):
    """Parse ..._YYYY_MM_DD suffix into a date."""
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})$', version or "")
    if not m:
        return None
    y, mm, dd = map(int, m.groups())
    try:
        return datetime(y, mm, dd).date()
    except Exception:
        return None

def _last_training_date_from_mlflow(
    model_name: str = "credit_risk_model",
    stage: str = "Production",
) -> Optional[date]:
    """
    Return the train_date (YYYY-MM-DD) of the most recently promoted model in `stage`.
    Reads run tag `train_date`, with a fallback to JSON in model-version description.
    """
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()

        # Preferred: API that understands stages
        latest = []
        try:
            latest = client.get_latest_versions(name=model_name, stages=[stage]) or []
        except Exception as e:
            print(f"[MONTHLY_GATE] get_latest_versions failed ({type(e).__name__}: {e})")

        mv = None
        if latest:
            # pick the highest version just to be explicit
            mv = max(latest, key=lambda v: int(v.version))
        else:
            # Fallback: fetch all, filter by current_stage client-side
            all_mvs = client.search_model_versions(f"name = '{model_name}'")
            mvs_in_stage = [v for v in all_mvs if getattr(v, "current_stage", None) == stage]
            if not mvs_in_stage:
                print(f"[MONTHLY_GATE] No versions in stage {stage} for '{model_name}'.")
                return None
            mv = max(mvs_in_stage, key=lambda v: int(v.version))

        print(f"[MONTHLY_GATE] Using {model_name} v{mv.version} in {stage}; run_id={mv.run_id}")

        # Pull source run and read train_date tag (set during training)
        run = client.get_run(mv.run_id)
        date_str = (run.data.tags.get("train_date") or "").strip()

        # Fallback: if you saved JSON in model-version description during promotion
        if not date_str and getattr(mv, "description", None):
            try:
                desc = json.loads(mv.description)
                if isinstance(desc, dict) and "train_date" in desc:
                    date_str = str(desc["train_date"]).strip()
            except Exception:
                pass

        if not date_str:
            print(f"[MONTHLY_GATE] No 'train_date' on run {mv.run_id} or model description.")
            return None

        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        print(f"[MONTHLY_GATE] Latest production train_date = {dt}")
        return dt

    except Exception as e:
        print(f"[MONTHLY_GATE] Error reading MLflow production model: {e}")
        return None

# --- update your monthly_need_training to use MLflow instead of local file ---
def monthly_need_training(**ctx) -> bool:
    """
    Run monthly training only if:
      - no production model yet, or
      - (current ds - last training date) >= 3 calendar months
    """
    ds_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()

    # Optional: allow overriding via env vars
    model_name = os.getenv("MLFLOW_MODEL_NAME", "credit_risk_model")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    last_dt = _last_training_date_from_mlflow(model_name=model_name, stage=stage)
    if last_dt is None:
        print("[MONTHLY_GATE] No last training date from MLflow -> run monthly training")
        return True

    delta = relativedelta(ds_date, last_dt)
    months_apart = delta.years * 12 + delta.months
    should_run = months_apart >= 3
    print(f"[MONTHLY_GATE] ds={ds_date}, last={last_dt}, months_apart={months_apart} -> run={should_run}")
    return should_run

def _month_str(dt: datetime) -> str:
    """Return month token in YYYY-MM (dash) form."""
    return dt.strftime("%Y-%m")

def _part_exists(base: Path, ym_dash: str) -> bool:
    """
    Robust month presence check across common patterns:
      base/YYYY-MM
      *YYYY-MM*
      *YYYY_MM*
      *YYYYMM*
      and day-level variants within filenames.
    """
    ym_dash = (ym_dash or "").strip()         # 2023-01
    if not ym_dash:
        return False
    ym_us   = ym_dash.replace("-", "_")       # 2023_01
    ym_comp = ym_dash.replace("-", "")        # 202301

    # Exact directory match
    if (base / ym_dash).exists():
        return True

    # Prefix dir/file match
    if any(base.glob(f"{ym_dash}*")):
        return True

    # Underscore variants (incl. day-level or suffixes)
    if any(base.glob(f"*{ym_us}*.parquet")):
        return True

    # Compact token somewhere in filename
    if any(base.glob(f"*{ym_comp}*.parquet")):
        return True

    return False

def have_min_months(**kwargs) -> bool:
    ds = kwargs["ds"]  # 'YYYY-MM-DD'
    params = (kwargs.get("dag_run").conf or {}) | kwargs["params"]
    min_months = int(params.get("min_months", MIN_MONTHS_REQUIRED))

    # anchor = previous month (start of month), then walk backwards
    anchor = datetime.strptime(ds, "%Y-%m-%d").replace(day=1) - relativedelta(months=1)

    count = 0
    cur = anchor
    while count < min_months:
        ym_dash = _month_str(cur)
        ok = (
            _part_exists(GOLD_CLICK_BASE, ym_dash)
            and _part_exists(GOLD_ATTR_BASE, ym_dash)
            and _part_exists(GOLD_FIN_BASE, ym_dash)
            and _part_exists(LABEL_BASE, ym_dash)
        )
        if not ok:
            break
        count += 1
        cur -= relativedelta(months=1)

    if count < min_months:
        print(f"Only {count} month(s) ready before {anchor.strftime('%Y-%m')} (need {min_months}). Short-circuiting training.")
        return False

    print(f"{count} month(s) ready. Proceeding to training.")
    return True

def _prod_model_exists_mlflow(model_name="credit_risk_model") -> bool:
    """
    Return True iff there is at least one PRODUCTION model version in MLflow Registry.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    try:
        latest = client.get_latest_versions(name=model_name, stages=["Production"])
        has_prod = bool(latest)
        print(f"[MLFLOW] Production exists for {model_name}? {has_prod}")
        return has_prod
    except Exception as e:
        # Be explicit so failures don’t silently disable bootstrap
        print(f"[MLFLOW] Could not query registry ({type(e).__name__}: {e}). Assuming NO Production.")
        return False


def _needs_bootstrap(**_) -> bool:
    """
    Run initial training exactly once: only when there is no PRODUCTION model
    registered in MLflow yet.
    """
    needs = not _prod_model_exists_mlflow("credit_risk_model")
    print(f"[BOOTSTRAP] needs_bootstrap={needs}")
    return needs

def decide_next_task(**ctx) -> str:
    """
    Branch to monthly training start or inference start.

    Priority:
    1) Manual override (run_training / force_training) -> TRAIN (if min-months satisfied)
    2) Auto gate: monthly_need_training AND have_min_months -> TRAIN
    3) Otherwise -> INFER (prod gate inside inference group will short-circuit if no prod)
    """
    # read overrides from dag_run.conf or params
    conf = (ctx.get("dag_run").conf or {}) | (ctx.get("params") or {})
    force_training = bool(conf.get("run_training") or conf.get("force_training"))

    # compute group prefix for fully-qualified task ids
    group_prefix = ctx["task"].task_group.group_id  # expected "monthly_tasks"
    train_target = f"{group_prefix}.model_training_monthly.start"
    infer_target = f"{group_prefix}.model_inference_monthly.start"

    has_min_months = have_min_months(**ctx)

    if force_training:
        print(f"[BRANCH] Manual override -> run_training=True | have_min_months={has_min_months}")
        # If not enough months, we still fall back to inference so the DAG can proceed.
        return train_target if has_min_months else infer_target

    need_training = monthly_need_training(**ctx)
    print(f"[BRANCH] need_training={need_training}, have_min_months={has_min_months}")

    if need_training and has_min_months:
        return train_target

    # Fall back to inference; internal prod_gate will short-circuit if no Production model
    return infer_target

# -----------------------------
# DAG
# -----------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="credit_risk_monthly_pipeline",
    default_args=default_args,
    description="Data pipeline: build gold, (optionally) train, infer, and monitor monthly",
    schedule_interval="0 0 1 * *",  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    catchup=True,
    tags=["etl", "gold", "pretrain", "training"],
    params={
        "run_training": False,           # manual/ad-hoc gate (kept for compatibility)
        "min_months": MIN_MONTHS_REQUIRED,
    },
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # =========================
    # LABEL STORE PIPELINE
    # =========================
    with TaskGroup(group_id="label_store") as label_store:
        dep_check_source_label_data = EmptyOperator(task_id="dep_check_source_label_data")

        bronze_label_store = BashOperator(
            task_id="run_bronze_label_store",
            bash_command='cd /opt/airflow/scripts && python3 bronze_label_store.py --snapshotdate "{{ ds }}"',
        )
        silver_label_store = BashOperator(
            task_id="run_silver_label_store",
            bash_command='cd /opt/airflow/scripts && python3 silver_label_store.py --snapshotdate "{{ ds }}"',
        )
        gold_label_store = BashOperator(
            task_id="run_gold_label_store",
            bash_command='cd /opt/airflow/scripts && python3 gold_label_store.py --snapshotdate "{{ ds }}" --dpd 30 --mob 6',
        )
        label_store_completed = EmptyOperator(
            task_id="label_store_completed",
            trigger_rule=TriggerRule.ALL_DONE
        )

        dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # =========================
    # FEATURE STORE PIPELINE
    # =========================
    with TaskGroup(group_id="feature_store") as feature_store:
        dep_check_source_data_bronze_1 = EmptyOperator(task_id="dep_check_source_data_bronze_1")
        dep_check_source_data_bronze_2 = EmptyOperator(task_id="dep_check_source_data_bronze_2")
        dep_check_source_data_bronze_3 = EmptyOperator(task_id="dep_check_source_data_bronze_3")

        bronze_clickstream = BashOperator(
            task_id="bronze_clickstream",
            bash_command='cd /opt/airflow/scripts && python3 bronze_clickstream.py --snapshotdate "{{ ds }}"',
        )
        bronze_financials = BashOperator(
            task_id="bronze_financials",
            bash_command='cd /opt/airflow/scripts && python3 bronze_financials.py --snapshotdate "{{ ds }}"',
        )
        bronze_attributes = BashOperator(
            task_id="bronze_attributes",
            bash_command='cd /opt/airflow/scripts && python3 bronze_attributes.py --snapshotdate "{{ ds }}"',
        )

        silver_clickstream = BashOperator(
            task_id="silver_clickstream",
            bash_command='cd /opt/airflow/scripts && python3 silver_clickstream.py --snapshotdate "{{ ds }}"',
        )
        silver_financials = BashOperator(
            task_id="silver_financials",
            bash_command='cd /opt/airflow/scripts && python3 silver_financials.py --snapshotdate "{{ ds }}"',
        )
        silver_attributes = BashOperator(
            task_id="silver_attributes",
            bash_command='cd /opt/airflow/scripts && python3 silver_attributes.py --snapshotdate "{{ ds }}"',
        )

        gold_clickstream = BashOperator(
            task_id="gold_clickstream",
            bash_command='cd /opt/airflow/scripts && python3 gold_clickstream.py --snapshotdate "{{ ds }}"',
        )
        gold_attributes = BashOperator(
            task_id="gold_attributes",
            bash_command='cd /opt/airflow/scripts && python3 gold_attributes.py --snapshotdate "{{ ds }}"',
        )
        gold_financials = BashOperator(
            task_id="gold_financials",
            bash_command='cd /opt/airflow/scripts && python3 gold_financials.py --snapshotdate "{{ ds }}"',
        )

        feature_store_completed = EmptyOperator(task_id="feature_store_completed", trigger_rule=TriggerRule.ALL_DONE)

        dep_check_source_data_bronze_1 >> bronze_clickstream
        dep_check_source_data_bronze_2 >> bronze_financials
        dep_check_source_data_bronze_3 >> bronze_attributes

        bronze_clickstream >> silver_clickstream
        bronze_financials >> silver_financials
        bronze_attributes >> silver_attributes

        silver_clickstream >> gold_clickstream
        silver_attributes >> gold_attributes
        silver_financials >> gold_financials

        [gold_clickstream, gold_attributes, gold_financials] >> feature_store_completed

    # =========================
    # MANUAL / AD-HOC TRAINING (kept)
    # =========================
    check_min_14m_data = ShortCircuitOperator(task_id="check_min_14m_data", python_callable=have_min_months)

    needs_bootstrap = ShortCircuitOperator(
        task_id="needs_bootstrap",
        python_callable=_needs_bootstrap,
    )

    prod_gate = ShortCircuitOperator(task_id="check_production_model", python_callable=_prod_model_exists_mlflow)

    with TaskGroup(group_id="initial_training") as initial_training:
        initial_training_start = EmptyOperator(task_id="start")
        build_pretrain_from_gold_manual = BashOperator(
            task_id="build_pretrain_from_gold_manual",
            bash_command='cd /opt/airflow/scripts && python3 pretrain_gold_features.py --snapshotdate "{{ ds }}" --mob 6',
        )

        train_xgb_manual = BashOperator(
            task_id="train_xgboost_manual",
            bash_command=(
                "set -euo pipefail && "
                "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                "cd /opt/airflow/scripts && "
                'python3 train_xgboost_ml.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 --oot-months 2 --train-ratio 0.8 --n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/"
            ),
        )

        train_logreg_manual = BashOperator(
            task_id="train_logreg_manual",
            bash_command=(
                "set -euo pipefail && "
                "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                "cd /opt/airflow/scripts && "
                'python3 train_logreg_ml.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 --oot-months 2 --train-ratio 0.8 --n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/"
            ),
        )

        promote_best_manual = BashOperator(
            task_id="promote_best_manual",
            bash_command=(
                "set -euo pipefail && "
                "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                "cd /opt/airflow/scripts && "
                "python3 promote_best_ml.py "
                '--train-date "{{ ds }}" '
                "--experiment credit_risk_training "
                "--model-name credit_risk_model "
                "--metric auc_oot "
                "--min-test-auc 0.70 "
                "--stage Production "
                "--archive-existing 1 "
                "--dry-run 0"
            ),
            trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.NONE_FAILED),
        )

        training_done_manual = EmptyOperator(task_id="training_done_manual", trigger_rule=TriggerRule.ALL_SUCCESS)

        initial_training_start >> build_pretrain_from_gold_manual >> [train_xgb_manual, train_logreg_manual] >> promote_best_manual >> training_done_manual

    # =========================
    # MONTHLY TRAIN / INFER / MONITOR
    # =========================
    with TaskGroup(group_id="monthly_tasks") as monthly_tasks:
        monthly_tasks_entry = EmptyOperator(task_id="start")

        branch_train_or_infer = BranchPythonOperator(task_id="train_or_infer", python_callable=decide_next_task)

        with TaskGroup(group_id="model_training_monthly") as model_training_monthly:
            training_start = EmptyOperator(task_id="start")
            build_pretrain_from_gold_monthly = BashOperator(
                task_id="build_pretrain_from_gold_monthly",
                bash_command='cd /opt/airflow/scripts && python3 pretrain_gold_features.py --snapshotdate "{{ ds }}" --mob 6',
            )

            train_xgb_monthly = BashOperator(
                task_id="train_xgboost_monthly",
                bash_command=(
                    "set -euo pipefail && "
                    "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                    "cd /opt/airflow/scripts && "
                    'python3 train_xgboost_ml.py '
                    '--model-train-date "{{ ds }}" '
                    "--train-test-months 12 --oot-months 2 --train-ratio 0.8 --n-iter 50 "
                    "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                    "--gold-labels-dir datamart/gold/labels/"
                ),
            )

            train_logreg_monthly = BashOperator(
                task_id="train_logreg_monthly",
                bash_command=(
                    "set -euo pipefail && "
                    "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                    "cd /opt/airflow/scripts && "
                    'python3 train_logreg_ml.py '
                    '--model-train-date "{{ ds }}" '
                    "--train-test-months 12 --oot-months 2 --train-ratio 0.8 --n-iter 50 "
                    "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                    "--gold-labels-dir datamart/gold/labels/"
                ),
            )

            promote_best_monthly = BashOperator(
                task_id="promote_best_monthly",
                bash_command=(
                    "set -euo pipefail && "
                    "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                    "cd /opt/airflow/scripts && "
                    "python3 promote_best_ml.py "
                    '--train-date "{{ ds }}" '
                    "--experiment credit_risk_training "
                    "--model-name credit_risk_model "
                    "--metric auc_oot "
                    "--min-test-auc 0.70 "
                    "--stage Production "
                    "--archive-existing 1 "
                    "--dry-run 0"
                ),
                trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.NONE_FAILED),
)

            training_done_monthly = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

            training_start >> build_pretrain_from_gold_monthly >> [train_xgb_monthly, train_logreg_monthly] >> promote_best_monthly >> training_done_monthly

        with TaskGroup(group_id="model_inference_monthly") as model_inference_monthly:
            inference_start = EmptyOperator(task_id="start", trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.ONE_SUCCESS))
            pretrain_gold_for_infer = BashOperator(
                task_id="pretrain_gold_for_infer",
                bash_command='cd /opt/airflow/scripts && python3 pretrain_gold_features.py --snapshotdate "{{ ds }}" --mob 6',
                trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.ONE_SUCCESS),
            )
            
            run_inference = BashOperator(
                task_id="run_inference_production",
                bash_command=(
                    "set -euo pipefail && "
                    "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                    "cd /opt/airflow/scripts && "
                    "python3 model_inference_ml.py "
                    '--snapshotdate "{{ ds }}" '
                    "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                    "--predictions-out-dir datamart/gold/model_predictions/ "
                    "--model-name credit_risk_model "
                    "--model-stage Production "
                    "--experiment credit_risk_inference"
                    # " --write-csv"  # <- uncomment if you also want a CSV alongside the parquet
                ),
            )
            model_inference_completed = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

            inference_start >> pretrain_gold_for_infer >> run_inference >> model_inference_completed

        with TaskGroup(group_id="model_monitor_monthly") as model_monitor_monthly:
            model_monitor_start = EmptyOperator(task_id="start")
            monitor_perf = BashOperator(
                task_id="monitor_model_performance",
                bash_command=(
                    "set -euo pipefail && "
                    "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
                    "cd /opt/airflow/scripts && "
                    "python3 monitor_model_performance_ml.py "
                    "--use-mlflow "
                    "--mlflow-experiment credit_risk_inference "
                    "--mlflow-artifact-name predictions "   # <— folder, not a file
                    "--gold-labels-dir datamart/gold/labels/ "
                    "--end-date '{{ ds }}' "
                    f"--out-dir {MONITOR_OUT_DIR}/ "
                    "--threshold 0.5"
                ),
            )
            model_monitor_completed = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)
            model_monitor_start >> monitor_perf >> model_monitor_completed

        monthly_tasks_entry >> branch_train_or_infer
        branch_train_or_infer >> model_training_monthly >> model_inference_monthly >> model_monitor_monthly
        branch_train_or_infer >> model_inference_monthly >> model_monitor_monthly

    # =========================
    # Top-level dependencies
    # =========================
    start >> [label_store, feature_store] >> check_min_14m_data
    check_min_14m_data >> needs_bootstrap >> initial_training >> end
    check_min_14m_data >> prod_gate >> monthly_tasks >> end
