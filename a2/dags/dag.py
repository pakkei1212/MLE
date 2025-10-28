from airflow import DAG
from pathlib import Path
from airflow.operators.empty import EmptyOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os, re

# -----------------------------
# Configurable knobs
# -----------------------------
MIN_MONTHS_REQUIRED = 14
GOLD_CLICK_BASE = Path("/opt/airflow/scripts/datamart/gold/clickstream")
GOLD_ATTR_BASE  = Path("/opt/airflow/scripts/datamart/gold/attributes")
GOLD_FIN_BASE   = Path("/opt/airflow/scripts/datamart/gold/financials")
LABEL_BASE      = Path("/opt/airflow/scripts/datamart/gold/labels")
PROD_DIR = "/opt/airflow/scripts/model_bank/production/best"

# -----------------------------
# Helpers
# -----------------------------
def monthly_skip_training(**ctx) -> bool:
    # True means: we are NOT running monthly training this schedule
    return not monthly_need_training(**ctx)

def _parse_train_date_from_version(version: str):
    """
    Accepts e.g. credit_model_xgb_2024_01_01 / credit_model_logreg_2024_01_01.
    Returns datetime.date or None.
    """
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})$', version or "")
    if not m:
        return None
    y, mm, dd = map(int, m.groups())
    try:
        return datetime(y, mm, dd).date()
    except Exception:
        return None

def _last_training_date_from_prod(**kwargs):
    """
    Reads production/best/model_version.txt and parses the date suffix.
    Returns a date or None if not found.
    """
    ver_path = os.path.join(PROD_DIR, "model_version.txt")
    if not os.path.exists(ver_path):
        print("[MONTHLY_GATE] No production model_version.txt")
        return None
    with open(ver_path, "r") as f:
        version = f.read().strip()
    dt = _parse_train_date_from_version(version)
    print(f"[MONTHLY_GATE] production version={version} parsed_date={dt}")
    return dt

def monthly_need_training(**ctx) -> bool:
    """
    Run monthly training only if:
      - NO production model yet  -> True (bootstrap)
      - OR (current ds date - last training date) >= 3 calendar months -> True
      - ELSE -> False (skip if < 3 months)
    """
    ds_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()
    last_dt = _last_training_date_from_prod(**ctx)
    if last_dt is None:
        print("[MONTHLY_GATE] No last training date -> run monthly training")
        return True
    delta = relativedelta(ds_date, last_dt)
    months_apart = delta.years * 12 + delta.months
    should_run = months_apart >= 3
    print(f"[MONTHLY_GATE] ds={ds_date}, last={last_dt}, months_apart={months_apart} -> run={should_run}")
    return should_run

def _month_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def _part_exists(base: Path, ym: str) -> bool:
    return (base / ym).exists() or any(p.name.startswith(ym) for p in base.glob(f"{ym}*"))

def have_min_months(**kwargs) -> bool:
    ds = kwargs["ds"]  # 'YYYY-MM-DD'
    params = (kwargs.get("dag_run").conf or {}) | kwargs["params"]
    min_months = int(params.get("min_months", MIN_MONTHS_REQUIRED))

    anchor = datetime.strptime(ds, "%Y-%m-%d").replace(day=1) - relativedelta(months=1)

    count = 0
    cur = anchor
    while count < min_months:
        ym = _month_str(cur)
        ok = (
            _part_exists(GOLD_CLICK_BASE, ym)
            and _part_exists(GOLD_ATTR_BASE, ym)
            and _part_exists(GOLD_FIN_BASE, ym)
            and _part_exists(LABEL_BASE, ym)
        )
        if not ok:
            break
        count += 1
        cur -= relativedelta(months=1)

    if count < min_months:
        print(
            f"Only {count} month(s) ready before {anchor.strftime('%Y-%m')} "
            f"(need {min_months}). Short-circuiting training."
        )
        return False

    print(f"{count} month(s) ready. Proceeding to training.")
    return True

def _prod_model_exists(**_):
    return os.path.exists(os.path.join(PROD_DIR, "model.pkl"))

# -----------------------------
# DAG
# -----------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    tags=["etl", "gold", "pretrain", "training"],
    params={
        "run_training": False,                 # manual/ad-hoc gate only
        "min_months": MIN_MONTHS_REQUIRED,
    },
) as dag:

    dag.params.update({"run_training": False})

    start = EmptyOperator(task_id="start")

    # =========================
    # LABEL STORE PIPELINE
    # =========================
    with TaskGroup(group_id="label_store") as label_store:

        dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

        bronze_label_store = BashOperator(
            task_id='run_bronze_label_store',
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 bronze_label_store.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        silver_label_store = BashOperator(
            task_id="run_silver_label_store",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 silver_label_store.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        gold_label_store = BashOperator(
            task_id="run_gold_label_store",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 gold_label_store.py '
                '--snapshotdate "{{ ds }}" '
                '--dpd 30 '
                '--mob 6'
            ),
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
            task_id='bronze_clickstream',
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 bronze_clickstream.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        bronze_financials = BashOperator(
            task_id='bronze_financials',
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 bronze_financials.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        bronze_attributes = BashOperator(
            task_id='bronze_attributes',
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 bronze_attributes.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        silver_clickstream = BashOperator(
            task_id="silver_clickstream",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 silver_clickstream.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        silver_financials = BashOperator(
            task_id="silver_financials",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 silver_financials.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        silver_attributes = BashOperator(
            task_id="silver_attributes",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 silver_attributes.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        gold_clickstream = BashOperator(
            task_id="gold_clickstream",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 gold_clickstream.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        gold_attributes = BashOperator(
            task_id="gold_attributes",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 gold_attributes.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        gold_financials = BashOperator(
            task_id="gold_financials",
            bash_command=(
                'cd /opt/airflow/scripts && '
                'python3 gold_financials.py '
                '--snapshotdate "{{ ds }}"'
            ),
        )

        feature_store_completed = EmptyOperator(
            task_id="feature_store_completed",
            trigger_rule=TriggerRule.ALL_DONE
        )

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
    
    # =========================================================
    # SCHEDULED MONTHLY TRAINING BRANCH (new gate)
    # Skip if last training < 3 months ago
    # =========================================================
    monthly_need_training_gate = ShortCircuitOperator(
        task_id="monthly_need_training_gate",
        python_callable=monthly_need_training,
    )
    monthly_check_min_12m = ShortCircuitOperator(
        task_id="monthly_check_min_12m",
        python_callable=have_min_months,
    )

    # =========================================================
    # MANUAL / AD-HOC TRAINING BRANCH (unchanged behavior)
    # =========================================================
    manual_training_gate = ShortCircuitOperator(
        task_id="manual_training_gate",
        python_callable=lambda **ctx: bool(
            (ctx["dag_run"].conf or {}).get("run_training", ctx["params"].get("run_training", False))
        ),
    )
    
    manual_check_min_12m = ShortCircuitOperator(
        task_id="manual_check_min_12m",
        python_callable=have_min_months,
    )

    with TaskGroup(group_id="model_training_manual") as model_training_manual:

        build_pretrain_from_gold_manual = BashOperator(
            task_id="build_pretrain_from_gold_manual",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 pretrain_gold_features.py '
                '--snapshotdate "{{ ds }}" --mob 6'
            ),
        )

        train_xgb_manual = BashOperator(
            task_id="train_xgboost_manual",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_xgboost.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )

        train_logreg_manual = BashOperator(
            task_id="train_logreg_manual",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_logreg.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )

        promote_best_manual = BashOperator(
            task_id="promote_best_manual",
            bash_command=(
                "cd /opt/airflow/scripts && "
                "python3 promote_best.py "
                "--registry-dir datamart/gold/model_registry "
                "--model-bank-dir model_bank "
                "--production-dir model_bank/production/best "
                '--train-date "{{ ds }}"'
            ),
            trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.NONE_FAILED),
        )

        training_done_manual = EmptyOperator(
            task_id="training_done_manual",
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )

        build_pretrain_from_gold_manual >> [train_xgb_manual, train_logreg_manual] >> promote_best_manual >> training_done_manual

    # =========================================================
    # SCHEDULED MONTHLY TRAINING BRANCH (new)
    # Always attempts training each schedule if 12m data exists.
    # =========================================================
    with TaskGroup(group_id="model_training_monthly") as model_training_monthly:

        build_pretrain_from_gold_monthly = BashOperator(
            task_id="build_pretrain_from_gold_monthly",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 pretrain_gold_features.py '
                '--snapshotdate "{{ ds }}" --mob 6'
            ),
        )

        train_xgb_monthly = BashOperator(
            task_id="train_xgboost_monthly",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_xgboost.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )

        train_logreg_monthly = BashOperator(
            task_id="train_logreg_monthly",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_logreg.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )

        promote_best_monthly = BashOperator(
            task_id="promote_best_monthly",
            bash_command=(
                "cd /opt/airflow/scripts && "
                "python3 promote_best.py "
                "--registry-dir datamart/gold/model_registry "
                "--model-bank-dir model_bank "
                "--production-dir model_bank/production/best "
                '--train-date "{{ ds }}"'
            ),
            trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.NONE_FAILED),
        )

        training_done_monthly = EmptyOperator(
            task_id="training_done_monthly",
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )

        build_pretrain_from_gold_monthly >> [train_xgb_monthly, train_logreg_monthly] >> promote_best_monthly >> training_done_monthly
    
    # Gate that succeeds only when we do NOT need monthly training
    monthly_skip_training_gate = ShortCircuitOperator(
        task_id="monthly_skip_training_gate",
        python_callable=monthly_skip_training,
    )

    # Data readiness for inference (always required)
    data_ready_for_inference = EmptyOperator(task_id="data_ready_for_inference")

    # Join node that releases inference when EITHER monthly training finished
    # OR we explicitly skipped monthly training. It tolerates the other path being skipped.
    inference_entry = EmptyOperator(
        task_id="inference_entry",
        trigger_rule=getattr(TriggerRule, "NONE_FAILED_MIN_ONE_SUCCESS", TriggerRule.ONE_SUCCESS),
    )

    # =========================
    # MODEL INFERENCE (always when prod exists)
    # =========================
    with TaskGroup(group_id="model_inference") as model_inference:

        pretrain_for_scoring = BashOperator(
            task_id="pretrain_for_scoring",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 pretrain_gold_features.py '
                '--snapshotdate "{{ ds }}" --mob 6'
            ),
        )

        prod_gate = ShortCircuitOperator(
            task_id="check_production_model",
            python_callable=_prod_model_exists,
        )

        run_inference = BashOperator(
            task_id="run_inference_production",
            bash_command=(
                "cd /opt/airflow/scripts && "
                "python3 model_inference.py "
                '--snapshotdate "{{ ds }}" '
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--model-bank-dir model_bank/ "
                "--predictions-out-dir datamart/gold/model_predictions/ "
                "--use-production"
            ),
        )

        model_inference_completed = EmptyOperator(
            task_id="model_inference_completed",
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )

        pretrain_for_scoring >> prod_gate >> run_inference >> model_inference_completed

    # =========================
    # MODEL MONITORING
    # =========================
    with TaskGroup(group_id="model_monitor") as model_monitor:
        model_monitor_start = EmptyOperator(task_id="model_monitor_start")
        model_1_monitor = EmptyOperator(task_id="model_1_monitor")
        model_2_monitor = EmptyOperator(task_id="model_2_monitor")
        model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")

        model_monitor_start >> model_1_monitor >> model_monitor_completed
        model_monitor_start >> model_2_monitor >> model_monitor_completed

    # =========================
    # Top-level dependencies
    # =========================
    start >> [label_store, feature_store]

    # Manual/ad-hoc training path (requires flag + min months)
    [label_store, feature_store] >> manual_training_gate >> manual_check_min_12m >> model_training_manual

    # Scheduled monthly training path (no flag; only min months guard)
    [label_store, feature_store] >> monthly_need_training_gate >> monthly_check_min_12m >> model_training_monthly

    training_done_monthly >> inference_entry
    monthly_skip_training_gate >> inference_entry

    # Inference runs each schedule if production exists
    [label_store, feature_store] >> data_ready_for_inference
    [data_ready_for_inference, inference_entry] >> model_inference

    # Monitoring after inference
    model_inference >> model_monitor
