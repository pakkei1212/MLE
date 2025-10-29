from airflow import DAG
from pathlib import Path
from airflow.operators.empty import EmptyOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os, re

# Configurable knobs
MIN_MONTHS_REQUIRED = 14
GOLD_CLICK_BASE = Path("/opt/airflow/scripts/datamart/gold/clickstream")
GOLD_ATTR_BASE  = Path("/opt/airflow/scripts/datamart/gold/attributes")
GOLD_FIN_BASE   = Path("/opt/airflow/scripts/datamart/gold/financials")
LABEL_BASE      = Path("/opt/airflow/scripts/datamart/gold/labels")
PROD_DIR = "/opt/airflow/scripts/model_bank/production/best"

def _month_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def _part_exists(base: Path, ym: str) -> bool:
    # adjust if your partitions are like .../YYYY/MM/... instead of YYYY-MM
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

def _parse_train_date_from_version(version: str):
    """
    Accepts names like:
      credit_model_xgb_2024_01_01
      credit_model_logreg_2024_01_01
      credit_model_lgbm_2024_01_01
    Returns datetime.date or None.
    """
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})$', version or "")
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    try:
        return datetime(y, mth, d).date()
    except Exception:
        return None

def _last_training_date_from_prod(**kwargs):
    """
    Read production/best/model_version.txt and parse the date token.
    Fallback: if not present, return None.
    """
    ver_path = os.path.join(PROD_DIR, "model_version.txt")
    if not os.path.exists(ver_path):
        print("[NEED_TRAIN] No production version file found.")
        return None
    with open(ver_path, "r") as f:
        version = f.read().strip()
    dt = _parse_train_date_from_version(version)
    print(f"[NEED_TRAIN] Production model_version={version} parsed_date={dt}")
    return dt

def need_training(**ctx) -> bool:
    """
    Training is needed if:
      - user requested it via run_training=True (dag_run.conf or dag.params), OR
      - current ds - last_training_date >= 3 months (stale production)
    """
    # flag from params/run conf
    run_flag = bool((ctx["dag_run"].conf or {}).get("run_training", ctx["params"].get("run_training", False)))

    # staleness check
    ds_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()
    last_dt = _last_training_date_from_prod(**ctx)
    if last_dt is None:
        stale_flag = False  # no prod → let inference gate handle skipping; keep training under run_flag/min-months
    else:
        delta = relativedelta(ds_date, last_dt)
        months_apart = delta.years * 12 + delta.months
        stale_flag = months_apart >= 3
        print(f"[NEED_TRAIN] ds={ds_date} last_train_date={last_dt} months_apart={months_apart} stale={stale_flag}")

    need = run_flag or stale_flag
    print(f"[NEED_TRAIN] run_flag={run_flag} → training_needed={need}")
    return need

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
    params={  # scheduler defaults: training OFF; allow override per manual run
        "run_training": False,
        "min_months": MIN_MONTHS_REQUIRED,
    },
) as dag:

    # Put sensible defaults so the scheduler never trains by itself
    # (these can also go in the DAG(...) call as `params=...`)
    dag.params.update({"run_training": False})
        # data pipeline

    start = EmptyOperator(task_id="start")  # <-- new start node

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

        # Replace placeholder with actual gold script
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

        # Define task dependencies to run scripts sequentially
        dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
    
    # =========================
    # FEATURE STORE PIPELINE
    # =========================
    with TaskGroup(group_id="feature_store") as feature_store:
        # dep checks
        dep_check_source_data_bronze_1 = EmptyOperator(task_id="dep_check_source_data_bronze_1")
        dep_check_source_data_bronze_2 = EmptyOperator(task_id="dep_check_source_data_bronze_2")
        dep_check_source_data_bronze_3 = EmptyOperator(task_id="dep_check_source_data_bronze_3")

        # --- Bronze ---
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

        # --- Silver ---
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

        # --- Gold (split into 3 independent outputs; no mob/cutoff/merge) ---
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

        # dep checks -> all bronze
        dep_check_source_data_bronze_1 >> bronze_clickstream 
        dep_check_source_data_bronze_2 >> bronze_financials
        dep_check_source_data_bronze_3 >> bronze_attributes

        # all bronze -> all silver
        bronze_clickstream >> silver_clickstream
        bronze_financials >> silver_financials
        bronze_attributes >> silver_attributes

        # all silver -> gold
        silver_clickstream >> gold_clickstream
        silver_attributes >> gold_attributes
        silver_financials >> gold_financials

        [gold_clickstream, gold_attributes, gold_financials] >> feature_store_completed

    # =========================
    # MODEL TRAINING (manual, gated)
    # =========================
    run_training_gate = ShortCircuitOperator(
        task_id="run_training_gate",
        python_callable=lambda **ctx: bool(
            (ctx["dag_run"].conf or {}).get("run_training", ctx["params"].get("run_training", False))
        ),
    )

    need_training_gate = ShortCircuitOperator(
        task_id="need_training_gate",
        python_callable=need_training,
    )

    check_min_12m = ShortCircuitOperator(
        task_id="check_min_12m",
        python_callable=have_min_months,
    )   

    with TaskGroup(group_id="model_training") as model_training:

        build_pretrain_from_gold = BashOperator(
            task_id="build_pretrain_from_gold",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 pretrain_gold_features.py '
                '--snapshotdate "{{ ds }}" --mob 6'
            ),
        )                
        
        # Train XGBoost
        train_xgb = BashOperator(
            task_id="train_xgboost",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_xgboost.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                # Use your existing locations; override defaults explicitly
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )

        # Train Logistc Regression
        train_logreg = BashOperator(
            task_id="train_logreg",
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

        # Train LightGBM
        '''train_lgbm = BashOperator(
            task_id="train_lightgbm",
            bash_command=(
                "cd /opt/airflow/scripts && "
                'python3 train_lightgbm.py '
                '--model-train-date "{{ ds }}" '
                "--train-test-months 12 "
                "--oot-months 2 "
                "--train-ratio 0.8 "
                "--n-iter 50 "
                "--gold-pretrain-features-dir datamart/pretrain_gold/features/ "
                "--gold-labels-dir datamart/gold/labels/ "
                "--model-bank-dir model_bank/"
            ),
        )'''

        # Promote the best (by OOT AUC) among all candidates saved in model_bank
        promote_best = BashOperator(
            task_id="promote_best",
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

        model_training_completed = EmptyOperator(
            task_id="model_training_completed",
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )

        '''publish_metrics = PythonOperator(
            task_id="publish_metrics",
            python_callable=publish_auc,
            provide_context=True,
        )'''

        build_pretrain_from_gold >> [train_xgb, train_logreg] >> promote_best >> model_training_completed

    def _prod_model_exists(**_):
        import os
        return os.path.exists(os.path.join(PROD_DIR, "model.pkl"))

    # =========================
    # MODEL INFERENCE (always when prod exists)
    # =========================
    with TaskGroup(group_id="model_inference") as model_inference:

        # build pretrain features for the scoring date as well (independent of training)
        pretrain_for_inferencing = BashOperator(
            task_id="pretrain_for_inferencing",
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
                "python3 scripts/inference/model_inference.py "
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

        prod_gate >> run_inference >> model_inference_completed

    # =========================
    # MODEL MONITORING
    # =========================
    with TaskGroup(group_id="model_monitor") as model_monitor:
        model_monitor_start = EmptyOperator(task_id="model_monitor_start")
        model_1_monitor = EmptyOperator(task_id="model_1_monitor")
        model_2_monitor = EmptyOperator(task_id="model_2_monitor")
        model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")
    
        # Define task dependencies to run scripts sequentially
        model_monitor_start >> model_1_monitor >> model_monitor_completed
        model_monitor_start >> model_2_monitor >> model_monitor_completed


    # =========================
    # MODEL AUTO-TRAIN
    # =========================
    with TaskGroup(group_id="model_automl") as model_automl:
        model_automl_start = EmptyOperator(task_id="model_automl_start")
        model_1_automl = EmptyOperator(task_id="model_1_automl")
        model_2_automl = EmptyOperator(task_id="model_2_automl")
        model_automl_completed = EmptyOperator(task_id="model_automl_completed")
    
        # Define task dependencies to run scripts sequentially
        model_automl_start >> model_1_automl >> model_automl_completed
        model_automl_start >> model_2_automl >> model_automl_completed
    
    # ---------------------------------
        # Top-level dependencies
    # ---------------------------------
    start >> [label_store, feature_store]

    # Model training needs both labels & gold features done for the date
    [label_store, feature_store] >> run_training_gate  >> check_min_12m >> model_training

    # Inference should happen every schedule if production model exists
    # (requires feature/label store to be ready for pretrain features)
    [label_store, feature_store] >> model_inference

    # Monitoring after inference
    model_inference >> model_monitor

    # AutoML needs both labels & feature store
    [feature_store, label_store] >> model_automl