# 🧠 Credit Risk Prediction MLOps Pipeline

> **Automated End-to-End Model Lifecycle for Default Risk Prediction**  
> Built with **Apache Airflow**, **MLflow**, **Grafana**, and **PostgreSQL**, orchestrated through **Docker Compose**.

---

## 📊 System Overview

This project implements a **fully automated MLOps pipeline** for credit risk prediction using structured data such as **financial attributes, clickstreams, and loan repayment behavior**.  
The workflow ensures data quality, model reproducibility, and continuous performance monitoring.

All components — **data ingestion, training, inference, retraining, and monitoring** — are orchestrated by **Apache Airflow** and tracked via **MLflow** and **Grafana** dashboards.

---

## 🏗️ Architecture

### 🔹 Data Pipeline
The data processing follows the **Medallion Architecture** pattern:
```
Bronze → Silver → Gold
```
- **Bronze:** Raw ingestion from transactional and external sources.  
- **Silver:** Cleaned, joined, and standardized records.  
- **Gold:** Feature-engineered datasets for training and inference.

---

### 🔹 Workflow Orchestration (Airflow)
The workflow is managed by **Apache Airflow DAGs**, coordinating:

1. **Data Availability Check**  
   Ensures at least 14 months of data are available in the Gold Datamart.
2. **Initial Model Training**  
   Runs when enough data exists for a 12-month train–test and 2-month out-of-time (OOT) split.
3. **Model Selection**  
   Compares candidates (Logistic Regression & XGBoost) using metrics tracked in MLflow.
4. **Batch Inference**  
   Predicts default risk monthly using the latest production model.
5. **Model Monitoring**  
   Logs metrics such as AUC, PSI, KS, and drift statistics to MLflow → visualized in Grafana.
6. **Model Retraining Cycle**  
   Triggered if at least 3 months have passed since last retraining.  
   Automatically promotes the best model to **Production** if performance improves.

---

## 🧩 Component Diagram

| Component | Purpose | Tech Stack |
|------------|----------|-------------|
| **Data Layer** | Medallion architecture for data preprocessing | PySpark, Pandas |
| **Workflow Orchestration** | Automates data, model, inference tasks | Apache Airflow |
| **Experiment Tracking** | Stores runs, metrics, parameters, models | MLflow + PostgreSQL |
| **Monitoring Dashboard** | Visualizes PSI, KS, and performance drift | Grafana + Prometheus |
| **Model Registry** | Version control and deployment staging | MLflow Model Registry |
| **Artifact Store** | Persisted model binaries and metrics | Docker volume `/tmp/mlflow/artifacts` |

---

## ⚙️ Project Structure

```
a2/
├── dags/
│   └── dag.py
│
├── scripts/
│   ├── model_training.py
│   ├── train_xgboost.py
│   ├── train_logreg.py
│   ├── model_inference.py
│   ├── monitor_model_performance.py
│   ├── promote_best.py
│   ├── ml_transforms.py
│   └── utils/
│        ├── model_training_utils.py
│        ├── data_processing_*.py
│        └── __init__.py
│
├── datamart/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── docker-compose.yaml
├── Dockerfile
├── prometheus.yml
└── requirements.txt
```

---

## 🐳 Deployment (Docker Compose)

### 1️⃣ Start All Services
```bash
docker compose up -d
```

This launches:
| Service | Port | Purpose |
|----------|------|----------|
| **Airflow Webserver** | [http://localhost:8080](http://localhost:8080) | DAG orchestration |
| **MLflow Tracking UI** | [http://localhost:5000](http://localhost:5000) | Model registry & tracking |
| **Grafana Dashboard** | [http://localhost:3000](http://localhost:3000) | Monitoring metrics |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Time-series metrics |
| **PgAdmin** | [http://localhost:5050](http://localhost:5050) | Database admin |
| **PostgreSQL (Airflow)** | `localhost:5432` | Airflow metadata |
| **PostgreSQL (MLflow)** | `localhost:5433` | MLflow tracking DB |
| **Jupyter Notebook** | [http://localhost:8892](http://localhost:8892) | Development workspace |

---

## 🧮 Model Training Details

### Label Definition
- **Target:** Default / Non-default (binary classification)
- **Observation Window:** MOB = 6 (Month of Booking)
- **Label Source:** Extracted from Gold Datamart

### Feature Engineering
- Financial ratios, clickstream patterns, user attributes
- Normalization → `N(0,1)`
- One-hot encoding for categorical features
- Median / mode imputation for missing values

### Models
| Model | Purpose | Notes |
|--------|----------|--------|
| **Logistic Regression** | Baseline, interpretable | Benchmark model |
| **XGBoost** | Nonlinear boosting | Used for production inference |

### Hyperparameter Optimization
- Conducted via **Optuna**
- Objective metric:  
  - **AUC** – universal ranking metric  
  - **Gini (2×AUC−1)** – business-friendly measure

---

## 📈 Monitoring Metrics

| Metric | Description | Logged In |
|---------|--------------|-----------|
| **AUC / Gini** | Model performance | MLflow |
| **PSI (Population Stability Index)** | Drift in feature distribution | Grafana |
| **KS (Kolmogorov–Smirnov)** | Separation between good/bad | Grafana |
| **Accuracy, Recall, F1** | Classification quality | MLflow |
| **Sample Size** | Batch monitoring volume | Grafana |

---

## 🧠 Workflow Logic

> From the **flowchart**:

- If **data < 14 months** → skip training  
- If **data ≥ 14 months** and **no prior model** → initial training  
- If **last retrain ≥ 3 months** → retrain model  
- If **last retrain < 3 months** → skip retraining  
- Monthly inference → performance logged → Grafana dashboard updates automatically  

---

## 📊 Dashboards

### Grafana Panels:
1. **Inference vs Training AUC Over Time**  
   - Compare AUC_ROC (inference) vs AUC_OOT (training)  
2. **Drift Monitoring (PSI & KS)**  
   - Detects data drift thresholds:  
     - PSI < 0.1 → Stable  
     - 0.1–0.25 → Moderate shift  
     - > 0.25 → Significant drift (investigate features)

---

## 🧰 Development Tools

- **Python 3.9**  
- **Pandas, PySpark, XGBoost, Optuna, Scikit-learn**  
- **MLflow, Airflow, Grafana, Prometheus, PostgreSQL**  
- **Docker Compose for orchestration**

---

## 🏁 Result

This MLOps system achieves:
- Automated **data-to-deployment** workflow  
- **Reproducible** experiments via MLflow  
- Continuous **model drift monitoring** via Grafana  
- Modular, extensible architecture using **Airflow DAG orchestration**

---

## 👨‍💻 Author

**Patrick Yip Pak Kei**  
Master of IT in Business (AI Track) – Singapore Management University  
Project: *Credit Risk Inference Pipeline with Automated Retraining & Monitoring*
