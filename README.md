# 🛍️ Store Sales - Time Series Forecasting (MLOps Project)

This project is based on the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). It demonstrates a full MLOps pipeline using modern tools like **DVC**, **MLflow**, and modular **Pipelines**.

In order to view the pipeline, the metrics.csv, and the pickle files, you need to visit my [DagsHub](https://dagshub.com/RePlay-h/StoreSales-MLOps)

## 🔧 Project Structure
```
## 🔧 Project Structure

store-sales-forecasting-mlops/
├── .dvc/ # DVC internals
├── data/
│ ├── processed/ # Processed data (tracked by DVC)
│ │ ├── data.csv
│ │ └── data.csv.dvc
│ └── raw/ # Raw input data (also DVC-tracked)
│ ├── holidays_events.csv
│ ├── oil.csv
│ ├── sample_submission.csv
│ ├── stores.csv
│ ├── test.csv
│ ├── train.csv
│ ├── transactions.csv
│ ├── holidays_events.csv.dvc
│ ├── oil.csv.dvc
│ ├── sample_submission.csv.dvc
│ ├── stores.csv.dvc
│ ├── test.csv.dvc
│ ├── train.csv.dvc
│ └── transactions.csv.dvc
├── metrics/
│ └── metrics.csv # Model evaluation metrics
├── model/
│ ├── rf.pkl # Random Forest model
│ └── xgb.pkl # XGBoost model
├── src/
│ ├── preprocess.py # Data cleaning and feature engineering
│ └── train.py # Model training pipeline
├── venv/ # Virtual environment (should be in .gitignore)
├── .dvcignore # DVC ignore file
├── .gitignore # Git ignore file
├── EDA.ipynb # Exploratory data analysis notebook
├── params.yaml # Pipeline configuration and hyperparameters
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

## 🚀 Pipeline Overview

The end-to-end pipeline includes:

1. **Data Ingestion**: Load and version raw data using DVC.
2. **Preprocessing**: Clean and prepare time series data for modeling.
3. **Feature Engineering**: Extract meaningful time-based features (e.g., holidays, promotions).
4. **Model Training**: Train forecasting models.
5. **Experiment Tracking**: All experiments are tracked using MLflow.
6. **Model Versioning**: Models and metrics are versioned using DVC and MLflow.

## 📦 DVC

DVC (Data Version Control) is used to:

- Track data files and models
- Define and reproduce ML pipelines
- Share experiments across team members

## 📊 MLflow

MLflow is used for:

- Tracking experiments (metrics, parameters, models)
- Comparing model versions
- Serving and registering models

## ⚙️ How to Run

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the DVC pipeline**:
    ```bash
    dvc repro
    ```

3. **Track experiments with MLflow UI**:
    ```bash
    mlflow ui
    ```

## 📈 Results
Thanks to this project, I studied the work of the XGBoost model, and also compared it with the randomForest model. Based on the results, the XGB models work much more accurately with time series.