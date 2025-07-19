import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

import os
import pickle

from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/RePlay-h/StoreSales-MLOps.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'RePlay-h'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'ae89348619ef0ef08c309046ac149c61ae0d21a2'


def random_sample(input_path):
    df = pd.read_csv(input_path)

    ## Random sample
    sample_frac = 0.10
    data_sample = df.sample(frac=sample_frac, random_state=101).reset_index(drop=True)
    
    return data_sample

def hyperparameter_tuning(model, parameters, tscv, X_train, y_train, n_iters):
    
    rand_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=n_iters,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=101,
        verbose=1,
        error_score='raise'
    )

    rand_model.fit(X_train, y_train)
    
    return rand_model.best_estimator_, rand_model.best_params_


## Log parameters with mlflow
def log_parameters_and_metrics(model, best_params, model_name, model_path, signature, splits, records):   
    with mlflow.start_run():
        ## Calculate metrics and save them
        for split_name, (X, y) in splits.items():
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            records.append({
            "Model": model_name,
            "Split": split_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse)
            })

            mlflow.log_metric(f"MAE_{split_name}", mae)
            mlflow.log_metric(f"MSE_{split_name}", mse)
            mlflow.log_metric(f"RMSE_{split_name}", np.sqrt(mse))

        ## Log the model
        mlflow.log_params(best_params)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        ## Save the model
        if tracking_url_type_store != 'file':
            if model_name == "XGB":
                mlflow.xgboost.log_model(model, f"{model_name}_model", registered_model_name=f"Best {model_name} model")
            else:
                mlflow.sklearn.log_model(model, f"{model_name}_model", registered_model_name=f"Best {model_name} model")
        else:
            if model_name == "XGB":
                mlflow.xgboost.log_model(model, f"{model_name}_model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, f"{model_name}_model", signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(model, open(model_path, "wb"))

        print(f"Save {model_name} into pickle-file")

def train(df, save_rf_path, save_xgb_path):

    ## Prepare data for modeling
    X = df.drop('sales', axis=1)
    y = df['sales']

    ## Split data into train (60%), validation (20%), test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)

    ## Normalize features
    scaler_features = StandardScaler()
    X_train = scaler_features.fit_transform(X_train)
    X_test = scaler_features.transform(X_test)
    X_val = scaler_features.transform(X_val)

    splits = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test)
    }

    tscv = TimeSeriesSplit(n_splits=3)


    ## Save metrics
    records = []

    ## Track all model parameters and metrics
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    ## XGBoost Tuning
    xgb = XGBRegressor(eval_metric='mae', random_state=101)

    signature = infer_signature(X_test, y_test)

    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 2, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.4],
        'colsample_bytree': [0.8, 0.4],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }
        
        ## Save best XGBoost model
    best_xgb, best_xgb_params = hyperparameter_tuning(xgb, xgb_params, tscv, X_train, y_train, 20)

    log_parameters_and_metrics(best_xgb, best_xgb_params, "XGB", save_xgb_path, signature, splits, records)
   
    ## RandomForest Tuning
    rf = RandomForestRegressor(random_state=101)

    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 2, 5],
        'min_samples_split': [20, 50, 75],
        'min_samples_leaf':  [10, 20, 30]
    }

    best_rf, best_rf_params = hyperparameter_tuning(rf, rf_params, tscv, X_train, y_train, 3)

    log_parameters_and_metrics(best_rf, best_rf_params, "RandomForest", save_rf_path, signature, splits, records)

    return records

def metrics_table(records, save_path):
    ## Create metrics_table
    table = pd.DataFrame(records)
    ## Create a folder for the csv file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    table.to_csv(save_path, index=False)
    print(table)


if __name__ == '__main__':

    ## Get random data sample
    params = yaml.safe_load(open('params.yaml'))['train']
    print(params['input'])
    rand_sample = random_sample(params['input'])

    ## Train the models
    records = train(rand_sample, params['save_rf_path'], params['save_xgb_path'])

    ## Create metrics table
    metrics_table(records, params['save_metrics'])

   