#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import prefect
from prefect import flow, task

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

### Q1: What's the name of the orchestrator you chose?

print(">> Orchestrator: Prefect")

### Q2: What's the version of the orchestrator?
print(">> Prefect version:", prefect.__version__)

### Q3: Creating a pipeline. Let's read the March 2023 Yellow taxi trips data. How many records did we load?
def load_data(url):
    
    df = pd.read_parquet(url)

    print(f">> The number of files loaded by Prefect is {df.shape[0]}.")
    return df

df = load_data(url)


### Q4: Data preparation
@task
def read_dataframe(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

yellow_taxi_df = read_dataframe(df)
print(f">> The size of the dataframe after filtering: {yellow_taxi_df.shape[0]}.")

### Q5: Train a model
@flow(log_prints=True)
def transform_dataframe(df):
    dv = DictVectorizer()
    pu_do_loc_df = df[['PULocationID', 'DOLocationID']].astype(str)
    pu_do_loc_dict = pu_do_loc_df.to_dict(orient='records')
    pu_do_loc_arr = dv.fit_transform(pu_do_loc_dict)

    X = pu_do_loc_arr.copy()
    y = df['duration'].copy()

    lr = LinearRegression()
    lr.fit(X, y)
    
    return lr.intercept_


y_intercept = transform_dataframe(yellow_taxi_df)
print(f">> The y-intercept field is {np.round(y_intercept, 5)}")

### Q6: Register the model
EXPERIMENT_NAME = "nyc-taxi-experiment"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@task(log_prints=True)
def transform_dataframe(df):
    with mlflow.start_run():
        dv = DictVectorizer()
        pu_do_loc_df = df[['PULocationID', 'DOLocationID']].astype(str)
        pu_do_loc_dict = pu_do_loc_df.to_dict(orient='records')
        pu_do_loc_arr = dv.fit_transform(pu_do_loc_dict)

        X = pu_do_loc_arr.copy()
        y = df['duration'].copy()

        lr = LinearRegression()
        lr.fit(X, y)
        
        mlflow.sklearn.log_model(lr, artifact_path="model", registered_model_name="yellow-taxi-linear-regressor")

        return lr

lr = transform_dataframe(yellow_taxi_df)

@flow(log_prints=True)
def register_model():

    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
        )
    
    # Register the model
    print(f"run id: {runs[0].info.run_id}")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(model_uri)

    mlflow.register_model(model_uri, name="yellow-taxi-linear-regressor")


register_model()