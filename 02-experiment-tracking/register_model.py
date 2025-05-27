import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            if param in params:
                new_params[param] = int(params[param])
            elif param == 'random_state':
                new_params[param] = 42

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Calcul du RMSE sur l'ensemble de validation en deux étapes
        val_pred = rf.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        mlflow.log_metric("val_rmse", val_rmse)
        
        # Calcul du RMSE sur l'ensemble de test en deux étapes
        test_pred = rf.predict(X_test)
        test_rmse = root_mean_squared_error(y_test, test_pred)
        mlflow.log_metric("test_rmse", test_rmse)
        
        return test_rmse


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    
    # Dictionnaire pour stocker le RMSE de test pour chaque run
    run_id_to_test_rmse = {}
    
    for run in runs:
        test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        run_id_to_test_rmse[run.info.run_id] = test_rmse

    # Sélection du modèle avec le RMSE de test le plus bas
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]
    
    print(f"Best RMSE: {best_run.data.metrics['test_rmse']}")
    
    # Enregistrement du meilleur modèle dans le registre
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="nyc-taxi-duration-predictor")


if __name__ == '__main__':
    run_register_model()