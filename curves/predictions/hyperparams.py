from pathlib import Path

from darts import TimeSeries
from darts.dataprocessing.transformers import scaler
from darts.models import RNNModel, TSMixerModel
from darts.utils import missing_values
from loguru import logger
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import PowerTransformer
from torch.nn import L1Loss
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
)
import typer
import yaml

from curves.config import EXTERNAL_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, load_config

app = typer.Typer()


def train_lstm(model_args, callbacks, train, val, futcov_train, futcov_val):
    """
    Train LSTM model with the given arguments and callbacks. Function serves purpose only for hyperparameter optimization.
    Args:
        model_args (dict): Arguments for the model.
        callbacks (list): List of callbacks for the model.
        train (TimeSeries): Training data.
        val (TimeSeries): Validation data.
        futcov_train (TimeSeries): Training future covariates.
        futcov_val (TimeSeries): Validation future covariates.
    """

    # definic MetricCollection to track progress of training
    torch_metrics = MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "smape": SymmetricMeanAbsolutePercentageError(),
        }
    )

    # define optimizer with learning rate found by lr finder
    optimizer_kwargs = {"lr": model_args["lr"]}

    # Create the model using model_args provided from Ray Tune
    model = RNNModel(
        n_epochs=50,
        training_length=model_args["input_chunk_length"] + 24,
        input_chunk_length=model_args["input_chunk_length"],
        batch_size=model_args["batch_size"],
        hidden_dim=model_args["hidden_dim"],
        n_rnn_layers=model_args["n_rnn_layers"],
        dropout=model_args["dropout"],
        random_state=model_args["random_state"],
        add_encoders=model_args["add_encoders"],
        model="LSTM",
        model_name="lstm",
        loss_fn=L1Loss(),
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        optimizer_kwargs=optimizer_kwargs,
    )

    # fit the model
    model.fit(
        train,
        future_covariates=futcov_train,
        val_series=val,
        val_future_covariates=futcov_val,
    )


def train_tsmixer(model_args, callbacks, train, val, futcov_train, futcov_val):
    """
    Train TSMixer model with the given arguments and callbacks. Function serves purpose only for hyperparameter optimization.
    Args:
        model_args (dict): Arguments for the model.
        callbacks (list): List of callbacks for the model.
        train (TimeSeries): Training data.
        val (TimeSeries): Validation data.
        futcov_train (TimeSeries): Training future covariates.
        futcov_val (TimeSeries): Validation future covariates.
    """

    # definic MetricCollection to track progress of training
    torch_metrics = MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "smape": SymmetricMeanAbsolutePercentageError(),
        }
    )

    # define optimizer with learning rate found by lr finder
    optimizer_kwargs = {"lr": model_args["lr"]}

    # Create the model using model_args provided from Ray Tune
    model = TSMixerModel(
        n_epochs=50,
        output_chunk_length=24,
        model_name="tsmixer",
        normalize_before=False,  # from TSMixer ext part of paper
        input_chunk_length=model_args["input_chunk_length"],
        ff_size=model_args["hidden_size"],
        hidden_size=model_args["hidden_size"],
        batch_size=model_args["batch_size"],
        num_blocks=model_args["num_blocks"],
        dropout=model_args["dropout"],
        random_state=model_args["random_state"],
        add_encoders=model_args["add_encoders"],
        loss_fn=L1Loss(),
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        optimizer_kwargs=optimizer_kwargs,
    )

    # fit the model
    model.fit(
        train,
        future_covariates=futcov_train,
        val_series=val,
        val_future_covariates=futcov_val,
    )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
    config_path: Path = PROJ_ROOT,
    covariates_path: Path = EXTERNAL_DATA_DIR,
    model_path: Path = MODELS_DIR,
) -> None:
    # load config
    config = load_config(config_path / "config.yml")

    # variables from config
    dataset_prefix = config["preprocessing"]["dataset"]
    which_curves = config["predictions"]["which_curves"]
    dim_reduct_method = config["predictions"]["which_dim_reduct_method"]
    n_components = config["predictions"]["n_components"]
    modelname = config["predictions"]["model"]
    cpu = config["predictions"]["cpu"]
    gpu = config["predictions"]["gpu"]
    run_name = config["predictions"]["hyperparameter_optimization"]["run_name"]
    num_samples = config["predictions"]["hyperparameter_optimization"]["num_samples"]
    max_t = config["predictions"]["hyperparameter_optimization"]["max_t"]
    patience = config["predictions"]["hyperparameter_optimization"]["patience"]
    min_delta = config["predictions"]["hyperparameter_optimization"]["min_delta"]

    logger.info(
        f"Starting hyperparameter optimization for {modelname} for {which_curves} curves on {dataset_prefix} data..."
    )

    # filename creation
    filename_train = (
        dataset_prefix
        + "_"
        + dim_reduct_method
        + "_"
        + str(n_components)
        + "_train_"
        + which_curves
        + ".csv"
    )
    filename_val = (
        dataset_prefix
        + "_"
        + dim_reduct_method
        + "_"
        + str(n_components)
        + "_val_"
        + which_curves
        + ".csv"
    )
    filename_cov = dataset_prefix + "_all_covariates.csv"

    logger.info(f"Reading training file {filename_train}...")
    logger.info(f"Reading validation file {filename_val}...")
    logger.info(f"Reading covariates file {filename_cov}...")

    # load the files
    df_train = pd.read_csv(input_path / filename_train)
    df_val = pd.read_csv(input_path / filename_val)
    covariates = pd.read_csv(covariates_path / filename_cov)

    logger.success("Files loaded succesfully.")

    # convert datetime columns to datetime
    df_train["datetime"] = pd.to_datetime(df_train["datetime"])
    df_val["datetime"] = pd.to_datetime(df_val["datetime"])
    covariates["datetime"] = pd.to_datetime(covariates["datetime"], format="%Y-%m-%dT%H:%M:%SZ")

    # reformat the data to TimeSeries class of darts library
    # target - transformed curves
    target_train = TimeSeries.from_dataframe(
        df_train,
        time_col="datetime",
        value_cols=df_train.loc[:, df_train.columns != "datetime"].columns.tolist(),
        freq="1h",
    )
    target_val = TimeSeries.from_dataframe(
        df_val,
        time_col="datetime",
        value_cols=df_val.loc[:, df_val.columns != "datetime"].columns.tolist(),
        freq="1h",
    )

    # future covariates - forecasts of Wind, Solar and load
    covariates = TimeSeries.from_dataframe(
        covariates,
        time_col="datetime",
        value_cols=covariates.loc[:, covariates.columns != "datetime"].columns.tolist(),
        freq="1h",
    )

    # fill missing gaps (created by daytime saving)
    target_train = missing_values.fill_missing_values(target_train)
    target_val = missing_values.fill_missing_values(target_val)

    # split covariates
    covariates_train = covariates.slice(target_train.start_time(), target_train.end_time())
    covariates_val = covariates.slice(target_val.start_time(), target_val.end_time())

    # fit transformer of data on train (Yeo-Johnson)
    transformer_target = scaler.Scaler(PowerTransformer()).fit(target_train)
    transformer_future = scaler.Scaler(PowerTransformer()).fit(covariates_train)

    # transform the data
    target_train = transformer_target.transform(target_train)
    covariates_train = transformer_future.transform(covariates_train)

    target_val = transformer_target.transform(target_val)
    covariates_val = transformer_future.transform(covariates_val)

    # add dayofweek as future covariate to the model (their options)
    add_encoders1 = {
        "cyclic": {"future": ["dayofweek", "month", "hour"]},
        "tz": "UTC",
    }

    add_encoders2 = {
        "cyclic": {"future": ["dayofweek"]},
        "tz": "UTC",
    }

    add_encoders3 = {
        "cyclic": {"future": ["month"]},
        "tz": "UTC",
    }

    add_encoders4 = {
        "cyclic": {"future": ["hour"]},
        "tz": "UTC",
    }

    add_encoders5 = {
        "cyclic": {"future": ["dayofweek", "month"]},
        "tz": "UTC",
    }

    add_encoders6 = {
        "cyclic": {"future": ["month", "hour"]},
        "tz": "UTC",
    }

    add_encoders7 = {
        "cyclic": {"future": ["dayofweek", "hour"]},
        "tz": "UTC",
    }

    # define Lightning EarlyStopping criterion
    my_stopper = EarlyStopping(
        monitor="val_mae",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    # set up ray tune callback
    tune_callback = TuneReportCallback(
        {
            "loss": "val_loss",
            "MSE": "val_mse",
            "MAE": "val_mae",
            "sMAPE": "val_smape",
        },
        on="validation_end",
    )

    if modelname == "tsmixer":
        hyperparams = config["predictions"]["hyperparameter_optimization"][
            "tsmixer_hyperparameter_space"
        ]
        # initialize raytune with the train function, callbacks and data
        train_fn_with_parameters = tune.with_parameters(
            train_tsmixer,
            callbacks=[my_stopper, tune_callback],
            train=target_train,
            val=target_val,
            futcov_train=covariates_train,
            futcov_val=covariates_val,
        )

        # define the hyperparameter space
        hp_space = {
            "input_chunk_length": tune.choice(hyperparams["input_chunk_length"]),
            "batch_size": tune.choice(hyperparams["batch_size"]),
            "hidden_size": tune.choice(hyperparams["hidden_size"]),
            "num_blocks": tune.choice(hyperparams["num_blocks"]),
            "dropout": tune.choice(hyperparams["dropout"]),
            "random_state": tune.choice(hyperparams["random_state"]),
            "lr": tune.choice(hyperparams["learning_rate"]),
            "add_encoders": tune.choice(
                [
                    None,
                    add_encoders1,
                    add_encoders2,
                    add_encoders3,
                    add_encoders4,
                    add_encoders5,
                    add_encoders6,
                    add_encoders7,
                ]
            ),
        }
    elif modelname == "lstm":
        hyperparams = config["predictions"]["hyperparameter_optimization"][
            "lstm_hyperparameter_space"
        ]
        # initialize raytune with the train function, callbacks and data
        train_fn_with_parameters = tune.with_parameters(
            train_lstm,
            callbacks=[my_stopper, tune_callback],
            train=target_train,
            val=target_val,
            futcov_train=covariates_train,
            futcov_val=covariates_val,
        )

        # define the hyperparameter space
        hp_space = {
            "input_chunk_length": tune.choice(hyperparams["input_chunk_length"]),
            "batch_size": tune.choice(hyperparams["batch_size"]),
            "hidden_dim": tune.choice(hyperparams["hidden_dim"]),
            "n_rnn_layers": tune.choice(hyperparams["n_rnn_layers"]),
            "dropout": tune.choice(hyperparams["dropout"]),
            "random_state": tune.choice(hyperparams["random_state"]),
            "lr": tune.choice(hyperparams["learning_rate"]),
            "add_encoders": tune.choice(
                [
                    None,
                    add_encoders1,
                    add_encoders2,
                    add_encoders3,
                    add_encoders4,
                    add_encoders5,
                    add_encoders6,
                    add_encoders7,
                ]
            ),
        }
    else:
        raise ValueError("Model name not recognized. Please use 'tsmixer' or 'lstm'.")

    # define Command Line Reporter to report current parameters metrics and training progress
    reporter = CLIReporter(
        parameter_columns=list(hp_space.keys()),
        metric_columns=["loss", "MSE", "MAE", "sMAPE", "training_iteration"],
    )

    # resources to allocate to each trial
    resources_per_trial = {"cpu": cpu, "gpu": gpu}

    # scheduler for RayTune
    scheduler = ASHAScheduler(max_t=max_t)

    # run raytune and evaluate the hyperparameter combination on val MAE
    analysis = tune.run(
        train_fn_with_parameters,  # initialized tuning via with_parameters
        resources_per_trial=resources_per_trial,
        metric="MAE",  # any value in TuneReportCallback
        mode="min",
        config=hp_space,  # config of hyperparameters
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=MODELS_DIR / "ray_results",
        name=run_name,
    )

    logger.success(f"Best hyperparameters found were: \n{analysis.best_config}")

    hp_filename = (
        dataset_prefix
        + "_"
        + modelname
        + "_"
        + dim_reduct_method
        + "_"
        + str(n_components)
        + "_"
        + which_curves
        + ".yml"
    )

    logger.info(f"Saving the {hp_filename} best hyperparams to {model_path}...")

    with open(model_path / "best_hp" / hp_filename, "w") as f:
        yaml.dump(analysis.best_config, f, default_flow_style=False)

    logger.success(
        f"Hyperparameter optimization for {modelname} for {which_curves} curves on {dataset_prefix} data is DONE."
    )


if __name__ == "__main__":
    app()
