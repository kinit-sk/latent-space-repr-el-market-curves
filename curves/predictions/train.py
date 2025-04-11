from pathlib import Path

from darts import TimeSeries
from darts.dataprocessing.transformers import scaler
from darts.models import RNNModel, TSMixerModel
from darts.utils import missing_values
from loguru import logger
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import PowerTransformer
from torch.nn import L1Loss
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
)
import typer
import wandb
import yaml

from curves.config import EXTERNAL_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, load_config

app = typer.Typer()


def train_lstm(
    train, val, futcov_train, futcov_val, runname, n_epochs, model_args, pl_trainer_kwargs
):
    """
    Train LSTM model with the given arguments and callbacks. Function serves purpose only for training.
    Args:
        train (TimeSeries): Training data.
        val (TimeSeries): Validation data.
        futcov_train (TimeSeries): Training future covariates.
        futcov_val (TimeSeries): Validation future covariates.
        runname (str): Name of the experiment.
        n_epochs (int): Number of epochs.
        model_args (dict): Arguments for the model.
        pl_trainer_kwargs (dict): Arguments for Lightning Trainer.
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
        n_epochs=n_epochs,
        training_length=model_args["input_chunk_length"] + 24,
        input_chunk_length=model_args["input_chunk_length"],
        batch_size=model_args["batch_size"],
        hidden_dim=model_args["hidden_dim"],
        n_rnn_layers=model_args["n_rnn_layers"],
        dropout=model_args["dropout"],
        random_state=model_args["random_state"],
        add_encoders=model_args["add_encoders"],
        model="LSTM",
        model_name=runname,
        loss_fn=L1Loss(),
        torch_metrics=torch_metrics,
        pl_trainer_kwargs=pl_trainer_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )

    # fit the model
    model.fit(
        train,
        future_covariates=futcov_train,
        val_series=val,
        val_future_covariates=futcov_val,
    )


def train_tsmixer(
    train, val, futcov_train, futcov_val, runname, n_epochs, model_args, pl_trainer_kwargs
):
    """
    Train TSMixer model with the given arguments and callbacks. Function serves purpose only for training.
    Args:
        train (TimeSeries): Training data.
        val (TimeSeries): Validation data.
        futcov_train (TimeSeries): Training future covariates.
        futcov_val (TimeSeries): Validation future covariates.
        runname (str): Name of the experiment.
        n_epochs (int): Number of epochs.
        model_args (dict): Arguments for the model.
        pl_trainer_kwargs (dict): Arguments for Lightning Trainer.
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
        n_epochs=n_epochs,
        output_chunk_length=24,
        model_name=runname,
        normalize_before=False,  # from TSMixer ext part of paper
        save_checkpoints=True,
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
        pl_trainer_kwargs=pl_trainer_kwargs,
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
    gpu = config["predictions"]["gpu"]
    runname = config["predictions"]["training"]["run_name"]
    patience = config["predictions"]["training"]["patience"]
    min_delta = config["predictions"]["training"]["min_delta"]
    n_epochs = config["predictions"]["training"]["n_epochs"]

    logger.info(
        f"Starting training for {modelname} for {which_curves} curves on {dataset_prefix} data..."
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

    # hyperparameter filename
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

    logger.info(f"Reading training file {filename_train}...")
    logger.info(f"Reading validation file {filename_val}...")
    logger.info(f"Reading covariates file {filename_cov}...")
    logger.info(f"Reading hyperparameters file {hp_filename}...")

    # load the files
    df_train = pd.read_csv(input_path / filename_train)
    df_val = pd.read_csv(input_path / filename_val)
    covariates = pd.read_csv(covariates_path / filename_cov)

    with open(model_path / "best_hp" / hp_filename, "r") as f:
        best_hp = yaml.safe_load(f)

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

    # define Lightning EarlyStopping criterion
    my_stopper = EarlyStopping(
        monitor="val_mae",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    # define Lightning WandB logger
    wandblogger = WandbLogger(project=runname)

    # PyTorch Lightning training definition
    if gpu == 1:
        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "logger": [wandblogger],
            "accelerator": "gpu",
            "devices": [0],
        }
    else:
        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "logger": [wandblogger],
            "accelerator": "cpu",
        }

    if modelname == "tsmixer":
        train_tsmixer(
            train=target_train,
            val=target_val,
            futcov_train=covariates_train,
            futcov_val=covariates_val,
            runname=runname,
            n_epochs=n_epochs,
            model_args=best_hp,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )

        wandb.finish()

    elif modelname == "lstm":
        train_lstm(
            train=target_train,
            val=target_val,
            futcov_train=covariates_train,
            futcov_val=covariates_val,
            runname=runname,
            n_epochs=n_epochs,
            model_args=best_hp,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )

        wandb.finish()
    else:
        raise ValueError("Model name not recognized. Please use 'tsmixer' or 'lstm'.")

    logger.success(f"Finished training for run {runname}")


if __name__ == "__main__":
    app()
