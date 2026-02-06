from pathlib import Path
import pickle as pkl
from typing_extensions import List

from datetime import datetime
from loguru import logger
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typer
import umap
import yaml

from curves.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, load_config
from curves.dataset import preprocess_dataset_for_dim_reduct


app = typer.Typer()


@app.command()
def moving_window_dim_reduct_models(
    input_path: Path = PROCESSED_DATA_DIR,
    config_path: Path = PROJ_ROOT,
    output_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR,
) -> None:
    logger.info("Starting moving window dimensionality reduction...")
    config = load_config(config_path / "config.yml")

    # Prepare parameters
    dataset_prefix = config["preprocessing"]["dataset"]
    test_start = pl.datetime(*config["preprocessing"]["test_start"])
    test_end = pl.datetime(*config["preprocessing"]["test_end"])
    method = config["dim_reduct"]["method"]
    n_components_s = config["dim_reduct"][method]["supply_n_components"]
    n_components_d = config["dim_reduct"][method]["demand_n_components"]
    interval = config["dim_reduct"]["evaluation"]["retrain_interval"]
    modelname_s = dataset_prefix + "_" + method + "_" + str(n_components_s) + "_supply.pkl"
    modelname_d = dataset_prefix + "_" + method + "_" + str(n_components_d) + "_demand.pkl"

    # Load models
    with open(model_path / modelname_d, "rb") as f:
        model_d = pkl.load(f)

    with open(model_path / modelname_s, "rb") as f:
        model_s = pkl.load(f)

    # Load data
    df_s = pl.read_csv(input_path / f"{dataset_prefix}_processed_supply_curves.csv",
                       schema={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})
    df_d = pl.read_csv(input_path / f"{dataset_prefix}_processed_demand_curves.csv",
                       schema={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})

    # Create empty result dataframes with datetime, volume, price structure
    test_data_s = df_s.filter(pl.col("datetime").is_between(test_start, test_end))
    test_data_d = df_d.filter(pl.col("datetime").is_between(test_start, test_end))
    
    results_s = test_data_s.select(["datetime", "volume"]).with_columns(pl.lit(0.0).alias("price"))
    results_d = test_data_d.select(["datetime", "volume"]).with_columns(pl.lit(0.0).alias("price"))
    
    # Create latent space result dataframes
    test_dates = test_data_s["datetime"].unique().sort()
    latent_results_s = pl.DataFrame({
        "datetime": test_dates,
        **{f"{method}{i+1}": [0.0] * len(test_dates) for i in range(n_components_s)}
    })
    latent_results_d = pl.DataFrame({
        "datetime": test_dates,
        **{f"{method}{i+1}": [0.0] * len(test_dates) for i in range(n_components_d)}
    })

    retrain_dates = pl.date_range(test_start, test_end, interval=interval, closed="left", eager=True)
    
    for i, retrain_date in enumerate(retrain_dates):
        next_retrain = retrain_dates[i+1] if i+1 < len(retrain_dates) else test_end
        logger.info(f"Processing period {retrain_date} to {next_retrain}")
        
        # Get training data up to retrain_date
        train_data_s = df_s.filter(pl.col("datetime") < retrain_date)
        train_data_d = df_d.filter(pl.col("datetime") < retrain_date)
        
        # Standardize training data
        unique_dates_s = train_data_s["datetime"].unique().sort()
        n_features = len(train_data_s) // len(unique_dates_s)
        train_matrix_s = train_data_s["price"].to_numpy().reshape(len(unique_dates_s), n_features)
        means_s, stds_s = train_matrix_s.mean(axis=0), train_matrix_s.std(axis=0)
        train_matrix_s = (train_matrix_s - means_s) / stds_s
        
        unique_dates_d = train_data_d["datetime"].unique().sort()
        train_matrix_d = train_data_d["price"].to_numpy().reshape(len(unique_dates_d), n_features)
        means_d, stds_d = train_matrix_d.mean(axis=0), train_matrix_d.std(axis=0)
        train_matrix_d = (train_matrix_d - means_d) / stds_d
        
        # Fit models
        if method == "autoencoder":
            model_s.fit(train_matrix_s.astype(np.float32), train_matrix_s.astype(np.float32))
            model_d.fit(train_matrix_d.astype(np.float32), train_matrix_d.astype(np.float32))
        else:
            model_s.fit(train_matrix_s)
            model_d.fit(train_matrix_d)
        
        # Process test data for this period
        period_data_s = df_s.filter(pl.col("datetime").is_between(retrain_date, next_retrain, closed="left"))
        period_data_d = df_d.filter(pl.col("datetime").is_between(retrain_date, next_retrain, closed="left"))
        
        if len(period_data_s) > 0:
            period_dates = period_data_s["datetime"].unique().sort()
            period_matrix_s = period_data_s["price"].to_numpy().reshape(len(period_dates), n_features)
            period_matrix_s = (period_matrix_s - means_s) / stds_s
            
            period_matrix_d = period_data_d["price"].to_numpy().reshape(len(period_dates), n_features)
            period_matrix_d = (period_matrix_d - means_d) / stds_d
            
            # Transform and inverse transform
            if method == "autoencoder":
                latent_s = model_s.forward(period_matrix_s.astype(np.float32))[1].detach().numpy()
                reconstructed_s = model_s.predict(period_matrix_s.astype(np.float32)) * stds_s + means_s

                latent_d = model_d.forward(period_matrix_d.astype(np.float32))[1].detach().numpy()
                reconstructed_d = model_d.predict(period_matrix_d.astype(np.float32)) * stds_d + means_d
            else:
                latent_s = model_s.transform(period_matrix_s)
                reconstructed_s = model_s.inverse_transform(latent_s)
                reconstructed_s = reconstructed_s * stds_s + means_s
                
                latent_d = model_d.transform(period_matrix_d)
                reconstructed_d = model_d.inverse_transform(latent_d)
                reconstructed_d = reconstructed_d * stds_d + means_d
                
            # Update results with reconstructed data
            for j, date in enumerate(period_dates):
                period_s_data = period_data_s.filter(pl.col("datetime") == date)
                period_d_data = period_data_d.filter(pl.col("datetime") == date)
                
                # Update latent results
                for k in range(n_components_s):
                    latent_results_s = latent_results_s.with_columns(
                        pl.when(pl.col("datetime") == date)
                        .then(latent_s[j, k])
                        .otherwise(pl.col(f"{method}{k+1}"))
                        .alias(f"{method}{k+1}")
                    )
                
                for k in range(n_components_d):
                    latent_results_d = latent_results_d.with_columns(
                        pl.when(pl.col("datetime") == date)
                        .then(latent_d[j, k])
                        .otherwise(pl.col(f"{method}{k+1}"))
                        .alias(f"{method}{k+1}")
                    )
                
                for k, (_, volume) in enumerate(period_s_data.select(["datetime", "volume"]).iter_rows()):
                    results_s = results_s.with_columns(
                        pl.when((pl.col("datetime") == date) & (pl.col("volume") == volume))
                        .then(reconstructed_s[j, k])
                        .otherwise(pl.col("price"))
                        .alias("price")
                    )
                
                for k, (_, volume) in enumerate(period_d_data.select(["datetime", "volume"]).iter_rows()):
                    results_d = results_d.with_columns(
                        pl.when((pl.col("datetime") == date) & (pl.col("volume") == volume))
                        .then(reconstructed_d[j, k])
                        .otherwise(pl.col("price"))
                        .alias("price")
                    )
    
    # Save results
    output_file_s = f"{dataset_prefix}_{method}_{n_components_s}_reconstructed_test_supply.csv"
    output_file_d = f"{dataset_prefix}_{method}_{n_components_d}_reconstructed_test_demand.csv"
    latent_file_s = f"{dataset_prefix}_{method}_{n_components_s}_test_supply_retrain.csv"
    latent_file_d = f"{dataset_prefix}_{method}_{n_components_d}_test_demand_retrain.csv"
    
    results_s.write_csv(output_path / output_file_s)
    results_d.write_csv(output_path / output_file_d)
    latent_results_s.write_csv(output_path / latent_file_s)
    latent_results_d.write_csv(output_path / latent_file_d)
    
    logger.success(f"Saved reconstructed results to {output_file_s} and {output_file_d}")
    logger.success(f"Saved latent results to {latent_file_s} and {latent_file_d}")
   
if __name__ == "__main__":
    app()     