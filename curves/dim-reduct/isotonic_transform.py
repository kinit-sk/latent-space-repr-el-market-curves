import polars as pl
from pathlib import Path
import numpy as np
import pickle as pkl
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm
import typer

from curves.config import load_config, PROJ_ROOT, PROCESSED_DATA_DIR
from curves.dataset import preprocess_dataset_for_dim_reduct

app = typer.Typer()


@app.command()
def apply_isotonic_transformation():
    """Apply isotonic regression transformation to reconstructed predictions."""
    
    # Load config
    config = load_config(PROJ_ROOT / "config.yml")

    # Variables from config
    dataset_prefix = config["preprocessing"]["dataset"]
    train_start = pl.datetime(*config["preprocessing"]["train_start"])
    train_end = pl.datetime(*config["preprocessing"]["train_end"])
    val_start = pl.datetime(*config["preprocessing"]["val_start"])
    val_end = pl.datetime(*config["preprocessing"]["val_end"])
    test_start = pl.datetime(*config["preprocessing"]["test_start"])
    test_end = pl.datetime(*config["preprocessing"]["test_end"])
    method = config["dim_reduct"]["method"]
    n_components_s = config["dim_reduct"][method]["supply_n_components"]
    n_components_d = config["dim_reduct"][method]["demand_n_components"]   

    # Path creation
    dataset_filename_supply = dataset_prefix + "_processed_supply_curves.csv"
    dataset_filename_demand = dataset_prefix + "_processed_demand_curves.csv"
    output_file_s = f"{dataset_prefix}_{method}_{n_components_s}_reconstructed_test_supply.csv"
    output_file_d = f"{dataset_prefix}_{method}_{n_components_d}_reconstructed_test_demand.csv"
    latent_file_s = f"{dataset_prefix}_{method}_{n_components_s}_supply.pkl"
    latent_file_d = f"{dataset_prefix}_{method}_{n_components_d}_demand.pkl"
    
    # Load preprocessed data
    df_s = pl.read_csv(
        PROCESSED_DATA_DIR / dataset_filename_supply,
        schema_overrides={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    df_d = pl.read_csv(
        PROCESSED_DATA_DIR / dataset_filename_demand,
        schema_overrides={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    # Load predictions
    pred_s = pl.read_csv(
        PROCESSED_DATA_DIR / output_file_s,
        schema_overrides={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    pred_d = pl.read_csv(
        PROCESSED_DATA_DIR / output_file_d,
        schema_overrides={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    # Load pickle object of latent space
    with open(PROJ_ROOT / "models" / latent_file_s, "rb") as f:
        latent_s = pkl.load(f)

    with open(PROJ_ROOT / "models" / latent_file_d, "rb") as f:
        latent_d = pkl.load(f)

    # Preprocess datasets
    df_train_s, _, df_test_s, means_s, stds_s, _, _, dates_test_s = (
        preprocess_dataset_for_dim_reduct(
            df_s,
            train_start,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
        )
    )

    df_train_d, _, df_test_d, means_d, stds_d, _, _, dates_test_d = (
        preprocess_dataset_for_dim_reduct(
            df_d,
            train_start,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
        )
    )

    # Create train data for isotonic regression
    if method == "autoencoder":
        df_train_recon_s = (latent_s.predict(df_train_s.astype(np.float32)) * stds_s + means_s).flatten()
        df_train_recon_d = (latent_d.predict(df_train_d.astype(np.float32)) * stds_d + means_d).flatten()
    else:
        df_train_recon_s = (latent_s.inverse_transform(latent_s.transform(df_train_s)) * stds_s + means_s).flatten()
        df_train_recon_d = (latent_d.inverse_transform(latent_d.transform(df_train_d)) * stds_d + means_d).flatten()

    y_train_s = (df_train_s * stds_s + means_s).flatten()
    y_train_d = (df_train_d * stds_d + means_d).flatten()
    
    # Train isotonic regression models
    iso_d = IsotonicRegression(
        y_min=df_d["price"].min(), 
        y_max=df_d["price"].max(), 
        out_of_bounds='clip',
        increasing=True,
        )

    iso_s = IsotonicRegression(
        y_min=df_s["price"].min(), 
        y_max=df_s["price"].max(), 
        out_of_bounds='clip',
        increasing=True,
    )

    iso_d.fit(df_train_recon_d, y_train_d)
    iso_s.fit(df_train_recon_s, y_train_s)
    
    # Get unique test dates
    test_dates = pred_s["datetime"].unique().sort()
    
    # Pre-allocate numpy arrays for speed
    mono_prices_s = np.zeros(len(pred_s))
    mono_prices_d = np.zeros(len(pred_d))
    
    # Process each datetime
    start_idx_s = 0
    start_idx_d = 0
    
    for date in tqdm(test_dates, total=len(test_dates)):
        # Get data for this datetime
        date_data_s = pred_s.filter(pl.col("datetime") == date)
        date_data_d = pred_d.filter(pl.col("datetime") == date)
        
        n_points_s = len(date_data_s)
        n_points_d = len(date_data_d)
        
        # Transform using isotonic regression
        mono_s = iso_s.transform(date_data_s["price"].to_numpy())
        mono_d = iso_d.transform(date_data_d["price"].to_numpy())
        
        # Store in pre-allocated arrays
        mono_prices_s[start_idx_s:start_idx_s + n_points_s] = mono_s
        mono_prices_d[start_idx_d:start_idx_d + n_points_d] = mono_d
        
        start_idx_s += n_points_s
        start_idx_d += n_points_d
    
    # Add monotonic prices as new columns
    pred_s = pred_s.with_columns(pl.Series("price", mono_prices_s))
    pred_d = pred_d.with_columns(pl.Series("price", mono_prices_d))
    
    # Save results
    mono_file_s = f"{dataset_prefix}_{method}_{n_components_s}_reconstructed_test_supply_mono.csv"
    mono_file_d = f"{dataset_prefix}_{method}_{n_components_d}_reconstructed_test_demand_mono.csv"
    
    pred_s.write_csv(PROCESSED_DATA_DIR / mono_file_s)
    pred_d.write_csv(PROCESSED_DATA_DIR / mono_file_d)
    
    print(f"Saved monotonic results to {mono_file_s} and {mono_file_d}")
    
    return pred_s, pred_d

if __name__ == "__main__":
    app()