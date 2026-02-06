from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
import typer

from curves.config import PROCESSED_DATA_DIR, PROJ_ROOT, load_config

app = typer.Typer()

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate reconstruction metrics."""
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    bias = np.mean(predicted - actual)
    wape = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "WAPE": wape}


@app.command()
def evaluate_reconstruction(
    input_path: Path = PROCESSED_DATA_DIR,
    config_path: Path = PROJ_ROOT,
    output_path: Path = PROJ_ROOT / "reports",
) -> None:
    logger.info("Starting reconstruction evaluation...")
    config = load_config(config_path / "config.yml")
    
    dataset_prefix = config["preprocessing"]["dataset"]
    method = config["dim_reduct"]["method"]
    n_components_s = config["dim_reduct"][method]["supply_n_components"]
    n_components_d = config["dim_reduct"][method]["demand_n_components"]
    evaluate_monotonic = config["dim_reduct"]["evaluation"]["evaluate_monotonic"]
    test_start = pl.datetime(*config["preprocessing"]["test_start"])
    test_end = pl.datetime(*config["preprocessing"]["test_end"])
    
    # Load actual data
    actual_s = pl.read_csv(input_path / f"{dataset_prefix}_processed_supply_curves.csv",
                          schema_overrides={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})
    actual_d = pl.read_csv(input_path / f"{dataset_prefix}_processed_demand_curves.csv",
                          schema_overrides={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})
    
    # Load reconstructed data
    suffix_mono = "_mono" if evaluate_monotonic else ""

    recon_s = pl.read_csv(input_path / f"{dataset_prefix}_{method}_{n_components_s}_reconstructed_test_supply{suffix_mono}.csv",
                          schema_overrides={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})
    recon_d = pl.read_csv(input_path / f"{dataset_prefix}_{method}_{n_components_d}_reconstructed_test_demand{suffix_mono}.csv",
                          schema_overrides={"datetime": pl.Datetime, "price": pl.Float64, "volume": pl.Float64})
    
    # Filter actual data to test period
    actual_s = actual_s.filter(pl.col("datetime").is_between(test_start, test_end))
    actual_d = actual_d.filter(pl.col("datetime").is_between(test_start, test_end))
    
    # Join actual and reconstructed data
    joined_s = actual_s.join(recon_s, on=["datetime", "volume"], suffix="_recon")
    joined_d = actual_d.join(recon_d, on=["datetime", "volume"], suffix="_recon")
    
    results = []
    
    # Overall metrics
    for curve_type, joined_df in [("supply", joined_s), ("demand", joined_d)]:
        actual_prices = joined_df["price"].to_numpy()
        recon_prices = joined_df["price_recon"].to_numpy()
        metrics = calculate_metrics(actual_prices, recon_prices)
        results.append({
            "curve_type": curve_type,
            "aggregation": "overall",
            "group": "all",
            **metrics
        })
    
    # Per hour metrics
    for curve_type, joined_df in [("supply", joined_s), ("demand", joined_d)]:
        hourly_df = joined_df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        for hour in range(24):
            hour_data = hourly_df.filter(pl.col("hour") == hour)
            if len(hour_data) > 0:
                actual_prices = hour_data["price"].to_numpy()
                recon_prices = hour_data["price_recon"].to_numpy()
                metrics = calculate_metrics(actual_prices, recon_prices)
                results.append({
                    "curve_type": curve_type,
                    "aggregation": "hourly",
                    "group": f"hour_{hour:02d}",
                    **metrics
                })
    
    # Per weekday metrics
    for curve_type, joined_df in [("supply", joined_s), ("demand", joined_d)]:
        weekday_df = joined_df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"))
        for weekday in range(1, 8):  # 1=Monday, 7=Sunday
            weekday_data = weekday_df.filter(pl.col("weekday") == weekday)
            if len(weekday_data) > 0:
                actual_prices = weekday_data["price"].to_numpy()
                recon_prices = weekday_data["price_recon"].to_numpy()
                metrics = calculate_metrics(actual_prices, recon_prices)
                weekday_names = ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                results.append({
                    "curve_type": curve_type,
                    "aggregation": "weekday",
                    "group": weekday_names[weekday],
                    **metrics
                })
    
    # Save results
    output_path.mkdir(exist_ok=True)
    results_df = pl.DataFrame(results)
    output_file = output_path / f"{dataset_prefix}_{method}_d{n_components_d}_s{n_components_s}_reconstruction_evaluation{suffix_mono}.csv"
    results_df.write_csv(output_file)
    
    logger.success(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    app()