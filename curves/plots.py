from pathlib import Path

import datetime
from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import KernelPCA
from typing import Optional, Tuple
import pickle as pkl

from curves.config import FIGURES_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()


def visualize_supply_demand_reconstruction(
    supply_data: np.ndarray,
    demand_data: np.ndarray,
    supply_latent: np.ndarray,
    demand_latent: np.ndarray,
    supply_model: object = None,
    demand_model: object = None,
    supply_means: np.ndarray = None,
    supply_stds: np.ndarray = None,
    demand_means: np.ndarray = None,
    demand_stds: np.ndarray = None,
    sample_idx: int = None,
    figsize: Tuple[int, int] = (15, 6),
    dttm: Optional[str] = None,
    model_name: Optional[str] = None,
    x_values: Optional[np.ndarray] = None
) -> plt.Figure:
    """Visualize original vs reconstructed supply and demand curves together."""

    if all(x is not None for x in [supply_model, demand_model, supply_means, demand_means, supply_stds, demand_stds, sample_idx]):
        # Reconstruct curves
        supply_recon = supply_model.inverse_transform(supply_latent[sample_idx:sample_idx+1])
        demand_recon = demand_model.inverse_transform(demand_latent[sample_idx:sample_idx+1])
        
        # Denormalize
        supply_orig = supply_data[sample_idx] * supply_stds + supply_means
        supply_rec = supply_recon[0] * supply_stds + supply_means
        demand_orig = demand_data[sample_idx] * demand_stds + demand_means
        demand_rec = demand_recon[0] * demand_stds + demand_means
    else:
        supply_orig = supply_data
        demand_orig = demand_data
        supply_rec = supply_latent
        demand_rec = demand_latent
    
    # Create title with optional datetime and model name
    title = "Original vs Reconstructed"
    if type(dttm) == np.datetime64:
        formatted_datetime = dttm.astype(datetime.datetime).strftime('%Y-%m-%d %H:%M')
        title += f" - {formatted_datetime}"
    elif type(dttm) == datetime.datetime:
        title += f" - {dttm.strftime('%Y-%m-%d %H:%M')}"

    if model_name:
        title += f" ({model_name})"
    
    # Use custom x values if provided
    x_axis = x_values if x_values is not None else np.arange(len(supply_orig))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Comparison
    ax.plot(x_axis, supply_orig, 'b-', linewidth=2, label='Supply Original')
    ax.plot(x_axis, supply_rec, 'b--', linewidth=2, label='Supply Reconstructed')
    ax.plot(x_axis, demand_orig, 'g-', linewidth=2, label='Demand Original')
    ax.plot(x_axis, demand_rec, 'g--', linewidth=2, label='Demand Reconstructed')
    ax.set_title(title)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    return fig


def visualize_supply_demand_reconstruction_with_monotonic(
    supply_data: np.ndarray,
    demand_data: np.ndarray,
    supply_latent: np.ndarray,
    demand_latent: np.ndarray,
    supply_monotonic: np.ndarray,
    demand_monotonic: np.ndarray,
    supply_model: object = None,
    demand_model: object = None,
    supply_means: np.ndarray = None,
    supply_stds: np.ndarray = None,
    demand_means: np.ndarray = None,
    demand_stds: np.ndarray = None,
    sample_idx: int = None,
    figsize: Tuple[int, int] = (15, 6),
    dttm: Optional[str] = None,
    model_name: Optional[str] = None,
    x_values: Optional[np.ndarray] = None
) -> plt.Figure:
    """Visualize original vs reconstructed vs monotonic supply and demand curves together."""

    if all(x is not None for x in [supply_model, demand_model, supply_means, demand_means, supply_stds, demand_stds, sample_idx]):
        # Reconstruct curves
        supply_recon = supply_model.inverse_transform(supply_latent[sample_idx:sample_idx+1])
        demand_recon = demand_model.inverse_transform(demand_latent[sample_idx:sample_idx+1])
        
        # Denormalize
        supply_orig = supply_data[sample_idx] * supply_stds + supply_means
        supply_rec = supply_recon[0] * supply_stds + supply_means
        demand_orig = demand_data[sample_idx] * demand_stds + demand_means
        demand_rec = demand_recon[0] * demand_stds + demand_means
        supply_mono = supply_monotonic[sample_idx] * supply_stds + supply_means
        demand_mono = demand_monotonic[sample_idx] * demand_stds + demand_means
    else:
        supply_orig = supply_data
        demand_orig = demand_data
        supply_rec = supply_latent
        demand_rec = demand_latent
        supply_mono = supply_monotonic
        demand_mono = demand_monotonic
    
    # Create title with optional datetime and model name
    title = "Original vs Recon (+IR)"
    if type(dttm) == np.datetime64:
        formatted_datetime = dttm.astype(datetime.datetime).strftime('%Y-%m-%d %H:%M')
        title += f" - {formatted_datetime}"
    elif type(dttm) == datetime.datetime:
        title += f" - {dttm.strftime('%Y-%m-%d %H:%M')}"

    if model_name:
        title += f" ({model_name})"
    
    # Use custom x values if provided
    x_axis = x_values if x_values is not None else np.arange(len(supply_orig))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Comparison
    ax.plot(x_axis, supply_orig, 'b-', linewidth=2, label='Supply Original')
    ax.plot(x_axis, supply_rec, 'b--', linewidth=2, label='Supply Recon')
    ax.plot(x_axis, supply_mono, 'b:', linewidth=2, label='Supply Recon+IR')
    ax.plot(x_axis, demand_orig, 'g-', linewidth=2, label='Demand Original')
    ax.plot(x_axis, demand_rec, 'g--', linewidth=2, label='Demand Recon')
    ax.plot(x_axis, demand_mono, 'g:', linewidth=2, label='Demand Recon+IR')
    ax.set_title(title)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    return fig


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
