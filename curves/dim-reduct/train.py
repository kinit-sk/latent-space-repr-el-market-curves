from pathlib import Path
import pickle as pkl

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


def train_pca(
    df_train: np.array,
    df_val: np.array,
    variance_threshold: float = 0.9,
) -> tuple:
    """
    Perform dimensionality reduction on the input data using the principal component analysis (PCA).

    Args:
        df_train (np.array): The training data.
        df_val (np.array): The validation data.
        variance_threshold (float): Ratio of explained variance needed to be explained by PCA. Value in [0,1]. Defaults to 0.9.

    Returns:
        model: Model object of preselected method.
        n (int): Number of dimensions the PCA reduced the data to.
    """
    # check if variance threshold is in range
    if variance_threshold < 0 or variance_threshold > 1:
        logger.error("Variance threshold must be in range [0,1].")
        raise ValueError("Variance threshold must be in range [0,1].")

    logger.info(f"Training PCA until threshold of {variance_threshold} isn't reached...")

    # setup variables
    n = 0
    explained_variance = 0.0

    while explained_variance < variance_threshold:
        n += 1
        # pca object
        model = PCA(n_components=n)

        # fit the pca
        model.fit_transform(df_train)

        # calculated explained variance ratio
        explained_variance = model.explained_variance_ratio_.sum()

        logger.info(f"components = {n}, explained variance = {explained_variance:.4f}")

    # save best params to dict
    best_params = {
        "n_components": n,
    }

    logger.success(
        f"PCA training complete. Reduced dimension is {n}, explained variance is {explained_variance:.4f}."
    )

    return model, best_params


def train_kpca(
    df_train: np.array,
    df_val: np.array,
    training_means: np.array,
    training_stds: np.array,
    n_components: int = 2,
    kernels: list = ["poly", "rbf", "sigmoid", "cosine"],
    random_state: int = 42,
    n_jobs: int = -1,
) -> object:
    """
    Perform dimensionality reduction on the input data using the kernel principal component analysis (kPCA).

    Args:
        df_train (np.array): The training data.
        df_val (np.array): The validation data.
        training_means (np.array): Training data means (part of standardization).
        training_stds (np.array): Training data standard deviations (part of standardization)
        n_components (int, optional): Number of dimensions the kPCA reduces the data to. Defaults to 2.
        kernels (list, optional): Which kernels for kPCA to choose from. Defaults to ['poly', 'rbf', 'sigmoid', 'cosine'].
        random_state (int, optional): Random state of the kPCA. Defaults to 42.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

    Returns:
        model
    """
    logger.info(f"Training kPCA for {n_components} components and with kernels: {kernels}...")

    # Initialize variables to track best model
    best_mae = float("inf")
    best_kernel = None
    best_model = None

    for kernel in kernels:
        # define the model
        model = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            fit_inverse_transform=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        # fit the model
        model.fit(df_train)

        # reconstruct the validation curves from the model
        reconstructed = model.inverse_transform(model.transform(df_val))

        # calculate current metrics on reconstructed validation set
        current_mse = mean_squared_error(
            df_val * training_stds + training_means, reconstructed * training_stds + training_means
        )
        current_mae = mean_absolute_error(
            df_val * training_stds + training_means, reconstructed * training_stds + training_means
        )

        # Check if this is the best model so far
        if current_mae < best_mae:
            best_mae = current_mae
            best_mse = current_mse
            best_kernel = kernel
            best_model = model
            logger.success("New best model found!")

        logger.info(
            f"Current kPCA: kernel: {kernel}, number of components: {n_components}, MAE: {current_mae:.4f}, MSE: {current_mse:.4f}"
        )

    # save best params
    best_params = {"kernel": best_kernel}

    logger.success(
        f"Final best kPCA model: kernel: {best_kernel}, number of components: {n_components}, MAE: {best_mae:.4f}, MSE: {best_mse:.4f}"
    )
    logger.success("kPCA training complete.")

    return best_model, best_params


def train_umap(
    df_train: np.array,
    df_val: np.array,
    training_means: np.array,
    training_stds: np.array,
    n_components: int = 2,
    no_of_neighbors: list = [10, 11, 12, 13, 14, 15],
    metrics: list = ["euclidean", "manhattan", "chebyshev"],
    min_distances: list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple:
    """
    Perform dimensionality reduction on the input data using the UMAP method.

    Args:
        df_train (np.array): The training data.
        df_val (np.array): The validation data.
        training_means (np.array): Training data means (part of standardization).
        training_stds (np.array): Training data standard deviations (part of standardization)
        n_components (int, optional): Number of dimensions the UMAP reduces the data to. Defaults to 2.
        no_of_neighbors (list, optional): UMAP parameter - number of neighbours - to choose from. Defaults to [10,11,12,13,14,15].
        metrics (list, optional): UMAP parameter - distance metrics - to choose from. Defaults to ['euclidean', 'manhattan', 'chebyshev'].
        min_distances (list, optional): UMAP parameter - minimum distance to a point - to choose from. Defaults to [.001, .01, .1, .2, .3, .4, .5].
        random_state (int, optional): Random state of the UMAP. Defaults to 42.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

    Returns:
        tuple
    """
    logger.info(
        f"Training UMAP for {n_components} components and with no_of_neighbors: {no_of_neighbors}, metrics: {metrics}, min_distances: {min_distances}..."
    )

    # Initialize variables to track best model
    best_mae = float("inf")
    best_n_neighbors = None
    best_min_dist = None
    best_metric = None
    best_model = None

    for n_neighbors in no_of_neighbors:
        for min_dist in min_distances:
            for metric in metrics:
                # define the model
                model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )

                # fit the model
                model.fit(df_train)

                # reconstruct the validation curves from the model
                reconstructed = model.inverse_transform(model.transform(df_val))

                # calculate current metrics on reconstructed validation set
                current_mse = mean_squared_error(
                    df_val * training_stds + training_means,
                    reconstructed * training_stds + training_means,
                )
                current_mae = mean_absolute_error(
                    df_val * training_stds + training_means,
                    reconstructed * training_stds + training_means,
                )

                # Check if this is the best model so far
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_n_neighbors = n_neighbors
                    best_min_dist = min_dist
                    best_metric = metric
                    best_model = model
                    logger.success("New best model found!")

                logger.info(
                    f"Current UMAP: number of components: {n_components}, n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}, MAE: {current_mae:.4f}, MSE: {current_mse:.4f}"
                )

    # save best params to dict
    best_params = {
        "n_neighbors": best_n_neighbors,
        "min_dist": best_min_dist,
        "metric": best_metric,
    }

    logger.success(
        f"Final best UMAP model: number of components: {n_components}, n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}, MAE: {current_mae:.4f}, MSE: {current_mse:.4f}"
    )
    logger.success("UMAP training complete.")

    return best_model, best_params


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
    config_path: Path = PROJ_ROOT,
    output_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR,
) -> None:
    logger.info("Starting dimensionality reduction training...")
    # load config
    config = load_config(config_path / "config.yml")

    # variables from config
    dataset_prefix = config["preprocessing"]["dataset"]
    train_start = pl.datetime(
        config["preprocessing"]["train_start"][0],
        config["preprocessing"]["train_start"][1],
        config["preprocessing"]["train_start"][2],
    )
    train_end = pl.datetime(
        config["preprocessing"]["train_end"][0],
        config["preprocessing"]["train_end"][1],
        config["preprocessing"]["train_end"][2],
    )
    val_start = pl.datetime(
        config["preprocessing"]["val_start"][0],
        config["preprocessing"]["val_start"][1],
        config["preprocessing"]["val_start"][2],
    )
    val_end = pl.datetime(
        config["preprocessing"]["val_end"][0],
        config["preprocessing"]["val_end"][1],
        config["preprocessing"]["val_end"][2],
    )
    test_start = pl.datetime(
        config["preprocessing"]["test_start"][0],
        config["preprocessing"]["test_start"][1],
        config["preprocessing"]["test_start"][2],
    )
    test_end = pl.datetime(
        config["preprocessing"]["test_end"][0],
        config["preprocessing"]["test_end"][1],
        config["preprocessing"]["test_end"][2],
    )
    method = config["dim_reduct"]["method"]

    # path creation
    dataset_filename_supply = dataset_prefix + "_processed_supply_curves.csv"
    dataset_filename_demand = dataset_prefix + "_processed_demand_curves.csv"

    # load preprocessed sigmoid transformed data
    df_s = pl.read_csv(
        input_path / dataset_filename_supply,
        schema={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    df_d = pl.read_csv(
        input_path / dataset_filename_demand,
        schema={
            "datetime": pl.Datetime,
            "price": pl.Float64,
            "volume": pl.Float64,
        },
    )

    # prepare the data for the dimensionality reduction task
    df_train_s, df_val_s, df_test_s, means_s, stds_s, dates_train_s, dates_val_s, dates_test_s = (
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

    df_train_d, df_val_d, df_test_d, means_d, stds_d, dates_train_d, dates_val_d, dates_test_d = (
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

    # train the dimensionality reduction models
    if method == "pca":
        logger.info(f"Training {method} for supply on {dataset_prefix} dataset...")
        model_s, hyperparams_s = train_pca(
            df_train_s,
            df_val_s,
            config["dim_reduct"]["pca"]["variance_threshold"],
        )
        logger.success(f"Training {method} for supply on {dataset_prefix} dataset complete.")

        logger.info(f"Training {method} for demand on {dataset_prefix} dataset...")
        model_d, hyperparams_d = train_pca(
            df_train_d,
            df_val_d,
            config["dim_reduct"]["pca"]["variance_threshold"],
        )
        logger.success(f"Training {method} for demand on {dataset_prefix} dataset complete.")

        # save n components hyperparam
        n_components_s = hyperparams_s["n_components"]
        n_components_d = hyperparams_d["n_components"]

    elif method == "kpca":
        logger.info(f"Training {method} for supply on {dataset_prefix} dataset...")
        model_s, hyperparams_s = train_kpca(
            df_train_s,
            df_val_s,
            means_s,
            stds_s,
            config["dim_reduct"]["kpca"]["supply_n_components"],
            config["dim_reduct"]["kpca"]["kernels"],
            config["dim_reduct"]["kpca"]["random_state"],
            config["dim_reduct"]["kpca"]["n_jobs"],
        )
        logger.success(f"Training {method} for supply on {dataset_prefix} dataset complete.")

        logger.info(f"Training {method} for demand on {dataset_prefix} dataset...")
        model_d, hyperparams_d = train_kpca(
            df_train_d,
            df_val_d,
            means_d,
            stds_d,
            config["dim_reduct"]["kpca"]["demand_n_components"],
            config["dim_reduct"]["kpca"]["kernels"],
            config["dim_reduct"]["kpca"]["random_state"],
            config["dim_reduct"]["kpca"]["n_jobs"],
        )
        logger.success(f"Training {method} for demand on {dataset_prefix} dataset complete.")

        # save n components hyperparam
        n_components_s = config["dim_reduct"]["kpca"]["supply_n_components"]
        n_components_d = config["dim_reduct"]["kpca"]["demand_n_components"]

    elif method == "umap":
        logger.info(f"Training {method} for supply on {dataset_prefix} dataset...")
        model_s, hyperparams_s = train_umap(
            df_train_s,
            df_val_s,
            means_s,
            stds_s,
            config["dim_reduct"]["umap"]["supply_n_components"],
            config["dim_reduct"]["umap"]["no_of_neighbors"],
            config["dim_reduct"]["umap"]["metrics"],
            config["dim_reduct"]["umap"]["min_distances"],
            config["dim_reduct"]["umap"]["random_state"],
            config["dim_reduct"]["umap"]["n_jobs"],
        )
        logger.success(f"Training {method} for supply on {dataset_prefix} dataset complete.")

        logger.info(f"Training {method} for demand on {dataset_prefix} dataset...")
        model_d, hyperparams_d = train_umap(
            df_train_d,
            df_val_d,
            means_d,
            stds_d,
            config["dim_reduct"]["umap"]["demand_n_components"],
            config["dim_reduct"]["umap"]["no_of_neighbors"],
            config["dim_reduct"]["umap"]["metrics"],
            config["dim_reduct"]["umap"]["min_distances"],
            config["dim_reduct"]["umap"]["random_state"],
            config["dim_reduct"]["umap"]["n_jobs"],
        )
        logger.success(f"Training {method} for demand on {dataset_prefix} dataset complete.")

        # save n components hyperparam
        n_components_s = config["dim_reduct"]["umap"]["supply_n_components"]
        n_components_d = config["dim_reduct"]["umap"]["demand_n_components"]

    else:
        logger.error(f"Invalid method: {method}.")
        return

    # save model hyperparams to yaml
    logger.info(f"Saving the {method} hyperparams to yamls to {model_path}...")
    hp_filename_s = dataset_prefix + "_" + method + "_" + str(n_components_s) + "_supply.yml"
    hp_filename_d = dataset_prefix + "_" + method + "_" + str(n_components_d) + "_demand.yml"

    with open(model_path / hp_filename_s, "w") as f:
        yaml.dump(hyperparams_s, f, default_flow_style=False)

    with open(model_path / hp_filename_d, "w") as f:
        yaml.dump(hyperparams_d, f, default_flow_style=False)

    logger.success(f"Saving the {method} hyperparams to yamls completed.")

    # save the models to pickles
    logger.info(f"Saving the {method} models to pickles to {model_path}...")
    modelname_s = dataset_prefix + "_" + method + "_" + str(n_components_s) + "_supply.pkl"
    modelname_d = dataset_prefix + "_" + method + "_" + str(n_components_d) + "_demand.pkl"

    with open(model_path / modelname_s, "wb") as f:
        pkl.dump(model_s, f)

    with open(model_path / modelname_d, "wb") as f:
        pkl.dump(model_d, f)

    logger.info(f"Transforming the data to the latent space by {method}...")
    # transform the data to the latent space
    df_train_s = model_s.transform(df_train_s)
    df_val_s = model_s.transform(df_val_s)
    df_test_s = model_s.transform(df_test_s)
    df_train_d = model_d.transform(df_train_d)
    df_val_d = model_d.transform(df_val_d)
    df_test_d = model_d.transform(df_test_d)

    logger.success(f"Transforming the data to the latent space by {method} complete.")

    # create dataframes from numpys
    df_train_s = pl.DataFrame(
        df_train_s, schema=[method + "_" + str(i + 1) for i in range(df_train_s.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_train_s))

    df_val_s = pl.DataFrame(
        df_val_s, schema=[method + "_" + str(i + 1) for i in range(df_val_s.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_val_s))

    df_test_s = pl.DataFrame(
        df_test_s, schema=[method + "_" + str(i + 1) for i in range(df_test_s.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_test_s))

    df_train_d = pl.DataFrame(
        df_train_d, schema=[method + "_" + str(i + 1) for i in range(df_train_d.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_train_d))

    df_val_d = pl.DataFrame(
        df_val_d, schema=[method + "_" + str(i + 1) for i in range(df_val_d.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_val_d))

    df_test_d = pl.DataFrame(
        df_test_d, schema=[method + "_" + str(i + 1) for i in range(df_test_d.shape[1])]
    ).with_columns(pl.Series(name="datetime", values=dates_test_d))

    logger.success(f"Saving the {method} models to pickles completed.")

    # save the arrays to csvs
    logger.info(f"Saving the {method} arrays to csvs to {output_path}...")
    filename_train_s = (
        dataset_prefix + "_" + method + "_" + str(n_components_s) + "_train_supply.csv"
    )
    filename_val_s = dataset_prefix + "_" + method + "_" + str(n_components_s) + "_val_supply.csv"
    filename_test_s = (
        dataset_prefix + "_" + method + "_" + str(n_components_s) + "_test_supply.csv"
    )
    filename_train_d = (
        dataset_prefix + "_" + method + "_" + str(n_components_d) + "_train_demand.csv"
    )
    filename_val_d = dataset_prefix + "_" + method + "_" + str(n_components_d) + "_val_demand.csv"
    filename_test_d = (
        dataset_prefix + "_" + method + "_" + str(n_components_d) + "_test_demand.csv"
    )

    df_train_s.write_csv(output_path / filename_train_s)
    df_val_s.write_csv(output_path / filename_val_s)
    df_test_s.write_csv(output_path / filename_test_s)
    df_train_d.write_csv(output_path / filename_train_d)
    df_val_d.write_csv(output_path / filename_val_d)
    df_test_d.write_csv(output_path / filename_test_d)

    logger.success(f"Saving the {method} arrays to csvs completed.")
    logger.success(f"Training {method} for {dataset_prefix} dataset complete.")


if __name__ == "__main__":
    app()
