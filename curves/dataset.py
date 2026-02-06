from pathlib import Path

from loguru import logger
import numpy as np
import polars as pl
from tqdm import tqdm
import typer

from curves.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR, load_config

app = typer.Typer()


def load_raw_data(
    path: Path,
) -> pl.DataFrame:
    """
    Load raw market curve data from a CSV file.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """

    logger.info(f"Loading dataset from {path}...")

    df = pl.read_csv(
        path,
        schema={
            "datetime": pl.Datetime,
            "Price": pl.Float64,
            "Volume": pl.Float64,
            "MCV": pl.Float64,
            "MCP": pl.Float64,
        },
    )

    df = df.select(pl.all().name.to_lowercase())

    logger.success(f"Loaded {len(df)} rows from {path}")

    return df


def winsorize(
    df: pl.DataFrame,
    lower: float,
    upper: float,
    col: str,
) -> pl.DataFrame:
    """
    Winsorize the values in the specified column of the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        lower (float): The lower quantile value for winsorization.
        upper (float): The upper quantile value for winsorization.
        col (str): The column name to winsorize.

    Returns:
        pl.DataFrame: The winsorized DataFrame.
    """

    # crop the prices to predefined quantile values
    new_df = df.with_columns(
        pl.col(col).clip(
            lower_bound=lower,
            upper_bound=upper,
        )
    )
    return new_df


def merging_prices(
    df: pl.DataFrame,
    iterator: range,
    step: int,
    supply: bool = True,
) -> pl.DataFrame:
    """
    Merge the prices in the DataFrame based on the given iterator and step.

    Args:
        df (pl.DataFrame): The input DataFrame.
        iterator (range): The iterator for merging prices.
        step (int): The step size for merging prices.
        supply (bool): Boolean for indicating if supply or demand curves are being processed. Defaults to True.

    Returns:
        pl.DataFrame: The DataFrame with merged prices.
    """
    # list for less memory intensive preprocessing
    all_obs = []

    datetimes = df["datetime"].unique().sort()  # datetimes
    # TODO in live remove this:
    # datetimes = df.filter(
    #     pl.col("datetime") < pl.datetime(2018, 2, 1)
    # )['datetime'].unique().sort()

    # iterate over datetimes
    for datetime in tqdm(datetimes, desc="Step II"):
        # filter out one hour data with Price,Volume and datetime columns only
        onehour_data = df.filter(pl.col("datetime") == datetime).select(
            ["datetime", "price", "volume"]
        )

        for bucket in iterator:
            # get the value with largest volume as the new_observation for the given bucket
            if supply:
                new_obs = (
                    onehour_data.filter(pl.col("price") <= (bucket + 0.5 * step))
                    .sort("volume")
                    .tail(1)
                )
            else:
                new_obs = (
                    onehour_data.filter(pl.col("price") >= (bucket + 0.5 * step))
                    .sort("volume")
                    .tail(1)
                )

            if len(new_obs) == 1:
                # modify the Price value and convert to dictionary
                new_obs = new_obs.with_columns(pl.lit(bucket).alias("price"))
                all_obs.append(new_obs.to_dict(as_series=False))

    # create dataframe from the list
    new_df = pl.DataFrame(all_obs)

    # Convert types to match original dataframe
    new_df = new_df.explode(pl.all())

    return new_df


def sampling_uniform_volume(
    df: pl.DataFrame,
    iterator: range,
) -> pl.DataFrame:
    """
    Sample the prices uniformly across volume dimension based on iterator.

    Args:
        df (pl.DataFrame): The input DataFrame.
        iterator (range): The iterator for uniform volume sampling.

    Returns:
        pl.DataFrame: The DataFrame with prices sampled uniformly across volume.
    """
    # list for less memory intensive preprocessing
    all_obs = []

    # datetimes to iterate over
    datetimes = df["datetime"].unique().sort()

    # iterate over datetimes
    for datetime in tqdm(datetimes, desc="Step III"):
        # get only one hour of data and sort them by volumes
        onehour_data = df.filter(pl.col("datetime") == datetime).sort("volume")

        # in certain buckets, no bids are present, therefore we need to put NaNs into these buckets
        new_obs_min = onehour_data.head(1)
        new_obs_min = new_obs_min.with_columns(
            [pl.lit(datetime).alias("datetime"), pl.lit(None).cast(pl.Int64).alias("price")]
        )

        # iterate over volume buckets
        for bucket in iterator:
            # get the largest volume out of these buckets
            new_obs = onehour_data.filter(pl.col("volume") <= bucket).tail(1)

            # if there is a price in the bucket, append it to all_obs with Volume = bucket
            if len(new_obs) == 1:
                new_obs = new_obs.with_columns(pl.lit(bucket).alias("volume"))
                all_obs.append(new_obs.row(0))

            # if there is not an observation put NaN into that bucket
            else:
                temp_obs = new_obs_min.with_columns(pl.lit(bucket).alias("volume"))
                all_obs.append(temp_obs.row(0))

    # create dataframe from the list
    new_df = pl.DataFrame(all_obs, schema=df.schema, orient="row")

    return new_df


def price_transformation(df, qnt, r_0=10):
    """
    Sigmoid transformation step from Guo et al. paper to transform the prices by custom sigmoid.

    Args:
        df (pl.DataFrame): Data to transform. Can be one day or whole dataframe.
        qnt (dict or pl.DataFrame): Modified quantiles of input_data.
        r_0 (int, optional): Parameter r_0 used in sigmoid transformation to keep the values in [0,1]. Defaults to 10.

    Returns:
        new_df (pl.DataFrame): Sigmoid transformed input_data.
    """
    # Calculate r_min and r_max from paper
    r_min = r_0 / (qnt["0.5"].item() - qnt["0.1"].item())
    r_max = r_0 / (qnt["0.9"].item() - qnt["0.5"].item())

    # Create the sigmoid transformation using when-then-otherwise
    new_df = df.with_columns(
        pl.when(pl.col("price") < qnt["0.5"].item())
        .then(1 / (1 + np.exp(-r_min * (pl.col("price") - qnt["0.5"].item()))))
        .otherwise(1 / (1 + np.exp(-r_max * (pl.col("price") - qnt["0.5"].item()))))
        .alias("price")
    )

    return new_df


def preprocess_dataset(
    df: pl.DataFrame,
    supply: bool,
    price_winsor_lower: float,
    price_winsor_upper: float,
    step2_stepsize: int,
    volume_winsor_lower: float,
    volume_winsor_upper: float,
    step3_stepsize: int,
    step4_r_0: int,
) -> pl.DataFrame:
    """
    Preprocess the dataset by applying various transformations.

    Args:
        df (pl.DataFrame): The input DataFrame.
        supply (bool): Boolean for indicating if supply or demand curves are being processed.
        price_winsor_lower (float): Lower quantile value for price winsorization in [0,1].
        price_winsor_upper (float): Upper quantile value for price winsorization in [0,1].
        step2_stepsize (int): Step size for merging prices.
        volume_winsor_lower (float): Lower quantile value for volume winsorization in [0,1].
        volume_winsor_upper (float): Upper quantile value for volume winsorization in [0,1].
        step3_stepsize (int): Step size for uniform volume sampling .
        step4_r_0 (int): Parameter r_0 used in price transformation.

    Returns:
        pl.DataFrame: The preprocessed DataFrame.
    """
    logger.info("Preprocessing dataset...")

    # calculate quantiles of market clearing volumes/prices
    mcv_quantile = df.unique("mcv").select(
        [
            pl.col("mcv").quantile(q).alias(f"{q}")
            for q in [volume_winsor_lower, volume_winsor_upper]
        ]
    )

    mcp_quantile = df.unique("mcp").select(
        [pl.col("mcp").quantile(q).alias(f"{q}") for q in [price_winsor_lower, price_winsor_upper]]
    )

    ## paper Step I
    logger.info(f"Step I: Winsorizing prices to {mcp_quantile.row(0)}...")
    # crop the prices to predefined quantile of MCP
    df = winsorize(
        df,
        mcp_quantile[f"{price_winsor_lower}"],
        mcp_quantile[f"{price_winsor_upper}"],
        "price",
    )

    logger.success(f"Step I: Winsorizing prices to {mcp_quantile.row(0)} complete.")

    ## paper Step II
    # Calculate range parameters
    price_min = int(np.ceil(df["price"].min()))  # first value of range
    price_max = int(np.ceil(df["price"].max())) + step2_stepsize  # last value of range

    if supply:
        step2_iterator = range(price_min, price_max, step2_stepsize)  # iterator
        logger.info(
            f"Step II: Merging prices to uniform price step from {price_min} to {price_max} by {step2_stepsize}..."
        )
    else:
        step2_iterator = range(price_max, price_min, -step2_stepsize)  # iterator
        logger.info(
            f"Step II: Merging prices to uniform price step from {price_max} to {price_min} by {step2_stepsize}..."
        )

    df = merging_prices(
        df,
        step2_iterator,
        step2_stepsize,
        supply=supply,
    )
    logger.success(f"Step II: Merging prices to uniform price step complete. nrows={len(df)}")

    # paper Step III
    # Calculate range parameters
    volume_min = int(
        step3_stepsize
        * (np.ceil((mcv_quantile[f"{volume_winsor_lower}"].item()) / step3_stepsize) + 1)
    )  # first value of range
    volume_max = int(
        step3_stepsize
        * (np.ceil((mcv_quantile[f"{volume_winsor_upper}"].item()) / step3_stepsize) + 1)
    )  # last value of range
    step3_iterator = range(volume_min, volume_max, step3_stepsize)  # iterator

    logger.info(
        f"Step III: Sampling prices to uniform volume step from {volume_min} to {volume_max} by {step3_stepsize}..."
    )
    df = sampling_uniform_volume(
        df,
        step3_iterator,
    )
    logger.success(f"Step III: Sampling prices to uniform volume step complete. nrows={len(df)}")

    if supply:
        # fill the NaNs with min of Price
        df = df.with_columns(pl.col("price").fill_null(pl.col("price").min()))
    else:
        # fill the NaNs with max of Price
        df = df.with_columns(pl.col("price").fill_null(pl.col("price").max()))

    # calculate price quantiles
    quantiles = df.select([pl.col("price").quantile(q).alias(f"{q}") for q in [0.1, 0.5, 0.9]])
    quantiles = quantiles.with_columns(
        [(pl.lit(0.5) * (pl.col("0.9") + pl.col("0.1"))).alias("0.5")]
    )

    logger.info(f"Calculated quantiles of price (10%, 50%, 90%): {quantiles.row(0)}")

    return df, quantiles


def preprocess_dataset_for_dim_reduct(
    df: pl.DataFrame,
    train_start: pl.datetime,
    train_end: pl.datetime,
    val_start: pl.datetime,
    val_end: pl.datetime,
    test_start: pl.datetime,
    test_end: pl.datetime,
) -> tuple:
    """
    Preprocess the dataset for dimensionality reduction.

    Args:
        df (pl.DataFrame): The input DataFrame.
        train_end (pl.Datetime): The end date for the training set.
        val_start (pl.Datetime): The start date for the validation set.
        val_end (pl.Datetime): The end date for the validation set.
        test_start (pl.Datetime): The start date for the test set.
        test_end (pl.Datetime): The end date for the test set.

    Returns:
        tuple
    """
    logger.info("Preprocessing dataset for dimensionality reduction...")

    # split the data into train, val, test
    df_train = df.filter((pl.col("datetime") < train_end) & (pl.col("datetime") >= train_start))
    df_val = df.filter((pl.col("datetime") < val_end) & (pl.col("datetime") >= val_start))
    df_test = df.filter((pl.col("datetime") < test_end) & (pl.col("datetime") >= test_start))

    # reshape dataframe such that rows = datetime observations and cols = N volume buckets + convert to numpy array
    unique_dates = df_train.select(pl.col("datetime").unique()).to_numpy().flatten()
    N = int(len(df_train) / len(unique_dates))
    df_train_np = np.reshape(df_train.select(pl.col("price")).to_numpy(), (len(unique_dates), N))

    # column means and stdevs
    mean_cols = df_train_np.mean(axis=0)
    sd_cols = df_train_np.std(axis=0)
    [mean_cols.shape, sd_cols.shape]

    # standardize dataframe column wise (for every bucket)
    df_train_np = (df_train_np - mean_cols) / sd_cols

    # transform the test and val set as well
    unique_dates_val = df_val.select(pl.col("datetime").unique()).to_numpy().flatten()
    df_val_np = np.reshape(df_val.select(pl.col("price")).to_numpy(), (len(unique_dates_val), N))
    df_val_np = (df_val_np - mean_cols) / sd_cols

    unique_dates_test = df_test.select(pl.col("datetime").unique()).to_numpy().flatten()
    df_test_np = np.reshape(
        df_test.select(pl.col("price")).to_numpy(), (len(unique_dates_test), N)
    )
    df_test_np = (df_test_np - mean_cols) / sd_cols

    return (
        df_train_np,
        df_val_np,
        df_test_np,
        mean_cols,
        sd_cols,
        unique_dates,
        unique_dates_val,
        unique_dates_test,
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    config_path: Path = PROJ_ROOT,
    output_path: Path = PROCESSED_DATA_DIR,
):
    # load config
    config = load_config(config_path / "config.yml")
    dataset_prefix = config["preprocessing"]["dataset"]

    logger.info(f"Processing {dataset_prefix} dataset...")

    # path creation
    dataset_filename_supply = dataset_prefix + "_supply_curves.csv"
    dataset_filename_demand = dataset_prefix + "_demand_curves.csv"

    output_filename_supply = dataset_prefix + "_processed_supply_curves.csv"
    output_filename_demand = dataset_prefix + "_processed_demand_curves.csv"

    quantiles_filename_supply = dataset_prefix + "_quantiles_supply.csv"
    quantiles_filename_demand = dataset_prefix + "_quantiles_demand.csv"

    ## supply curves
    # load dataset
    df_s = load_raw_data(input_path / dataset_filename_supply)

    # preprocess the dataset
    logger.info(f"Processing {dataset_prefix} supply dataset...")
    df_s, quantiles_s = preprocess_dataset(
        df_s,
        supply=True,
        price_winsor_lower=config["preprocessing"]["price_winsor_lower"],
        price_winsor_upper=config["preprocessing"]["price_winsor_upper"],
        step2_stepsize=config["preprocessing"]["step2_stepsize"],
        volume_winsor_lower=config["preprocessing"]["volume_winsor_lower"],
        volume_winsor_upper=config["preprocessing"]["volume_winsor_upper"],
        step3_stepsize=config["preprocessing"]["step3_stepsize"],
        step4_r_0=config["preprocessing"]["step4_r_0"],
    )
    logger.info(f"Processed {dataset_prefix} supply dataset.")

    # save the dataset
    df_s.write_csv(output_path / output_filename_supply)
    logger.success(f"Saved {output_filename_supply} to {output_path}.")
    quantiles_s.write_csv(output_path / quantiles_filename_supply)
    logger.success(f"Saved {quantiles_filename_supply} to {output_path}.")

    ## demand curves
    # load dataset
    df_d = load_raw_data(input_path / dataset_filename_demand)

    # preprocess the dataset
    logger.info(f"Processing {dataset_prefix} demand dataset...")
    df_d, quantiles_d = preprocess_dataset(
        df_d,
        supply=False,
        price_winsor_lower=config["preprocessing"]["price_winsor_lower"],
        price_winsor_upper=config["preprocessing"]["price_winsor_upper"],
        step2_stepsize=config["preprocessing"]["step2_stepsize"],
        volume_winsor_lower=config["preprocessing"]["volume_winsor_lower"],
        volume_winsor_upper=config["preprocessing"]["volume_winsor_upper"],
        step3_stepsize=config["preprocessing"]["step3_stepsize"],
        step4_r_0=config["preprocessing"]["step4_r_0"],
    )
    logger.info(f"Processed {dataset_prefix} demand dataset.")

    # save the dataset
    df_d.write_csv(output_path / output_filename_demand)
    logger.success(f"Saved {output_filename_demand} to {output_path}.")
    quantiles_d.write_csv(output_path / quantiles_filename_demand)
    logger.success(f"Saved {quantiles_filename_demand} to {output_path}.")

    logger.success(f"Processing {dataset_prefix} dataset complete.")


if __name__ == "__main__":
    app()
