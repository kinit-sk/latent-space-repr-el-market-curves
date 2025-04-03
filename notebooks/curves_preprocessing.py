# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: curves_env
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# directories
PROJ_ROOT = Path("/data/repo/latent-space-repr-el-market-curves")
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# %%
df = pl.read_csv(
    RAW_DATA_DIR / "mibel_supply_curves.csv",
    schema = {
        "datetime": pl.Datetime,
        "Price": pl.Float64,
        "Volume": pl.Float64,
        "MCV": pl.Float64,
        "MCP": pl.Float64,
    }
)

df = df.select(pl.all().name.to_lowercase())

# %%
df.head()

# %%
[df.shape, df.dtypes]

# %%
# look up NaNs
df.null_count()

# %%
# Create figure and axis objects with a single subplot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Add second y-axis sharing same x-axis
ax2 = ax1.twinx()

# Plot histogram on first y-axis
sns.histplot(data=df, x='mcp', ax=ax1, color='skyblue', alpha=0.6)

# Plot ECDF on second y-axis
sns.ecdfplot(data=df, x='mcp', ax=ax2, color='blue')

# Customize labels and title
ax1.set_xlabel('MCP')
ax1.set_ylabel('Count')
ax2.set_ylabel('Cumulative Probability')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, ['Histogram', 'ECDF'])

plt.tight_layout()
plt.show()


# %%
# Create figure and axis objects with a single subplot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Add second y-axis sharing same x-axis
ax2 = ax1.twinx()

# Plot histogram on first y-axis
sns.histplot(data=df, x='mcv', ax=ax1, color='skyblue', alpha=0.6)

# Plot ECDF on second y-axis
sns.ecdfplot(data=df, x='mcv', ax=ax2, color='blue')

# Customize labels and title
ax1.set_xlabel('MCV')
ax1.set_ylabel('Count')
ax2.set_ylabel('Cumulative Probability')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, ['Histogram', 'ECDF'])

plt.tight_layout()
plt.show()


# %%
# calculate quantiles
mcv_quantiles = df.unique('mcv').select([
    pl.col('mcv').quantile(q).alias(f'{q}') 
    for q in [0.005, 0.05, 0.1, 0.5, 0.9, 0.95, 0.995]
])
mcv_quantiles

# %%
# calculate quantiles
mcp_quantiles = df.unique('mcp').select([
    pl.col('mcp').quantile(q).alias(f'{q}') 
    for q in [0.005, 0.05, 0.1, 0.5, 0.9, 0.95, 0.995]
])
mcp_quantiles

# %%
# crop the prices to 99% CI of MCP
df = df.with_columns(
    pl.col('price').clip(
        lower_bound=mcp_quantiles[str(0.005)],
        upper_bound=mcp_quantiles[str(0.995)],
    )
)

# %%
df.head()

# %%
# Calculate range parameters
step = 1  # step - \delta_p value
price_min = int(np.ceil(df['price'].min()))  # first value of range
price_max = int(np.ceil(df['price'].max())) + step  # last value of range
#datetimes = df['datetime'].unique().sort()  # datetimes

# get only one month of datetimes
# If you need to filter dates before 2018-02-01, use:
datetimes = df.filter(
    pl.col("datetime") < pl.datetime(2018, 2, 1)
)['datetime'].unique().sort()

# list for less memory intensive preprocessing
all_obs = []

# iterate over datetimes
for datetime in tqdm(datetimes):
    # filter out one hour data with Price,Volume and datetime columns only
    onehour_data = df.filter(
        pl.col('datetime') == datetime
    ).select(['datetime', 'price', 'volume'])
    
    for bucket in range(price_min, price_max, step):
        # get the value with largest volume as the new_observation for the given bucket
        new_obs = (onehour_data
                  .filter(pl.col('price') <= (bucket + 0.5*step))
                  .sort('volume')
                  .tail(1))
        
        if len(new_obs) == 1:  
            # modify the Price value and convert to dictionary
            new_obs = new_obs.with_columns(pl.lit(bucket).alias('price'))
            all_obs.append(new_obs.to_dict(as_series=False))

# create dataframe from the list
df_2 = pl.DataFrame(all_obs)

# Convert types to match original dataframe
df_2 = df_2.explode(pl.all())


# %%
df_2.head()

# %%
df_2.tail()

# %%
# enhanced iterator accounting for extrema - not in the paper, but it provides better PCA results
quantiles = [mcv_quantiles['0.005'].item(), mcv_quantiles['0.995'].item()] # MCV clipping of volumes
step = 100 # one step of iterator - 100 MWh
volume_min = int((np.ceil(quantiles[0]/step) + 1) * step) # min volume in range
volume_max = int((np.ceil(quantiles[1]/step) + 1) * step) # max volume in range
#volume_min = int(1 * step) # min volume in range
#volume_max = int((np.ceil(df_2['volume'].max()/step) + 1) * step) # max volume in range
iterator = range(volume_min, volume_max, step) # iterator
iterator

# %%
# datetimes to iterate over
datetimes = df_2['datetime'].unique().sort()

# list for less memory intensive preprocessing
all_obs = []

# iterate over datetimes
for datetime in tqdm(datetimes):
    
    # get only one hour of data and sort them by volumes
    onehour_data = (df_2.filter(pl.col('datetime') == datetime)
                    .sort('volume'))
    
    # in certain buckets, no bids are present, therefore we need to put NaNs into these buckets
    new_obs_min = onehour_data.head(1)
    new_obs_min = new_obs_min.with_columns([
        pl.lit(datetime).alias('datetime'),
        pl.lit(None).cast(pl.Int64).alias('price')
    ])
    
    # iterate over volume buckets
    for bucket in iterator:
        
        # get the largest volume out of these buckets
        new_obs = (onehour_data
                  .filter(pl.col('volume') <= bucket)
                  .tail(1))
        
        # if there is a price in the bucket, append it to all_obs with Volume = bucket
        if len(new_obs) == 1:
            new_obs = new_obs.with_columns(pl.lit(bucket).alias('volume'))
            all_obs.append(new_obs.row(0))
            
        # if there is not an observation put NaN into that bucket
        else:
            temp_obs = new_obs_min.with_columns(pl.lit(bucket).alias('volume'))
            all_obs.append(temp_obs.row(0))

# create dataframe from the list       
df_3 = pl.DataFrame(all_obs, schema=df_2.schema, orient='row')



# %%
df_3

# %%
# fill the NaNs with min of Price
df_3 = df_3.with_columns(
    pl.col('price').fill_null(pl.col('price').min())
)
df_3.head()

# %%
# calculate quantiles
quantiles = df_3.select([
    pl.col('price').quantile(q).alias(f'{q}') 
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]
])
quantiles = quantiles.with_columns([
    (pl.lit(0.5)*(pl.col('0.9') + pl.col('0.1'))).alias('0.5')
])
quantiles


# %%
def sigmoid_transformation(input_data, qnt, r_0=10):
    """Sigmoid transformation step from Guo et al. paper to transform the prices by custom sigmoid.

    Args:
        input_data (polars.DataFrame): Data to transform. Can be one day or whole dataframe.
        qnt (dict or polars.DataFrame): Modified quantiles of input_data.
        r_0 (int, optional): Parameter r_0 used in sigmoid transformation to keep the values in [0,1]. Defaults to 10.

    Returns:
        output_data (polars.DataFrame): Sigmoid transformed input_data.
    """
    # Calculate r_min and r_max from paper
    r_min = r_0/(qnt['0.5'].item() - qnt['0.1'].item())
    r_max = r_0/(qnt['0.9'].item() - qnt['0.5'].item())
    
    # Create the sigmoid transformation using when-then-otherwise
    output_data = input_data.with_columns(
        pl.when(pl.col('price') < qnt['0.5'].item())
        .then(1 / (1 + np.exp(-r_min * (pl.col('price') - qnt['0.5'].item()))))
        .otherwise(1 / (1 + np.exp(-r_max * (pl.col('price') - qnt['0.5'].item()))))
        .alias('price')
    )
    
    return output_data



# %%
# S5 transformation on whole dataset
df_4 = sigmoid_transformation(df_3, quantiles)

# %%
df_4

# %%
# lets look at some subsets
subset1 = df_3.filter(pl.col("datetime") == pl.datetime(2018, 1, 1, 14, 0, 0))
subset2 = df_3.filter(pl.col("datetime") == pl.datetime(2018, 1, 5, 14, 0, 0))
subset3 = df_3.filter(pl.col("datetime") == pl.datetime(2018, 1, 10, 14, 0, 0))
subset4 = df_3.filter(pl.col("datetime") == pl.datetime(2018, 1, 20, 14, 0, 0))
subset5 = df_3.filter(pl.col("datetime") == pl.datetime(2018, 1, 30, 14, 0, 0))

transformed1 = sigmoid_transformation(subset1, quantiles).to_pandas()
transformed2 = sigmoid_transformation(subset2, quantiles).to_pandas()
transformed3 = sigmoid_transformation(subset3, quantiles).to_pandas()
transformed4 = sigmoid_transformation(subset4, quantiles).to_pandas()
transformed5 = sigmoid_transformation(subset5, quantiles).to_pandas()

subset1 = subset1.to_pandas()
subset2 = subset2.to_pandas()
subset3 = subset3.to_pandas()
subset4 = subset4.to_pandas()
subset5 = subset5.to_pandas()

# %%
# Create plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=subset1, x='volume', y='price', label=subset1.iloc[0,0])
sns.lineplot(data=subset2, x='volume', y='price', label=subset2.iloc[0,0])
sns.lineplot(data=subset3, x='volume', y='price', label=subset3.iloc[0,0])
sns.lineplot(data=subset4, x='volume', y='price', label=subset4.iloc[0,0])
sns.lineplot(data=subset5, x='volume', y='price', label=subset5.iloc[0,0])

plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.xlabel('Volume')
plt.ylabel('Price')
plt.title('Price vs Volume Bucket Index for Different Dates')
plt.tight_layout()
plt.show()

# %%
# Create plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=transformed1, x='volume', y='price', label=subset1.iloc[0,0])
sns.lineplot(data=transformed2, x='volume', y='price', label=subset2.iloc[0,0])
sns.lineplot(data=transformed3, x='volume', y='price', label=subset3.iloc[0,0])
sns.lineplot(data=transformed4, x='volume', y='price', label=subset4.iloc[0,0])
sns.lineplot(data=transformed5, x='volume', y='price', label=subset5.iloc[0,0])

plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.xlabel('Volume')
plt.ylabel('Price')
plt.title('Price vs Volume Bucket Index for Different Dates')
plt.tight_layout()
plt.show()
