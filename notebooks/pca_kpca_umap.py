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
import polars as pl
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import umap
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%
# directories
PROJ_ROOT = Path("/data/repo/latent-space-repr-el-market-curves")
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# %%
# load preprocessed sigmoid transformed data
df = pl.read_csv(
    PROCESSED_DATA_DIR / 'mibel_processed_demand_curves.csv',
    schema = {
        "datetime": pl.Datetime,
        "price": pl.Float64,
        "volume": pl.Float64,
    },
)
df.glimpse()

# %%
# split it into train and test
df_test = df.filter(pl.col("datetime") >= pl.datetime(2020, 1, 1))
df_val = df.filter((pl.col("datetime") < pl.datetime(2020, 1, 1)) & (pl.col("datetime") >= pl.datetime(2019, 10, 1)))
df_train = df.filter(pl.col("datetime") < pl.datetime(2019, 10, 1))
df_train.tail()

# %%
# reshape dataframe such that rows = datetime observations and cols = n1 volume buckets + convert to numpy array
unique_dates = df_train.select(pl.col("datetime").unique()).to_numpy()
n1 = int(len(df_train)/len(unique_dates))
df_train_np = np.reshape(df_train.select(pl.col("price")).to_numpy(), (len(unique_dates), n1))

# column means and stdevs
mean_cols = df_train_np.mean(axis=0)
sd_cols = df_train_np.std(axis=0)
[mean_cols.shape, sd_cols.shape]

# standardize dataframe column wise (for every bucket)
df_train_np = (df_train_np - mean_cols)/sd_cols

# %%
# transform the test and val set as well
unique_dates_val = df_val.select(pl.col("datetime").unique()).to_numpy()
df_val_np = np.reshape(df_val.select(pl.col("price")).to_numpy(), (len(unique_dates_val), n1))
df_val_np = (df_val_np - mean_cols)/sd_cols

unique_dates_test = df_test.select(pl.col("datetime").unique()).to_numpy()
df_test_np = np.reshape(df_test.select(pl.col("price")).to_numpy(), (len(unique_dates_test), n1))
df_test_np = (df_test_np - mean_cols)/sd_cols

# %%
df_train_np

# %%
# visualize covariance matrix
plt.imshow(np.matmul(np.transpose(df_train_np), df_train_np))
plt.colorbar()
plt.show()

# %%
# visualize covariance matrix
plt.imshow(np.matmul(np.transpose(df_val_np), df_val_np))
plt.colorbar()
plt.show()

# %%
# visualize covariance matrix
plt.imshow(np.matmul(np.transpose(df_test_np), df_test_np))
plt.colorbar()
plt.show()

# %%
# PCA for 1 to 50 components
max_components = 50
pca_res = {}
pca_mse = []
pca_mae = []
for n in range(1, max_components+1):
    # pca object
    pca = PCA(n_components = n)
    # fit the pca
    pca.fit_transform(df_train_np)
    # append explained variance to res
    pca_res[n] = pca.explained_variance_ratio_.sum()*100
    # calculate reconstruction MSE and MAE on test set
    pca_mse.append(mean_squared_error(df_val_np * sd_cols + mean_cols, pca.inverse_transform(pca.transform(df_val_np)) * sd_cols + mean_cols))
    pca_mae.append(mean_absolute_error(df_val_np * sd_cols + mean_cols, pca.inverse_transform(pca.transform(df_val_np)) * sd_cols + mean_cols))
    
    print(f'components = {n}, explained variance = {pca.explained_variance_ratio_.sum():.4f}, test reconstruction MSE = {pca_mse[n-1]:.4f}, test reconstruction MAE = {pca_mae[n-1]:.4f}')

# %%
# sorted by key, return a list of tuples
lists = sorted(pca_res.items()) 

# unpack a list of pairs into two tuples
x, y = zip(*lists) 

# plot components and explained variance ratio plot
plt.plot(x, y)
plt.axhline(y=80, color='r', linestyle='--')
plt.show()

# %%
### simple visualizations to compare how well pca carries over information in lower dimensions

# %%
pca = PCA(n_components = 5)
Xt_pca = pca.fit_transform(df_train_np)
Xt_pca_val = pca.transform(df_val_np)
Xt_pca_test = pca.transform(df_test_np)
X_pca_val = pca.inverse_transform(Xt_pca_val)

# %%
Xt_pca_test.mean(axis=0)

# %%
Xt_pca_val.mean(axis=0)

# %%
df_pca_train = pd.DataFrame(Xt_pca, index=df_train['datetime'].unique())
df_pca_val = pd.DataFrame(Xt_pca_val, index=df_val['datetime'].unique())
df_pca_test = pd.DataFrame(Xt_pca_test, index=df_test['datetime'].unique())

with open(Path.cwd() / 'data' / 'curves_pca_train_5', 'wb') as f:
    pkl.dump(df_pca_train, f)
    
with open(Path.cwd() / 'data' / 'curves_pca_val_5', 'wb') as f:
    pkl.dump(df_pca_val, f)
    
with open(Path.cwd() / 'data' / 'curves_pca_test_5', 'wb') as f:
    pkl.dump(df_pca_test, f)

# %%
idx = np.nonzero(unique_dates_test == pd.to_datetime('2020-08-15 14:00:00'))

# %%
plt.plot(np.squeeze(df_test_np[idx, :]*sd_cols+mean_cols), label='2019-10-15 14:00:00')
plt.legend(loc='upper left')
plt.show()

# %%
plt.plot(np.squeeze(pca.inverse_transform(Xt_pca_test[idx, :])*sd_cols+mean_cols))
plt.show()

# %%
# kernel PCA - finding best hyperparameter setup
max_components = 10
kernels = ['poly', 'rbf', 'sigmoid', 'cosine']
kpca_mse = {}
kpca_mae = {}
for kernel in kernels:
    one_mse = []
    one_mae = []
    for n in range(1, max_components+1):
        kpca = KernelPCA(n_components = n, kernel=kernel, fit_inverse_transform=True, random_state=42, n_jobs=-1)
        kpca.fit(df_train_np)
        reconstructed = kpca.inverse_transform(kpca.transform(df_val_np))
        one_mse.append(mean_squared_error(df_val_np*sd_cols+mean_cols, reconstructed*sd_cols+mean_cols))
        one_mae.append(mean_absolute_error(df_val_np*sd_cols+mean_cols, reconstructed*sd_cols+mean_cols))
        print(f'kernel = {kernel}, components = {n}, test reconstruction MSE = {one_mse[n-1]:.4f}, test reconstruction MAE = {one_mae[n-1]:.4f}')
        
    kpca_mse[kernel] = one_mse
    kpca_mae[kernel] = one_mae

# %%
### simple visualizations to compare how well pca carries over information in lower dimensions

# %%
kpca = KernelPCA(n_components = 5, kernel='cosine', fit_inverse_transform=True, random_state=42, n_jobs=-1)
Xt_kpca = kpca.fit_transform(df_train_np)
Xt_kpca_val = kpca.transform(df_val_np)
Xt_kpca_test = kpca.transform(df_test_np)
X_val = kpca.inverse_transform(Xt_kpca_val)

# %%
df_kpca_train = pd.DataFrame(Xt_kpca, index=df_train['datetime'].unique())
df_kpca_val = pd.DataFrame(Xt_kpca_val, index=df_val['datetime'].unique())
df_kpca_test = pd.DataFrame(Xt_kpca_test, index=df_test['datetime'].unique())

with open(Path.cwd() / 'data' / 'curves_kpca_train_5', 'wb') as f:
    pkl.dump(df_kpca_train, f)
    
with open(Path.cwd() / 'data' / 'curves_kpca_val_5', 'wb') as f:
    pkl.dump(df_kpca_val, f)
    
with open(Path.cwd() / 'data' / 'curves_kpca_test_5', 'wb') as f:
    pkl.dump(df_kpca_test, f)

# %%
idx = np.nonzero(unique_dates_val == pd.to_datetime('2019-12-15 14:00:00'))

plt.plot(np.squeeze(df_val_np[idx, :]*sd_cols+mean_cols), label='2019-10-15 14:00:00')
plt.legend(loc='upper left')
plt.show()

# %%
plt.plot(np.squeeze(X_val[idx, :]*sd_cols+mean_cols))
plt.show()

# %%
# UMAP - finding best hyperparameter setup
no_of_neighbors = [10,11,12,13,14,15]
distances = [.001, .01, .1, .2, .3, .4, .5]
metrics = ['euclidean', 'manhattan', 'chebyshev']

for n_neighbors in no_of_neighbors:
    for min_dist in distances:
        for metric in metrics:
            embedding = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_jobs=-1)
            embedding.fit(df_train_np)
            reconstructed = embedding.inverse_transform(embedding.transform(df_val_np))
            
            print(f'n_neigbors = {n_neighbors}, min_dist = {min_dist}, , metric = {metric}, test reconstruction MSE = {mean_squared_error(df_val_np*sd_cols+mean_cols, reconstructed*sd_cols+mean_cols):.8f}, test reconstruction MAE = {mean_absolute_error(df_val_np*sd_cols+mean_cols, reconstructed*sd_cols+mean_cols):.8f}')

# %%
### simple visualizations to compare how well umap carries over information in lower dimensions

# %%
# 2d embedding with january
# embedding = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.001, metric='manhattan')
# 3d embedding with january
embedding = umap.UMAP(n_components=3, n_neighbors=11, min_dist=0.5, metric='manhattan')
# embeddings without january
#embedding = umap.UMAP(n_components=2, n_neighbors=14, min_dist=0.5, metric='chebyshev')
#embedding = umap.UMAP(n_components=2, n_neighbors=11, min_dist=0.3, metric='minkowski')
Xt_umap = embedding.fit_transform(df_train_np)

# %%
Xt_umap_val = embedding.transform(df_val_np)
Xt_umap_test = embedding.transform(df_test_np)

# %%
df_umap_train = pd.DataFrame(Xt_umap, index=df_train['datetime'].unique())
df_umap_val = pd.DataFrame(Xt_umap_val, index=df_val['datetime'].unique())
df_umap_test = pd.DataFrame(Xt_umap_test, index=df_test['datetime'].unique())

with open(Path.cwd() / 'data' / 'curves_umap_train_3', 'wb') as f:
    pkl.dump(df_umap_train, f)
    
with open(Path.cwd() / 'data' / 'curves_umap_val_3', 'wb') as f:
    pkl.dump(df_umap_val, f)
    
with open(Path.cwd() / 'data' / 'curves_umap_test_3', 'wb') as f:
    pkl.dump(df_umap_test, f)

# %%
idx = np.nonzero(unique_dates_test == pd.to_datetime('2020-08-15 16:00:00'))

# %%
X_umap_val = embedding.inverse_transform(Xt_umap_test[idx])

# %%
plt.plot(np.squeeze(df_test_np[idx, :]*sd_cols+mean_cols))
plt.show()

# %%
plt.plot(np.squeeze(X_umap_val*sd_cols+mean_cols))
plt.show()

# %%
