import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
# import episcanpy as epi
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=Warning)



# dataPath = 'seurat_pbmc/20000/train_dataset/pbmc_8000.h5ad'
# dataPath = 'LUSC_10_types/marge_20000/combined_train.h5ad'
# dataPath = 'LUSC_10_types/marge_4000/combined_train_4000.h5ad'
# dataPath = 'E-MTAB/10000/HP1507101/4000/HP1507101_scData_10000_5000_4000.h5ad'
dataPath = 'GSE257541/10000/pbmc48h/10000/matrix_48h_10000.h5ad'

# dataPath = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/train_2500_20000_16000.h5ad'

adata = sc.read_h5ad(dataPath)
adata.var_names_make_unique()
if 'batch' not in adata.obs:
    adata.obs['batch'] = 'batch'
adata.obs['batch'] = adata.obs['batch'].astype('category')

print(adata)
# # # -----------------------select_var_feature-----------------------------------
import math
# 保留能够开方的最大特征数
n_obs, n_vars = adata.shape
print(n_obs, n_vars)
s = math.floor(math.sqrt(n_vars))
num_features_to_keep = s*s
print(num_features_to_keep)

# 按照顺序选择要保留的特征
indices_to_keep = np.arange(num_features_to_keep)
# 使用索引数组来选择要保留的特征
adata = adata[:, indices_to_keep]
print(adata)

np.random.seed(42)
# np.random.seed(41)
# np.random.seed(40)
# 获取数据的行数
num_rows = adata.shape[0]
# # 随机选择 100 个行索引作为测试集
num_all = 10000
test_indices = np.random.choice(num_rows, size=num_all, replace=False)
# 创建布尔数组用于选择训练集
mask = np.ones(num_rows, dtype=bool)
mask[test_indices] = False

# 使用索引数组来选择要保留的特征
adata = adata[:, indices_to_keep]
# # 使用布尔数组划分训练集和测试集
# adata_train = adata[mask]
# adata_test = adata[test_indices]
adata_all = adata[test_indices]

# 获取数据的行数
num_rows = adata_all.shape[0]
# # 随机选择 100 个行索引作为测试集
num_train = int(num_all*0.8)
test_indices = np.random.choice(num_rows, size=num_train, replace=False)
# 创建布尔数组用于选择训练集
mask = np.ones(num_rows, dtype=bool)
mask[test_indices] = False
# 使用索引数组来选择要保留的特征
adata_all = adata_all[:, indices_to_keep]
# # 使用布尔数组划分训练集和测试集
adata_test = adata_all[mask]
adata_train = adata_all[test_indices]
num_test = int(num_all*0.2)

# adata_path='LUSC_10_types/marge_20000/train_dataset'
# adata_path='LUSC_10_types/marge_20000/train_dataset/1500'
adata_path='GSE257541/10000/pbmc48h/10000'
# adata_path= 'hscr/all_2_seurat_object_common_8/20000/train_dataset/200'
# adata_path= 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/train_dataset'
# datasets_naame ='matrix_12h_5000'
# adata_filtered = adata_path+'/'+datasets_naame+'_train_' + str(num_features_to_keep) + '_' + str(num_all) + '_' + str(num_train) + '.h5ad'
adata_filtered = adata_path+'/train_' + str(num_features_to_keep) + '_' + str(num_all) + '_' + str(num_train) + '.h5ad'
# adata_filtered = 'adata_train_' + str(num_features_to_keep) + '.h5ad'
adata_train.write_h5ad(adata_filtered, compression='gzip')
print(f'AnnData 已保存到 {adata_filtered}')

# adata_filtered = adata_path+'/'+datasets_naame+'_test_' + str(num_features_to_keep) + '_' + str(num_all) + '_' + str(num_test) + '.h5ad'
adata_filtered = adata_path+'/test_' + str(num_features_to_keep) + '_' + str(num_all) + '_' + str(num_test) + '.h5ad'
# adata_filtered = 'adata_train_' + str(num_features_to_keep) + '.h5ad'
adata_test.write_h5ad(adata_filtered, compression='gzip')
print(f'AnnData 已保存到 {adata_filtered}')
