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

import episcanpy.api as epi

# np.warnings.filterwarnings('ignore')
DATA_PATH = os.path.expanduser("~")+'/.scalex/'
CHUNK_SIZE = 20000


def read_mtx(path):
    """
    Read mtx format data folder including: 
        matrix file: e.g. count.mtx or matrix.mtx
        barcode file: e.g. barcode.txt
        feature file: e.g. feature.txt
    """
    for filename in glob(path+'/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path+'/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            print(len(barcode), adata.shape[0])
            if len(barcode) != adata.shape[0]:
                adata = adata.transpose()
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename or 'feature' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            if len(gene) != adata.shape[1]:
                adata = adata.transpose()
            adata.var = pd.DataFrame(index=gene)
    return adata


def load_file(path):  
    """
    Load single cell dataset from file
    """
    if os.path.exists(DATA_PATH+path+'.h5ad'):
        adata = sc.read_h5ad(DATA_PATH+path+'.h5ad')
    elif os.path.isdir(path): # mtx format
        adata = read_mtx(path)
    elif os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac'])):
        import muon as mu
        adata = mu.read(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata


def load_files(root):
    """
    Load single cell dataset from files
    """
    if root.split('/')[-1] == '*':
        adata = []
        for root in sorted(glob(root)):
            adata.append(load_file(root))
        return AnnData.concatenate(*adata, batch_key='sub_batch', index_unique=None)
    else:
        return load_file(root)
    
    
def concat_data(
        data_list, 
        batch_categories=None, 
        join='inner',             
        batch_key='batch', 
        index_unique=None, 
        save=None
    ):
    """
    Concat multiple datasets
    """
    if len(data_list) == 1:
        return load_files(data_list[0])
    elif isinstance(data_list, str):
        return load_files(data_list)
    adata_list = []
    for root in data_list:
        adata = load_files(root)
        adata_list.append(adata)
        
    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)
    [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                batch_categories=batch_categories, index_unique=index_unique)  
    if save:
        concat.write(save, compression='gzip')
    return concat
        
    
def preprocessing_atac(
        adata, 
        min_genes=200, 
        min_cells=0.01, 
        n_top_genes=30000,
        target_sum=None,
        chunk_size=CHUNK_SIZE,
        log=None
    ):
    """
    preprocessing
    """
    print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
        
    adata.X[adata.X>1] = 1
    
    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if log: log.info('Filtering genes')
    if min_cells < 1:
        min_cells = min_cells * adata.shape[0]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if n_top_genes != -1:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
        # adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)
    
    # if log: log.infor('Normalizing total per cell')
    # sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if log: log.info('Batch specific maxabs scaling')
    
#     adata.X = maxabs_scale(adata.X)
#     adata = batch_scale(adata, chunk_size=chunk_size)
    
    print('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing_atac_n(
        adata,
        min_genes=200,
        min_cells=0.01,
        n_top_genes=30000,
        target_sum=None,
        chunk_size=CHUNK_SIZE,
        log=None
):
    """
    preprocessing
    """
    # print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata.X[adata.X > 1] = 1

    # if log: log.info('Filtering cells')
    # sc.pp.filter_cells(adata, min_genes=min_genes)
    # print(adata)
    # if log: log.info('Filtering genes')
    # # if min_cells < 1:
    # #     min_cells = min_cells * adata.shape[0]
    # sc.pp.filter_genes(adata, min_cells=min_cells)
    # print(adata)
    # if n_top_genes != -1:
    #     if log: log.info('Finding variable features')
    #     # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
    #     adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)

    # if log: log.infor('Normalizing total per cell')
    # sc.pp.normalize_total(adata, target_sum=target_sum)

    if log: log.info('Batch specific maxabs scaling')

    #     adata.X = maxabs_scale(adata.X)
    #     adata = batch_scale(adata, chunk_size=chunk_size)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

def batch_scale(adata, chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data
    """
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch']==b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        for i in range(len(idx)//chunk_size+1):
            adata.X[idx[i*chunk_size:(i+1)*chunk_size]] = scaler.transform(adata.X[idx[i*chunk_size:(i+1)*chunk_size]])

    return adata
        

def reindex(adata, genes):
    """
    Reindex AnnData with gene list
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    new_X = scipy.sparse.csr_matrix((adata.shape[0], len(genes)))
    new_X[:, idx] = adata[:, genes[idx]].X
    adata = AnnData(new_X, obs=adata.obs, var={'var_names':genes}) 
    return adata
        
    
class SingleCellDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata, k):
        self.adata = adata
        self.shape = adata.shape

        self.k = k  # 要保留的前 k 列
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray().squeeze()
        # label = self.adata.obs[idx]  # 标签数据
        label = self.adata.obs.iloc[idx, :self.k].values.astype(np.float32)
        # return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        return x, label


def load_dataset(
        data_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        min_genes=600,
        min_cells=0.01,
        n_top_genes=30000,
        batch_size=64,
        chunk_size=CHUNK_SIZE,
        log=None,
        transpose=False,
    ):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name!='batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    adata = preprocessing_atac(
        adata,
        min_genes=min_genes,
        min_cells=min_cells,
        n_top_genes=n_top_genes,
        chunk_size=chunk_size,
        log=log,
    )

    scdata = SingleCellDataset(adata) # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4
    )
#     batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
#     testloader = DataLoader(scdata, batch_sampler=batch_sampler)

    return adata, trainloader, testloader


def load_dataset_new(
        data_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        min_genes=600,
        min_cells=0.01,
        n_top_genes=30000,
        batch_size=64,
        chunk_size=CHUNK_SIZE,
        log=None,
        transpose=False,
):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    adata = preprocessing_atac_n(
        adata,
        min_genes=min_genes,
        min_cells=min_cells,
        n_top_genes=n_top_genes,
        chunk_size=chunk_size,
        log=log,
    )

    scdata = SingleCellDataset(adata)  # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4
    )
    #     batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    #     testloader = DataLoader(scdata, batch_sampler=batch_sampler)

    return adata, trainloader, testloader

def load_dataset_c(
        k,
        data_list,
        test_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        min_genes=600,
        min_cells=0.01,
        n_top_genes=30000,
        batch_size=64,
        chunk_size=CHUNK_SIZE,
        log=None,
        transpose=False,
):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    adata_test = concat_data(test_list, batch_categories, join=join, batch_key=batch_key)


    # print(adata)
    # # # # -----------------------select_var_feature-----------------------------------
    # # import math
    # # # 保留能够开方的最大特征数
    # # n_obs, n_vars = adata.shape
    # # print(n_obs, n_vars)
    # # s = math.floor(math.sqrt(n_vars))
    # # num_features_to_keep = s*s
    # # print(num_features_to_keep)
    # #
    # # # # 指定要保留的特征数量
    # # # num_features_to_keep = 43264  # 例如，要保留1000个特征
    # # # # 获取 X 的形状
    # # # n_obs, n_vars = adata.shape
    # # # # 随机生成一个索引数组，表示要保留的特征
    # # # indices_to_keep = np.random.choice(n_vars, num_features_to_keep, replace=False)
    # # # 按照顺序选择要保留的特征
    # # indices_to_keep = np.arange(num_features_to_keep)
    # # # 使用索引数组来选择要保留的特征
    # # adata = adata[:, indices_to_keep]
    # # print(adata)
    # #
    # # adata_filtered = 'adata_mouse_train_'+str(num_features_to_keep)+'.h5ad'
    # # # adata_filtered = 'adata_train_' + str(num_features_to_keep) + '.h5ad'
    # # adata.write_h5ad(adata_filtered, compression='gzip')
    # # print(f'AnnData 已保存到 {adata_filtered}')
    #
    #
    # # adata = preprocessing_atac_n(
    # #     adata,
    # #     min_genes=min_genes,
    # #     min_cells=min_cells,
    # #     n_top_genes=n_top_genes,
    # #     chunk_size=chunk_size,
    # #     log=log,
    # # )
    #
    # # 设置随机种子以保证可重复性
    # np.random.seed(42)
    # # 获取数据的行数
    # num_rows = adata.shape[0]
    # # # 随机选择 100 个行索引作为测试集
    # # test_indices = np.random.choice(num_rows, size=100, replace=False)
    # # # 创建布尔数组用于选择训练集
    # # mask = np.ones(num_rows, dtype=bool)
    # # mask[test_indices] = False
    #
    # # 按照顺序选择要保留的特征
    # # test_indices = np.arange(1000)
    # # mask = np.ones(num_rows, dtype=bool)
    # # mask[test_indices] = False
    # # # # 使用索引数组来选择要保留的特征
    # # # adata = adata[:, indices_to_keep]
    # #
    # # # 使用布尔数组划分训练集和测试集
    # # adata_train = adata[mask]
    # # adata_test = adata[test_indices]
    #
    #
    #
    # # 确定测试集的起始索引(选取倒数1000行作为测试)
    # #test_start_index = num_rows - 1000
    # #test_start_index = int(num_rows * (1-0.2))
    # test_start_index = int(num_rows * (1-0.1))
    # # 提取最后1000行数据作为测试集
    # adata_test = adata[test_start_index:]
    # # 提取前面部分作为训练集
    # adata_train = adata[:test_start_index]
    #
    #
    # # # 假设已经有了名为adata_train的AnnData对象
    # # # 复制adata_train
    # # adata_train_copy = adata_train.copy()
    # # # 合并两个数据集
    # # adata_train_combined = AnnData.concatenate(adata_train, adata_train_copy)
    #
    # # print(adata_train_combined)
    #
    # # scdata_train = SingleCellDataset(adata_train_combined)  # Wrap AnnData into Pytorch Dataset
    # scdata_train = SingleCellDataset(adata_train, k)  # Wrap AnnData into Pytorch Dataset

    scdata_train = SingleCellDataset(adata, k)  # Wrap AnnData into Pytorch Dataset
    # print(scdata_train)
    trainloader = DataLoader(
        scdata_train,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=4
    )
    scdata_test = SingleCellDataset(adata_test, k)  # Wrap AnnData into Pytorch Dataset
    # testloader = DataLoader(scdata_test, batch_size=batch_size, drop_last=False, shuffle=True)
    testloader = DataLoader(scdata_test, batch_size=500, drop_last=False, shuffle=True)
    return adata, trainloader, testloader





    # scdata = SingleCellDataset(adata)  # Wrap AnnData into Pytorch Dataset
    # print(scdata)
    # trainloader = DataLoader(
    #     scdata,
    #     batch_size=batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=4
    # )
    # print(trainloader)
    # #     batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    # testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    # #     testloader = DataLoader(scdata, batch_sampler=batch_sampler)
    # print(testloader)
    # return adata, trainloader, testloader

def load_dataset_test(k,
        data_list,
        test_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        min_genes=600,
        min_cells=0.01,
        n_top_genes=30000,
        batch_size=64,
        chunk_size=CHUNK_SIZE,
        log=None,
        transpose=False,
):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(test_list, batch_categories, join=join, batch_key=batch_key)
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    # adata = preprocessing_atac_n(
    #     adata,
    #     min_genes=min_genes,
    #     min_cells=min_cells,
    #     n_top_genes=n_top_genes,
    #     chunk_size=chunk_size,
    #     log=log,
    # )


    # # 假设已经有了名为adata的AnnData对象
    # # 获取唯一的细胞类型
    # cell_types = np.unique(adata.obs['cell_type'])
    # # 用于存储训练集和测试集的列表
    # train_indices = []
    # test_indices = []
    # # 对每个细胞类型进行循环
    # for cell_type in cell_types:
    #     # 获取属于当前细胞类型的索引
    #     indices = np.where(adata.obs['cell_type'] == cell_type)[0]
    #     # 随机打乱索引
    #     np.random.shuffle(indices)
    #     # 计算80%的索引
    #     train_size = int(0.8 * len(indices))
    #     # 将索引分成训练集和测试集
    #     train_indices.extend(indices[:train_size])
    #     test_indices.extend(indices[train_size:])
    #
    # # 根据索引获取训练集和测试集
    # adata_train = adata[train_indices]
    # adata_test = adata[test_indices]
    #
    #
    # # 假设已经有了名为adata_train的AnnData对象
    #
    # # 复制adata_train
    # adata_train_copy = adata_train.copy()
    #
    # # 合并两个数据集
    # adata_train_combined = AnnData.concatenate(adata_train, adata_train_copy)
    #
    # # print(adata_train_combined)
    #
    # scdata_train = SingleCellDataset(adata_train)  # Wrap AnnData into Pytorch Dataset
    # # print(scdata_train)
    # trainloader = DataLoader(
    #     scdata_train,
    #     batch_size=batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=4
    # )
    scdata_test = SingleCellDataset(adata, k)  # Wrap AnnData into Pytorch Dataset
    testloader = DataLoader(scdata_test, batch_size=batch_size, drop_last=False, shuffle=False)
    return adata, testloader





    # scdata = SingleCellDataset(adata)  # Wrap AnnData into Pytorch Dataset
    # print(scdata)
    # trainloader = DataLoader(
    #     scdata,
    #     batch_size=batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=4
    # )
    # print(trainloader)
    # #     batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    # testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    # #     testloader = DataLoader(scdata, batch_sampler=batch_sampler)
    # print(testloader)
    # return adata, trainloader, testloader