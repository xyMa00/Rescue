from .layer import *
from .model import *
from .loss import *
from .layer import *
from .model import *
from .loss import *
# from .dataset import load_dataset
from .utils import estimate_k, binarization
from .dataset import *

import time
import torch
import numpy as np
import pandas as pd
import os
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
# import scanpy as sc
sc.settings.autoshow = False
from anndata import AnnData
from typing import Union, List
from .plot import *
from sklearn.metrics import f1_score
from .ResNetAE_pytorch import ResNet_pred
import episcanpy.api as epi
import anndata as ad
from torch.utils.data import DataLoader
from .forebrain import ForeBrain
from labels_statistic import *


def get_score(true_labels, kmeans_labels):
    # 假设 kmeans_labels 是 K-Means 聚类的结果
    # 假设 true_labels 是真实的标签（如果有的话）
    # 如果没有真实标签，可以使用其他方法来评估聚类结果
    # 计算调整后的兰德指数（ARI）
    ari = adjusted_rand_score(true_labels, kmeans_labels)
    print("Adjusted Rand Index (ARI):", ari)
    # 计算归一化互信息（NMI）
    nmi = normalized_mutual_info_score(true_labels, kmeans_labels)
    print("Normalized Mutual Information (NMI):", nmi)
    # 计算 F1 分数
    f1 = f1_score(true_labels, kmeans_labels, average='weighted')
    print("F1 Score:", f1)
    return ari, nmi, f1



def some_function(
        data_list:Union[str, List],
        test_list:Union[str, List],
        n_centroids:int = 30,
        outdir:bool = None,
        verbose:bool = False,
        pretrain:str = None,
        lr:float = 0.0002,
        batch_size:int = 64,
        gpu:int = 0,
        seed:int = 18,
        encode_dim:List = [1024, 128],
        decode_dim:List = [],
        latent:int = 10,
        min_peaks:int = 100,
        min_cells:Union[float, int] = 3,
        n_feature:int = 100000,
        log_transform:bool = False,
        max_iter:int = 30000,
        weight_decay:float = 5e-4,
        impute:bool = False,
        binary:bool = False,
        embed:str = 'UMAP',
        reference:str = 'cell_type',
        cluster_method:str = 'leiden',
    )->AnnData:

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    print("\n**********************************************************************")
    print(" Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome.")
    print("**********************************************************************\n")
    if not pretrain:
        adata, trainloader, testloader = load_dataset_c(
            n_centroids,
            data_list,
            test_list,
            min_genes=min_peaks,
            min_cells=min_cells,
            n_top_genes=n_feature,
            batch_size=batch_size,
            log=None,
        )
    else:
        # -----------test-----------------
        adata, testloader = load_dataset_test(
            n_centroids,
            data_list,
            test_list,
            min_genes=min_peaks,
            min_cells=min_cells,
            n_top_genes=n_feature,
            batch_size=batch_size,
            log=None,
        )
    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    s = math.floor(math.sqrt(n_vars))
    cell_num = adata.shape[0]
    input_dim = adata.shape[1]
    k = n_centroids

    if outdir:
        outdir = outdir + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('outdir: {}'.format(outdir))

    print("\n======== Parameters ========")
    print(
        'Cell number: {}\nPeak number: {}\ndevice: {}\nlr: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}\nn_centroids: {}'.format(
            cell_num, input_dim, device, lr, batch_size, min_peaks, min_cells, k))
    print("============================")

    latent = 10
    encode_dim = [1024, 128]
    decode_dim = []
    dims = [input_dim, latent, encode_dim, decode_dim]

    model = ResNet_pred(input_shape=(s, s, 1), dims=dims, n_centroids=k,).to(device)
    model = model.to(torch.float)

    if not pretrain:
        print('\n## Training Model ##')
        model.fit_res(adata, trainloader, testloader, batch_size, k,
                             lr=lr,
                             verbose=verbose,
                             device=device,
                             max_iter=max_iter,
                             outdir=outdir
                             )

    else:
        model_path ='pre/model_LUSC_IA3_BC3_10_2500_5000_4000.pt'
        print('\n## Loading Model: {}\n'.format(model_path))
        # 使用 torch.load() 加载模型状态字典，并映射到CPU
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)

        output_pre = []
        with torch.no_grad():
            for inputs, labels in testloader:
                target_shape = (-1, 1, model.input_shape,  model.input_shape)
                # 重塑张量的形状
                inputs = inputs.view(target_shape)
                inputs = inputs.to(torch.float)
                inputs = inputs.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备

                outputs = model.encoder(inputs)
                output_pre.append(outputs.cpu().numpy())
        # 将所有批次的输出拼接成一个数组
        output_pre = np.concatenate(output_pre, axis=0)

        column_indices = adata.obs.columns[:n_centroids]
        row_indices = adata.obs.index
        output_df = pd.DataFrame(output_pre, index=row_indices, columns=column_indices)
        output_df.to_csv('pre/model_outputs.txt', sep='\t', index=True, header=True)
    print("over....")
