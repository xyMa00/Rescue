from .layer import *
from .loss import *
from .layer import *
from .loss import *
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
sc.settings.autoshow = False
from anndata import AnnData
from typing import Union, List
from sklearn.metrics import f1_score
from .ResNetAE_pytorch import ResNet_pred
import episcanpy.api as epi
import anndata as ad
from torch.utils.data import DataLoader
from labels_statistic import *



def some_function(
        data_list:Union[str, List],
        test_list:Union[str, List],
        model_path:Union[str, List],
        # n_centroids:int = 30,
        outdir:bool = None,
        verbose:bool = False,
        pretrain:str = None,
        lr:float = 0.0002,
        batch_size:int = 64,
        gpu:int = 0,
        seed:int = 18,
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
    n_centroids = 0
    if not pretrain:
        adata, trainloader, testloader,k = load_dataset_c(
            n_centroids,
            data_list,
            test_list,
            batch_size=batch_size,
        )
    else:
        # -----------test-----------------
        adata, testloader, k = load_dataset_test(
            n_centroids,
            data_list,
            test_list,
            batch_size=batch_size,
        )
    n_obs, n_vars = adata.shape
    s = math.floor(math.sqrt(n_vars))
    cell_num = adata.shape[0]
    input_dim = adata.shape[1]

    if outdir:
        outdir = outdir + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # print('outdir: {}'.format(outdir))
    print("\n======== Parameters ========")
    print(
        'Cell number: {}\nPeak number: {}\ndevice: {}\nlr: {}\nbatch_size: {}\nn_celltypess: {}'.format(
            cell_num, input_dim, device, lr, batch_size, k))
    print("============================")
    model = ResNet_pred(input_shape=(s, s, 1),  n_centroids=k,).to(device)
    model = model.to(torch.float)

    if not pretrain:
        print('\n## Training Model ##')
        model.fit_res(adata, trainloader, testloader, batch_size, k,
                             lr=lr,
                             verbose=verbose,
                             device=device,
                             outdir=outdir
                             )

    else:
        # model_path ='pre/model_pbmc3k_9_10000_4000.pt'
        model_path = model_path
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

                outputs = model.pre_resnet(inputs)
                output_pre.append(outputs.cpu().numpy())
        # 将所有批次的输出拼接成一个数组
        output_pre = np.concatenate(output_pre, axis=0)
        column_indices = adata.obs.columns[:k]
        row_indices = adata.obs.index
        output_df = pd.DataFrame(output_pre, index=row_indices, columns=column_indices)
        output_df.to_csv('pre/model_outputs.txt', sep='\t', index=True, header=True)
    print("over....")
