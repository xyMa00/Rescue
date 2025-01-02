from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.nn import functional as F
import torch
from .Resnet18 import *
from tqdm import tqdm, trange
import os
import numpy as np
import scanpy as sc
import anndata as ad
import episcanpy.api as epi
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def lins_ccc(x, y):
    """
    计算 Lin's Concordance Correlation Coefficient (CCC)
    参数:
    x: 第一组数据 (numpy array)
    y: 第二组数据 (numpy array)
    返回:
    ccc: Lin's CCC 值
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return ccc
def kl_divergence(mu, logvar):
    """
        Computes the KL-divergence of
        some element z.
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)

class ResNet_pred(torch.nn.Module):
    def __init__(self,
                 # (106,106,1)
                 input_shape=(256, 256, 3),
                 n_centroids=14,
                 dims=[],
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNet_pred, self).__init__()
        assert input_shape[0] == input_shape[1]
        self.input_shape = input_shape[0]
        self.pre_resnet = seresnet18(num_classes=n_centroids)
    def loss_function(self, x, label_list, k,device):
        z = self.pre_resnet(x)
        criterion = nn.MSELoss()
        labels = torch.tensor(label_list)  # 示例标签
        labels = labels.to(device)
        loss = criterion(z, labels)
        ccc_value = lins_ccc(z, labels)
        return loss, ccc_value

    def fit_res(self, adata, dataloader, dataloader_test, batch_size, k,
            lr=0.002,
            weight_decay=5e-4,
            device='cpu',
            beta=1,
            n=200,
            max_iter=30000,
            verbose=True,
            patience=100,
            outdir=None,
            ):
        self.to(device)
        # 开始计时
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Beta = DeterministicWarmup(n=n, t_max=beta)
        iteration = 0
        n_epoch = 1501
        # early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        epoch_min = 0
        count = 0
        ccc_value_max = 0
        min_loss = 1
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                count = count + 1
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                # 遍历数据加载器的每个批次
                for i, (x, labels) in tk0:
                    # print(labels)
                    # 将类标签转换为对应的整数形式
                    target_shape = (-1, 1, self.input_shape, self.input_shape)
                    # 重塑张量的形状
                    x = x.view(target_shape)
                    x = x.to(torch.float)
                    x = x.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备
                    optimizer.zero_grad()
                    loss, ccc_value = self.loss_function(x, labels, k, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()
                    tk0.set_postfix_str('loss={:.3f}'.format(
                        loss))
                    tk0.update(1)
                    iteration += 1
                # save model
                if loss < min_loss:
                    count=0
                    epoch_min = epoch
                    min_loss = loss
                    ccc_value_max = ccc_value
                    if outdir:
                        sc.settings.figdir = outdir
                        torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                    print(f"\nsave at epoch:{epoch}")
                print(f"\nepoch_now:{epoch},loss:{loss},ccc_value: {ccc_value},epoch_min:{epoch_min},min_loss: {min_loss}, ccc_value_min: {ccc_value_max}")
                if count == 500 or min_loss==0:
                    print(f"\nearly stop........\nepoch:{epoch_min},min_loss: {min_loss:.4f},ccc_value_min: {ccc_value_max}")
                    break