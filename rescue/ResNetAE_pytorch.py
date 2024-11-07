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

from sklearn.mixture import GaussianMixture

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
# def get_score(true_labels, kmeans_labels):
#     # 假设 kmeans_labels 是 K-Means 聚类的结果
#     # 假设 true_labels 是真实的标签（如果有的话）
#     # 如果没有真实标签，可以使用其他方法来评估聚类结果
#     # 计算调整后的兰德指数（ARI）
#     ari = adjusted_rand_score(true_labels, kmeans_labels)
#     print("Adjusted Rand Index (ARI):", ari)
#     # 计算归一化互信息（NMI）
#     nmi = normalized_mutual_info_score(true_labels, kmeans_labels)
#     print("Normalized Mutual Information (NMI):", nmi)
#     # 计算 F1 分数
#     # f1 = f1_score(true_labels, kmeans_labels, average='weighted')
#     f1 = 0
#     print("F1 Score:", f1)
#     return ari, nmi, f1

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
def elbo(recon_x, x, z_params, binary=True):
    """
    elbo = likelihood - kl_divergence
    L = -elbo

    Params:
        recon_x:
        x:
    """
    mu, logvar = z_params
    kld = kl_divergence(mu, logvar)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x)
    else:
        likelihood = -F.mse_loss(recon_x, x)
    return torch.sum(likelihood), torch.sum(kld)


# def elbo_SCALE(recon_x, x, gamma, c_params, z_params, binary=True):
def elbo_SCALE(recon_x, x, binary=True):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    # mu_c, var_c, pi = c_params;  # print(mu_c.size(), var_c.size(), pi.size())
    # var_c += 1e-8
    # n_centroids = pi.size(1)
    # mu, logvar = z_params
    # mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    # logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(x|z)
    if binary:
        likelihood = -binary_cross_entropy(recon_x,
                                           x)  # ;print(logvar_expand.size()) #, torch.exp(logvar_expand)/var_c)
    else:
        likelihood = -F.mse_loss(recon_x, x)

        # # 设置Huber损失的delta值（平滑参数）
        # huber_loss = nn.SmoothL1Loss()
        # # 计算Huber损失
        # likelihood = -huber_loss(recon_x, x)

    # # log p(z|c)
    # logpzc = -0.5 * torch.sum(gamma * torch.sum(math.log(2 * math.pi) + \
    #                                             torch.log(var_c) + \
    #                                             torch.exp(logvar_expand) / var_c + \
    #                                             (mu_expand - mu_c) ** 2 / var_c, dim=1), dim=1)
    #
    # # log p(c)
    # logpc = torch.sum(gamma * torch.log(pi), 1)
    #
    # # log q(z|x) or q entropy
    # qentropy = -0.5 * torch.sum(1 + logvar + math.log(2 * math.pi), 1)
    #
    # # log q(c|x)
    # logqcx = torch.sum(gamma * torch.log(gamma), 1)
    #
    # kld = -logpzc - logpc + qentropy + logqcx

    from scipy.stats import entropy
    # # # 计算 X 和 Y 的直方图
    # # hist_X, bins_X = np.histogram(x, bins='auto', density=True)
    # # hist_Y, bins_Y = np.histogram(recon_x, bins='auto', density=True)
    # # # 计算 KL 散度
    # # kl_loss = entropy(hist_X, hist_Y)
    # # print("KL Loss:", kl_loss)
    #
    # # 将张量转换为 NumPy 数组
    # X_np = x.cpu().detach().numpy().reshape(-1)
    #
    # Y_np = recon_x.cpu().detach().numpy().reshape(-1)
    #
    # # 计算 X 和 Y 的直方图
    # hist_X, bins_X = np.histogram(X_np, bins=X_np.size, density=True)
    # hist_Y, bins_Y = np.histogram(Y_np, bins=Y_np.size, density=True)
    # # 计算 KL 散度
    # # kl_loss = entropy(hist_X, hist_Y)
    # # print("KL Loss:", kl_loss)
    kl_loss = entropy(x.detach().numpy(), recon_x.detach().numpy())

    # 直接将kl_loss转换为PyTorch张量
    kl_loss_tensor = torch.tensor(kl_loss)

    # return torch.sum(likelihood), torch.sum(kld)
    return torch.sum(likelihood), torch.sum(kl_loss_tensor)


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir=None):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt') if outdir else None

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        if self.model_file:
            torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss
class ResNetVAE(torch.nn.Module):
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
        super(ResNetVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        # self.pi = nn.Parameter(torch.ones(n_centroids) / n_centroids)  # pc
        # self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids))  # mu
        # self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids))  # sigma^2
        # self.n_centroids = n_centroids
        self.input_shape = input_shape[0]

        image_channels = input_shape[2]

        self.z_dim = z_dim
        # 图像的潜在维度=输入图像的高度除以 2 的 n_levels 次幂,img_latent_dim=106/16=6
        # self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.img_latent_dim = 8

        self.encoder = ResNet18(num_classes=n_centroids)
        # self.encoder = CustomEncoder(num_classes=n_centroids)


        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.decoder = Decoder([z_dim, decode_dim, x_dim], output_activation=nn.Sigmoid())
        self.attention_z = nn.MultiheadAttention(embed_dim=z_dim, num_heads=5)
        self.dropout = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(z_dim)
        self.batch_norm =nn.BatchNorm1d(z_dim)
        self.fc0 = torch.nn.Linear(z_dim, 512)
        self.attention_1 = nn.MultiheadAttention(embed_dim=512, num_heads=16)
        # self.decoder = torch.nn.Linear(z_dim, 1 * input_shape[0] * input_shape[1])

        # self.layer_norm1 = nn.LayerNorm(input_dim)
        # self.layer_norm2 = nn.LayerNorm(input_dim)

        # Assumes the input to be of shape 256x256
        self.fc21 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc22 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc3 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

        self.fc4 = torch.nn.Linear(z_dim, 128)
        self.fc5 = torch.nn.Linear(128, 256)
        # self.fc6 = torch.nn.Linear(256, 1*106*106)
        # self.fc6 = torch.nn.Linear(256, 1 * 557 * 557)
        self.fc6 = torch.nn.Linear(256, 1 * input_shape[0] * input_shape[1])
        # self.fc6 = torch.nn.Linear(256, 512)
        # self.fc7 = torch.nn.Linear(512, 1*106*106)

    # def loss_function(self, x):
    #     # z, mu, logvar = self.encoder(x)
    #     z = self.encoder(x)
    #     recon_x = self.decode(z)
    #
    #     # gamma, mu_c, var_
    #     c, pi = self.get_gamma(z)
    #     # likelihood, kl_loss = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=True)
    #     likelihood, kl_loss = elbo_SCALE(recon_x, x, binary=True)
    #
    #     return (-likelihood, kl_loss)
    def loss_function(self, x, label_list, k):
        # z, mu, logvar = self.encoder(x)
        z = self.encoder(x)
        criterion = nn.CrossEntropyLoss()

        # my_list = [2]
        labels = torch.tensor(label_list)  # 示例标签
        # 获取标签张量的大小和类别数量
        batch_size = labels.size(0)
        num_classes = k  # 假设有3个类别
        # 创建一个大小为 [batch_size, num_classes] 的零张量，并使用scatter_函数将对应位置设为1
        one_hot_labels = torch.zeros(batch_size, num_classes)
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(z, one_hot_labels)
        # loss = loss + likelihood

        return loss

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1)  # NxK
        #         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = self.var_c.repeat(N, 1, 1) + 1e-8  # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(
            torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / (2 * var_c),
                                      dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi
    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init Ressac model with GMM model parameters
        """
        # n_components是GMM的成分数（即高斯分量的个数），covariance_type='diag'表示每个高斯分量的协方差矩阵是对角矩阵。
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        # 这里调用了self对象的另一个方法encodeBatch()，它接受dataloader和device作为输入，可能用于对输入数据进行编码，
        # 并返回编码后的结果z。
        z = self.encodeBatch(dataloader, device)
        # 这一步是对编码后的数据z进行GMM拟合，即使用GMM对数据进行聚类，估计各个高斯分量的参数（均值和协方差矩阵）
        gmm.fit(z)
        # 这里将拟合得到的GMM模型的均值参数（gmm.means_）转换为PyTorch张量，并将其复制到Ressac模型的参数mu_c中。
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        # 这里将拟合得到的GMM模型的协方差矩阵参数（gmm.covariances_）转换为PyTorch张量，并将其复制到Ressac模型的参数var_c中。
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x in dataloader:
            # target_shape = (-1, 1, 106, 106)
            # target_shape = (-1, 1, 557, 557)
            target_shape = (-1, 1, self.input_shape, self.input_shape)
            # 重塑张量的形状
            x = x.view(target_shape).to(device)
            x = x.to(torch.float)
            # 使用模型的编码器(self.encoder)将输入数据 x
            # 编码为潜在向量 z，以及潜在向量 z 对应的均值 mu 和对数方差 logvar。
            z = self.encoder(x)
            # z, mu, logvar = self.encoder(x)
            # 如果 out 参数是 'z'，表示需要返回潜在向量 z，则将 z 添加到 output 列表中，
            # 同时使用 detach().cpu() 将结果从计算设备中分离，并转移到 CPU 上。
            if out == 'z':
                output.append(z.detach().cpu())
            # 如果 out 参数是 'x'，表示需要返回解码后的数据 x，则使用模型的解码器(self.decoder)对潜在向量 z 进行解码，
            # 并将解码结果添加到 output 列表中。
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            # 如果 out 参数是 'logit'，表示需要返回聚类分配，这里使用了一个 self.get_gamma(z)[0] 方法，
            # 该方法似乎与聚类相关，返回一个与 z 相关的聚类分配。将聚类分配结果添加到 output 列表中。
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)
        # 将 output 列表中的所有结果拼接成一个大的张量，并将其转换为 NumPy 数组形式。
        output = torch.cat(output).numpy()

        return output

    def fit_res_at_mlp(self, adata, dataloader, dataloader_test, batch_size, k,
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
        # n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = 1000
        # early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        # ari_louvain_max = 0
        # ari_leiden_max = 0
        # ari_kmeans_max = 0
        # epoch_max = 0
        # nmi_max = 0
        # f1_max = 0
        accuracy_max = 0
        # 定义类标签映射
        label_mapping = {'EX3': 0, 'OC': 1, 'AC': 2, 'EX2': 3, 'IN1': 4, 'IN2': 5, 'MG': 6, 'EX1': 7}
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                # epoch_recon_loss, epoch_kl_loss = 0, 0
                running_loss = 0

                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                # 遍历数据加载器的每个批次
                for i, (x, labels) in tk0:
                    # print(labels)

                    # 将类标签转换为对应的整数形式
                    label_indices = [label_mapping[label] for label in labels]

                    # target_shape = (32, 1, 106, 106)
                    # target_shape = (32, 1, 557, 557)
                    target_shape = (batch_size, 1, self.input_shape, self.input_shape)
                    # 重塑张量的形状
                    x = x.view(target_shape)
                    x = x.to(torch.float)
                    x = x.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备
                    optimizer.zero_grad()

                    loss = self.loss_function(x, label_indices, k)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()
                    tk0.set_postfix_str('loss={:.3f}'.format(
                        loss))
                    tk0.update(1)
                    iteration += 1
                # tq.set_postfix_str('recon_loss {:.3f}'.format(
                #     epoch_recon_loss / ((i + 1) * len(x))))
                # print("test begain...\n")

                if epoch % 10 == 0:
                    # ------------------------test----------------------
                    print("test begain...\n")
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in dataloader_test:
                            label_indices = [label_mapping[label] for label in labels]
                            labels = torch.tensor(label_indices)  # 示例标签
                            # 获取标签张量的大小和类别数量
                            batch_size = labels.size(0)
                            num_classes = k  # 假设有3个类别
                            # 创建一个大小为 [batch_size, num_classes] 的零张量，并使用scatter_函数将对应位置设为1
                            one_hot_labels_true = torch.zeros(batch_size, num_classes)
                            one_hot_labels_true.scatter_(1, labels.view(-1, 1), 1)
                            # print(one_hot_labels_true)

                            target_shape = (batch_size, 1, self.input_shape, self.input_shape)
                            # 重塑张量的形状
                            inputs = inputs.view(target_shape)
                            inputs = inputs.to(torch.float)
                            inputs = inputs.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备

                            outputs = self.encoder(inputs)
                            # print(outputs)

                            max_indices = torch.argmax(outputs, dim=1)
                            # 使用 torch.zeros_like 创建一个全零张量，并在每行中最大值所在的索引位置设置为1
                            one_hot_tensor_pred = torch.zeros_like(outputs)
                            one_hot_tensor_pred[range(outputs.size(0)), max_indices] = 1
                            # print(one_hot_tensor_pred)

                            # 逐元素比较两个张量是否相等，生成一个布尔类型的张量
                            equal_elements = one_hot_labels_true == one_hot_tensor_pred
                            # 按行统计所有元素都相等的行的个数
                            num_rows_all_equal = (equal_elements.sum(dim=1) == equal_elements.size(1)).sum().item()

                            # print(num_rows_all_equal)


                            total += batch_size
                            correct += num_rows_all_equal

                        # accuracy = correct / total
                        # print(f"all:{total},correct:{correct},Accuracy on test set: {accuracy:.2%}")

                    accuracy = correct / total
                    print(f"epoch:{epoch},Accuracy on test set: {accuracy:.2%}")
                    if accuracy > accuracy_max:
                        if outdir:
                            sc.settings.figdir = outdir
                            # save = str(epoch)+'.png'
                            # save = '.png'
                            # torch.save(self.state_dict(), os.path.join(outdir, 'model_'+str(epoch)+'.pt'))  # save model
                            torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                            # adata.write(outdir + 'adata.h5ad', compression='gzip')

                # if epoch % 10 == 0:
                #     # adata.obsm['latent'] = self.encodeBatch(dataloader_test, device=device, out='z')
                #     # # print(adata.obsm['latent'])
                #     # # 2. cluster
                #     # sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
                #     # from sklearn.cluster import KMeans
                #     # kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
                #     # adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)
                #
                #     # result = kmeans.labels_
                #     # ari, nmi, f1 = get_score(data.labels, result)
                #
                #     # print(adata)
                #     adata.obsm['latent'] = self.encodeBatch(dataloader_test, device=device, out='z')
                #     # print(adata.obsm['latent'])
                #     # 2. cluster
                #     sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
                #     # epi.pp.lazy(adata)
                #     # print(adata)
                #     # louvain算法
                #     epi.tl.louvain(adata)
                #     ari_louvain = epi.tl.ARI(adata, 'louvain', 'cell_type')
                #     print(f'louvain_ari: {ari_louvain}.\n')
                #
                #     # leiden算法
                #     # epi.pp.lazy(adata)
                #     # print(adata)
                #     epi.tl.leiden(adata)
                #     ari_leiden = epi.tl.ARI(adata, 'leiden', 'cell_type')
                #     print(f'leiden_ari: {ari_leiden}.\n')
                #
                #     # # epi.tl.kmeans(adata, num_clusters=k)
                #     # from sklearn.cluster import KMeans
                #     # kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
                #     # adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)
                #     # ari_kmeans = epi.tl.ARI(adata, 'kmeans', 'cell_type')
                #     # 结束计时
                #     end_time = time.time()
                #     # 计算运行时间
                #     elapsed_time = end_time - start_time
                #     # print(f'ari_max:{ari}, nmi:{nmi}, f1:{f1}, epoch:{epoch}, with time:{elapsed_time} s.\n')
                #     # print(f'kmeans_ari: {ari_kmeans}, with time:{elapsed_time} s.')
                #     print(f'， with time:{elapsed_time} s.')
                #
                #     if ari_louvain > ari_louvain_max:
                #         ari_louvain_max = ari_louvain
                #         ari_leiden_max = ari_leiden
                #         # ari_kmeans_max = ari_kmeans
                #         # nmi_max = nmi
                #         # f1_max = f1
                #         epoch_max = epoch
                #         if outdir:
                #             sc.settings.figdir = outdir
                #             # save = str(epoch)+'.png'
                #             save = '.png'
                #             # torch.save(self.state_dict(), os.path.join(outdir, 'model_'+str(epoch)+'.pt'))  # save model
                #             torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                #             adata.write(outdir + 'adata.h5ad', compression='gzip')
                #             # print(adata)
                #             # ----------------------------UMAP---------------------------------
                #             sc.tl.umap(adata, min_dist=0.1)
                #             # print(adata)
                #             color = [c for c in ['louvain', 'cell_type'] if c in adata.obs]
                #             sc.pl.umap(adata, color=color, save='_louvain'+save, show=False, wspace=0.4, ncols=4)
                #
                #
                #             color = [c for c in ['leiden', 'cell_type'] if c in adata.obs]
                #             sc.pl.umap(adata, color=color, save='_leiden'+save, show=False, wspace=0.4,
                #                        ncols=4)
                #
                #             # color = [c for c in ['kmeans', 'cell_type'] if c in adata.obs]
                #             # sc.pl.umap(adata, color=color, save='_kmeans'+save, show=False, wspace=0.4,
                #             #            ncols=4)
                #
                #             # ----------------------------TSNE---------------------------------
                #             sc.tl.tsne(adata, use_rep='latent')
                #             color = [c for c in ['louvain', 'cell_type'] if c in adata.obs]
                #             sc.pl.tsne(adata, color=color, save='_louvain'+save, show=False, wspace=0.4, ncols=4)
                #             color = [c for c in ['leiden', 'cell_type'] if c in adata.obs]
                #             sc.pl.tsne(adata, color=color, save='_leiden'+save, show=False, wspace=0.4,
                #                        ncols=4)
                #             # color = [c for c in ['kmeans', 'cell_type'] if c in adata.obs]
                #             # sc.pl.tsne(adata, color=color, save='_kmeans'+save, show=False, wspace=0.4,
                #             #            ncols=4)
                # print(
                    # f'ari_louvain_max:{ari_louvain_max}, then ari_leiden:{ari_leiden_max}, ari_kmeans:{ari_kmeans_max}, epoch:{epoch_max}.\n')



    def encode(self, x):
        # x:(1,1,106,106)
        # h1 = self.encoder(x).view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim)
        # return self.fc21(h1), self.fc22(h1)
        # h1 = self.encoder(x)
        # h2 = self.encoder(x)
        z, mu, logvar = self.encoder(x)
        # return h1, h2
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        attention_output, _ = self.attention_z(z, z, z)
        # 将自注意力输出与MLP的全连接层输出相加
        z = z + self.dropout(attention_output)
        #z = z + attention_output
        # z = self.layer_norm(z)
        z = self.batch_norm(z)

        recon_x = self.decoder(z)
        # recon_x = F.relu(recon_x)
        recon_x = recon_x.view(-1, 1, self.input_shape, self.input_shape)

        return recon_x

    def forward(self, x):
        # mu, logvar:(1,128)
        z, mu, logvar = self.encoder(x)
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


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
        # self.n_centroids = n_centroids
        self.input_shape = input_shape[0]

        image_channels = input_shape[2]

        self.z_dim = z_dim
        # 图像的潜在维度=输入图像的高度除以 2 的 n_levels 次幂,img_latent_dim=106/16=6
        # self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.img_latent_dim = 8

        # self.encoder = ResNet18()
        # self.encoder = CustomEncoder(num_classes=n_centroids)
        self.encoder = seresnet18(num_classes=n_centroids)


        # [x_dim, z_dim, encode_dim, decode_dim] = dims
        # self.decoder = Decoder([z_dim, decode_dim, x_dim], output_activation=nn.Sigmoid())
        # self.attention_z = nn.MultiheadAttention(embed_dim=z_dim, num_heads=5)
        # self.dropout = nn.Dropout(p=0.2)
        # self.layer_norm = nn.LayerNorm(z_dim)
        # self.batch_norm =nn.BatchNorm1d(z_dim)
        # self.fc0 = torch.nn.Linear(z_dim, 512)
        # self.attention_1 = nn.MultiheadAttention(embed_dim=512, num_heads=16)
        #
        #
        # # Assumes the input to be of shape 256x256
        # self.fc21 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        # self.fc22 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        # self.fc3 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)
        #
        # self.fc4 = torch.nn.Linear(z_dim, 128)
        # self.fc5 = torch.nn.Linear(128, 256)
        # # self.fc6 = torch.nn.Linear(256, 1*106*106)
        # # self.fc6 = torch.nn.Linear(256, 1 * 557 * 557)
        # self.fc6 = torch.nn.Linear(256, 1 * input_shape[0] * input_shape[1])

    def loss_function(self, x, label_list, k,device):
        z = self.encoder(x)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        

        # my_list = [2]
        labels = torch.tensor(label_list)  # 示例标签
        labels = labels.to(device)
        # 获取标签张量的大小和类别数量
        # batch_size = labels.size(0)
        # num_classes = k  # 假设有3个类别
        # # 创建一个大小为 [batch_size, num_classes] 的零张量，并使用scatter_函数将对应位置设为1
        # one_hot_labels = torch.zeros(batch_size, num_classes)
        # one_hot_labels = one_hot_labels.to(device)
        # one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(z, labels)
        # loss = loss + likelihood

        return loss

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1)  # NxK
        #         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = self.var_c.repeat(N, 1, 1) + 1e-8  # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(
            torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / (2 * var_c),
                                      dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi
    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init Ressac model with GMM model parameters
        """
        # n_components是GMM的成分数（即高斯分量的个数），covariance_type='diag'表示每个高斯分量的协方差矩阵是对角矩阵。
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        # 这里调用了self对象的另一个方法encodeBatch()，它接受dataloader和device作为输入，可能用于对输入数据进行编码，
        # 并返回编码后的结果z。
        z = self.encodeBatch(dataloader, device)
        # 这一步是对编码后的数据z进行GMM拟合，即使用GMM对数据进行聚类，估计各个高斯分量的参数（均值和协方差矩阵）
        gmm.fit(z)
        # 这里将拟合得到的GMM模型的均值参数（gmm.means_）转换为PyTorch张量，并将其复制到Ressac模型的参数mu_c中。
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        # 这里将拟合得到的GMM模型的协方差矩阵参数（gmm.covariances_）转换为PyTorch张量，并将其复制到Ressac模型的参数var_c中。
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
    def encodeFeature(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for (x, labels) in dataloader:
            # target_shape = (-1, 1, 106, 106)
            # target_shape = (-1, 1, 557, 557)
            target_shape = (-1, 1, self.input_shape, self.input_shape)
            # 重塑张量的形状
            x = x.view(target_shape).to(device)
            x = x.to(torch.float)
            # 使用模型的编码器(self.encoder)将输入数据 x
            # 编码为潜在向量 z，以及潜在向量 z 对应的均值 mu 和对数方差 logvar。
            z = self.encoder(x)
            # # z, mu, logvar = self.encoder(x)
            # # 如果 out 参数是 'z'，表示需要返回潜在向量 z，则将 z 添加到 output 列表中，
            # # 同时使用 detach().cpu() 将结果从计算设备中分离，并转移到 CPU 上。
            # if out == 'z':
            output.append(z.detach().cpu())
            # # 如果 out 参数是 'x'，表示需要返回解码后的数据 x，则使用模型的解码器(self.decoder)对潜在向量 z 进行解码，
            # # 并将解码结果添加到 output 列表中。
            # elif out == 'x':
            #     recon_x = self.decoder(z)
            #     output.append(recon_x.detach().cpu().data)
            # # 如果 out 参数是 'logit'，表示需要返回聚类分配，这里使用了一个 self.get_gamma(z)[0] 方法，
            # # 该方法似乎与聚类相关，返回一个与 z 相关的聚类分配。将聚类分配结果添加到 output 列表中。
            # elif out == 'logit':
            #     output.append(self.get_gamma(z)[0].cpu().detach().data)
        # 将 output 列表中的所有结果拼接成一个大的张量，并将其转换为 NumPy 数组形式。
        output = torch.cat(output).numpy()

        return output

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
        # n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = 1501
        # early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        # ari_louvain_max = 0
        # ari_leiden_max = 0
        # ari_kmeans_max = 0
        # epoch_max = 0
        # nmi_max = 0
        # f1_max = 0
        accuracy_max = 0
        ccc_value_max = 0
        epoch_max = 0
        loss_test = 1
        output_list=[]
        labels_list=[]
        # # 定义类标签映射
        # label_mapping = {'EX3': 0, 'OC': 1, 'AC': 2, 'EX2': 3, 'IN1': 4, 'IN2': 5, 'MG': 6, 'EX1': 7}
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                # epoch_recon_loss, epoch_kl_loss = 0, 0
                # running_loss = 0

                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                # 遍历数据加载器的每个批次
                for i, (x, labels) in tk0:
                    # print(labels)
                    # 将类标签转换为对应的整数形式
                    # label_indices = [label_mapping[label] for label in labels]
                    target_shape = (-1, 1, self.input_shape, self.input_shape)
                    # 重塑张量的形状
                    x = x.view(target_shape)
                    x = x.to(torch.float)
                    x = x.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备
                    optimizer.zero_grad()
                    loss = self.loss_function(x, labels, k, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()
                    tk0.set_postfix_str('loss={:.3f}'.format(
                        loss))
                    tk0.update(1)
                    iteration += 1
                # ------------------------test----------------------
                print(f"\nepoch_now:{epoch},epoch_max:{epoch_max},ccc_value_max: {ccc_value_max:.4f}")
                if epoch % 5 == 0:
                    print("\ntest begain...")
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in dataloader_test:
                            labels = labels.to(device)
                            # 获取标签张量的大小和类别数量
                            target_shape = (-1, 1, self.input_shape, self.input_shape)
                            # 重塑张量的形状
                            inputs = inputs.view(target_shape)
                            inputs = inputs.to(torch.float)
                            inputs = inputs.to(device)  # 将输入数据移到GPU上，其中device是你的GPU设备

                            outputs = self.encoder(inputs)
                            break
                        #     output_list.append(outputs)
                        #     labels_list.append(labels)
                        #
                        # # 在计算 ccc_value 之前将张量移到 CPU 上并打平元素
                        # output_list_cpu = np.concatenate([tensor.cpu().numpy().ravel() for tensor in output_list])
                        # labels_list_cpu = np.concatenate([tensor.cpu().numpy().ravel() for tensor in labels_list])

                        # ccc_value = lins_ccc(output_list_cpu, labels_list_cpu)

                        # labels = torch.tensor(label_list)  # 示例标签
                        # labels = labels.to(device)
                        criterion = nn.MSELoss()
                        loss_test = criterion(outputs, labels)
                        ccc_value = lins_ccc(outputs, labels)
                        print(f"\nloss_test:{loss_test:.4f}")
                        print(f"\ntest_ccc_value:{ccc_value:.4f}")
                        if ccc_value > ccc_value_max:
                            ccc_value_max = ccc_value
                            epoch_max = epoch
                            # accuracy = correct / total
                            print(f"\nepoch:{epoch_max},ccc_value_max: {ccc_value_max:.4f}")
                            if outdir:
                                sc.settings.figdir = outdir
                                # save = str(epoch)+'.png'
                                # save = '.png'
                                # torch.save(self.state_dict(), os.path.join(outdir, 'model_'+str(epoch)+'.pt'))  # save model
                                torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model

                        if ccc_value_max == 1:
                            print(f"\nearly stop........\nepoch:{epoch_max},ccc_value_max: {ccc_value_max:.4f}")
                            break
            if ccc_value_max == 0:
                if outdir:
                    sc.settings.figdir = outdir
                    torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                print(f"\nsave at epoch:{epoch}")





    def encode(self, x):
        # x:(1,1,106,106)
        # h1 = self.encoder(x).view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim)
        # return self.fc21(h1), self.fc22(h1)
        # h1 = self.encoder(x)
        # h2 = self.encoder(x)
        z, mu, logvar = self.encoder(x)
        # return h1, h2
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        attention_output, _ = self.attention_z(z, z, z)
        # 将自注意力输出与MLP的全连接层输出相加
        z = z + self.dropout(attention_output)
        #z = z + attention_output
        # z = self.layer_norm(z)
        z = self.batch_norm(z)

        recon_x = self.decoder(z)
        # recon_x = F.relu(recon_x)
        recon_x = recon_x.view(-1, 1, self.input_shape, self.input_shape)

        return recon_x

    def forward(self, x):
        # mu, logvar:(1,128)
        z, mu, logvar = self.encoder(x)
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x in dataloader:
            # target_shape = (-1, 1, 106, 106)
            # target_shape = (-1, 1, 557, 557)
            target_shape = (-1, 1, self.input_shape, self.input_shape)
            # 重塑张量的形状
            x = x.view(target_shape).to(device)
            x = x.to(torch.float)
            # 使用模型的编码器(self.encoder)将输入数据 x
            # 编码为潜在向量 z，以及潜在向量 z 对应的均值 mu 和对数方差 logvar。
            # z = self.encoder(x)
            z, mu, logvar = self.encoder(x)
            # 如果 out 参数是 'z'，表示需要返回潜在向量 z，则将 z 添加到 output 列表中，
            # 同时使用 detach().cpu() 将结果从计算设备中分离，并转移到 CPU 上。
            if out == 'z':
                output.append(z.detach().cpu())
            # 如果 out 参数是 'x'，表示需要返回解码后的数据 x，则使用模型的解码器(self.decoder)对潜在向量 z 进行解码，
            # 并将解码结果添加到 output 列表中。
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            # 如果 out 参数是 'logit'，表示需要返回聚类分配，这里使用了一个 self.get_gamma(z)[0] 方法，
            # 该方法似乎与聚类相关，返回一个与 z 相关的聚类分配。将聚类分配结果添加到 output 列表中。
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)
        # 将 output 列表中的所有结果拼接成一个大的张量，并将其转换为 NumPy 数组形式。
        output = torch.cat(output).numpy()

        return output
class VectorQuantizer(torch.nn.Module):
    """
    Implementation of VectorQuantizer Layer from: simplegan.autoencoder.vq_vae
    url: https://simplegan.readthedocs.io/en/latest/_modules/simplegan/autoencoder/vq_vae.html
    """
    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        self.embedding = torch.nn.parameter.Parameter(torch.tensor(
            torch.randn(self.embedding_dim, self.num_embeddings)),
            requires_grad=True)

    def forward(self, x):

        flat_x = x.view([-1, self.embedding_dim])

        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_x, self.embedding)
            + torch.sum(self.embedding ** 2, dim=0, keepdim=True)
        )

        encoding_indices = torch.argmax(-distances, dim=1)
        encodings = (torch.eye(self.num_embeddings)[encoding_indices]).to(x.device)
        encoding_indices = torch.reshape(encoding_indices, x.shape[:1] + x.shape[2:])
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, x.shape)

        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return loss, quantized, perplexity, encoding_indices

    def quantize_encoding(self, x):
        encoding_indices = torch.flatten(x)
        encodings = torch.eye(self.num_embeddings)[encoding_indices]
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, torch.Size([-1, self.embedding_dim]) + x.shape[1:])
        return quantized



# if __name__ == '__main__':
#
#     encoder = ResNetEncoder()
#     decoder = ResNetDecoder()
#
#     test_input = torch.rand(10, 3, 256, 256)
#     out = encoder(test_input)
#
#     test_input = torch.rand(10, 10, 16, 16)
#     out = decoder(test_input)
