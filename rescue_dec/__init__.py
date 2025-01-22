from .dataset import *
import torch
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
sc.settings.autoshow = False
from anndata import AnnData
from typing import Union, List
from .ResNetAE_pytorch import ResNet_pred
# from labels_statistic import *
import math


def some_function(
        data_list: Union[str, List],
        test_list: Union[str, List],
        model_path: Union[str, List],
        # n_centroids:int = 30,
        outdir: bool = None,
        verbose: bool = False,
        pretrain: str = None,
        lr: float = 0.0002,
        batch_size: int = 64,
        gpu: int = 0,
        seed: int = 18,
) -> AnnData:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # cuda device
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    print("\n**********************************************************************")
    print(
        " Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome.")
    print("**********************************************************************\n")
    n_centroids = 0
    if not pretrain:
        adata, trainloader, testloader, k = load_dataset_train(
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
    print("\n======== Parameters ========")
    print(
        'Cell number: {}\nPeak number: {}\ndevice: {}\nlr: {}\nbatch_size: {}\nn_celltypess: {}'.format(
            cell_num, input_dim, device, lr, batch_size, k))
    print("============================")
    model = ResNet_pred(input_shape=(s, s, 1), n_centroids=k, ).to(device)
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
        model_path = model_path
        print('\n## Loading Model: {}\n'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        output_pre = []
        with torch.no_grad():
            for inputs, labels in testloader:
                target_shape = (-1, 1, model.input_shape, model.input_shape)
                inputs = inputs.view(target_shape)
                inputs = inputs.to(torch.float)
                inputs = inputs.to(device)
                outputs = model.pre_resnet(inputs)
                output_pre.append(outputs.cpu().numpy())
        output_pre = np.concatenate(output_pre, axis=0)
        column_indices = adata.obs.columns[:k]
        row_indices = adata.obs.index
        output_df = pd.DataFrame(output_pre, index=row_indices, columns=column_indices)
        output_df.to_csv('pre/model_outputs.txt', sep='\t', index=True, header=True)
    print("over....")
