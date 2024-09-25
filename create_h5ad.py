import os
from anndata import AnnData
import scanpy as sc
import pandas as pd
import scipy
from scipy.sparse import issparse
from glob import glob
from marge_h5ad import getFractions
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser("~")+'/.rescue/'


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






if __name__ == "__main__":
    # file_path = 'E-MTAB/10000/HP1507101/'
    # in_Path = file_path + 'RNA_data_7_10000_HP1507101.txt'
    # file_path = 'GSE257541/10000/pbmc0h/'
    # in_Path = file_path + 'RNA_data_new_0h_2.txt'
    # new_h5ad = load_file(in_Path)
    # print("over..........")
    #
    # # file_path = 'D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/scRNA/HP1508501T2D/'
    # # celltype_Path = file_path + 'data/HP1507101_scData_10000_celltypes.txt'
    # celltype_Path = file_path + 'data/matrix_0h_celltypes.txt'
    #
    # # 读取txt文件，假设文件是以tab分隔
    # df = pd.read_csv(celltype_Path, sep='\t')
    # # 统计‘Celltype’列中每个类别的数量
    # celltype_counts = df['Celltype'].value_counts()
    # # 计算每个类别的比例
    # celltype_proportions = celltype_counts / celltype_counts.sum()
    # # 将类别名称和比例添加到 AnnData 对象的 obs 属性中
    # for celltype, proportion in celltype_proportions.items():
    #     # 为每个类别创建 obs 列，列名为类别名，值为该类别的比例
    #     new_h5ad.obs[celltype] = proportion
    #
    # RNA_h5ad_path = file_path + 'RNA_data_new.h5ad'
    # # 保存为新的 h5ad 文件
    # new_h5ad.write(RNA_h5ad_path, compression='gzip')
    # # 输出获取占比和矩阵数据
    # getFractions(RNA_h5ad_path, file_path)
    # print(f"结果已保存")

    # -----------------------------------------绘制散点图-----------------------------


    # file = 'E-MTAB/10000/all_7/pred'
    file = 'GSE257541/10000/pbmc48h/4000/pred'

    file1_path = file + '/truth_data.txt'
    file2_path = file + '/out_predict_scaden.txt'
    tit = 'Scaden'
    # file2_path = file + '/prediction_res_4000.txt'
    # tit = 'Rescue'

    # 读取txt1和txt2文件的第一行数据
    txt1_df = pd.read_csv(file1_path, sep='\t', index_col=0)
    txt2_df = pd.read_csv(file2_path, sep='\t', index_col=0)
    # 获取第一行数据
    txt1_first_row = txt1_df.iloc[0]
    txt2_first_row = txt2_df.iloc[0]


    # 设置字体大小
    plt.figure(figsize=(9, 7))  # 调整图形尺寸
    # 绘制散点图，并调整点的大小
    plt.scatter(txt1_first_row, txt2_first_row, s=100)  # s参数设置点的大小

    # 在每个点上标注列名
    for i, txt in enumerate(txt1_first_row.index):
        plt.text(txt1_first_row[i], txt2_first_row[i], txt, fontsize=15, ha='right', va='bottom')
    # 设置图形标题和标签的字体大小

    # 绘制从零点 (0,0) 开始的对角线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)

    plt.title(tit, fontsize=24)
    plt.xlabel('Truth', fontsize=22)
    plt.ylabel('Predict', fontsize=22)
    # 设置刻度标签的字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # # 固定x轴和y轴的范围为0到1
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)

    # 保存图形为png文件
    output_path = file + '/' + tit + '_scatter_plot_2.png'
    plt.savefig(output_path, format='png')

    # 显示图形
    plt.show()
    print('Plot saved as', output_path)
    print('over............')



