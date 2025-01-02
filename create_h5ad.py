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

# 自定义排序键函数
def sort_key(s):
    import re
    # 分离字符串中的字母部分和数字部分
    match = re.match(r"([a-zA-Z]+)([0-9]+)", s)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return (letter_part, number_part)
    return (s, 0)




if __name__ == "__main__":
    data_name = 'HP1526901T2D'
    file_path = 'E-MTAB_1/10000/' + data_name + '/'
    in_Path = file_path + 'RNA_data_7_10000_'+data_name + '.txt'
    in_Path_1 = file_path + data_name + '_RNAdata.txt'
    in_Path_2 = 'E-MTAB_1/10000/data_matrix_with_var_names_1000.txt'
    in_Path_3 = file_path + data_name + '_RNAdata_3.txt'


    # 读取 TXT 文件，第一行和第一列作为索引
    df = pd.read_csv(in_Path, sep='\t', index_col=0)
    # 对行索引进行排序
    df = df.sort_index()
    # 输出到新的文件
    df.to_csv(in_Path_1, sep='\t')
    # -------------------------------------
    # 读取两个文件，第一列作为行索引
    df1 = pd.read_csv(in_Path_1, sep='\t', index_col=0)
    df2 = pd.read_csv(in_Path_2, sep='\t', index_col=0)

    # 检查行索引顺序是否一致
    if df1.index.equals(df2.index):
        print("两个文件的行索引顺序完全一致。")
    else:
        print("两个文件的行索引顺序不一致。")
        # 找出不一致的行名
        index_diff_1 = df1.index.difference(df2.index)
        index_diff_2 = df2.index.difference(df1.index)
        if not index_diff_1.empty:
            print(f"在file1中有但不在file2中的行索引: {index_diff_1.tolist()}")
        if not index_diff_2.empty:
            print(f"在file2中有但不在file1中的行索引: {index_diff_2.tolist()}")
        # 如果两者行名相同但顺序不同
        if set(df1.index) == set(df2.index) and not df1.index.equals(df2.index):
            print("两个文件的行索引相同，但顺序不同。")
    # -------------------------------------
    # 确保两个文件行数相同
    if len(df1) != len(df2):
        raise ValueError("两个文件的行数不一致，无法替换索引列。")
    # 用file2的行索引替换file1的行索引
    df1.index = df2.index
    # 输出替换后的文件
    df1.to_csv(in_Path_3, sep='\t')
    print("已使用file2的索引列替换file1的索引列，并输出到'output.txt'文件中。")
    # ------------------------------------
    in_Path = in_Path_3
    new_h5ad = load_file(in_Path)
    print("over..........")

    # file_path = 'D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/scRNA/HP1508501T2D/'
    # celltype_Path = file_path + 'data/HP1507101_scData_10000_celltypes.txt'
    # celltype_Path = file_path + 'data/matrix_0h_celltypes.txt'
    # celltype_Path = file_path + 'data/'+data_name+'_celltypes.txt'
    celltype_Path = file_path + 'data/' + data_name + '_scData_10000_celltypes.txt'

    # 读取txt文件，假设文件是以tab分隔
    df = pd.read_csv(celltype_Path, sep='\t')
    # 统计‘Celltype’列中每个类别的数量
    celltype_counts = df['Celltype'].value_counts()
    # 计算每个类别的比例
    celltype_proportions = celltype_counts / celltype_counts.sum()
    # 按照细胞类型名称字母a-z，数字递增顺序排列
    # sorted_celltype_proportions = celltype_proportions.sort_index()
    # 根据自定义排序键函数进行排序
    celltype_proportions_sorted = celltype_proportions.sort_index(key=lambda x: x.map(sort_key))

    type_name=[]
    # 将类别名称和比例添加到 AnnData 对象的 obs 属性中
    for celltype, proportion in celltype_proportions_sorted.items():
        # 为每个类别创建 obs 列，列名为类别名，值为该类别的比例
        new_h5ad.obs[celltype] = proportion
        type_name.append(celltype)
    new_h5ad.uns['cell_types'] = type_name

    RNA_h5ad_path = file_path + 'RNA_data_new_1.h5ad'
    # 保存为新的 h5ad 文件
    new_h5ad.write(RNA_h5ad_path, compression='gzip')
    # 输出获取占比和矩阵数据
    getFractions(RNA_h5ad_path, file_path)
    print(f"结果已保存")

    # # -----------------------------------------绘制散点图-----------------------------
    # # file = 'D:/cooperation/dataset/pbmc_4/marge_data_6_8_A_C_sdy67/pred'
    # # file = 'E-MTAB/10000/HP1526901T2D/4000/pred'
    # # file = 'GSE257541/10000/pbmc48h/4000/pred'
    # file = 'E-MTAB/10000/all/test/pred'
    #
    # file1_path = file + '/truth_data.txt'
    # # file2_path = file + '/out_predict_scaden.txt'
    # # tit = 'Scaden'
    # file2_path = file + '/prediction_res_4000.txt'
    # # file2_path = file + '/prediction_res.txt'
    # tit = 'Rescue'
    #
    # # 读取txt1和txt2文件的第一行数据
    # txt1_df = pd.read_csv(file1_path, sep='\t', index_col=0)
    # txt2_df = pd.read_csv(file2_path, sep='\t', index_col=0)
    # # 获取第一行数据
    # txt1_first_row = txt1_df.iloc[0]
    # txt2_first_row = txt2_df.iloc[0]
    # # 设置字体大小
    # plt.figure(figsize=(9, 7))  # 调整图形尺寸
    # # 绘制散点图，并调整点的大小
    # plt.scatter(txt1_first_row, txt2_first_row, s=100)  # s参数设置点的大小
    # # 在每个点上标注列名
    # for i, txt in enumerate(txt1_first_row.index):
    #     plt.text(txt1_first_row[i], txt2_first_row[i], txt, fontsize=15, ha='right', va='bottom')
    # # 设置图形标题和标签的字体大小
    #
    # # 绘制从零点 (0,0) 开始的对角线
    # # plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    # plt.title(tit, fontsize=24)
    # plt.xlabel('Truth', fontsize=22)
    # plt.ylabel('Predict', fontsize=22)
    # # 设置刻度标签的字体大小
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # # # 固定x轴和y轴的范围为0到1
    # # plt.xlim(0, 1)
    # # plt.ylim(0, 1)
    # # 保存图形为png文件
    # output_path = file + '/' + tit + '_scatter_plot.png'
    # plt.savefig(output_path, format='png')
    #
    # # 显示图形
    # plt.show()
    # print('Plot saved as', output_path)

    print('over............')



