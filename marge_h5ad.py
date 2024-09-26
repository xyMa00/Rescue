import anndata as ad
import scanpy as sc
import pandas as pd

import numpy as np



def getFractions(in_path, out_path):
    adata_combined_test = sc.read_h5ad(in_path)
    data_matrix = adata_combined_test.X.T
    var_names = adata_combined_test.var_names
    # 检查数据矩阵是否为稀疏矩阵，如果是，则转换为密集矩阵
    if hasattr(data_matrix, "toarray"):
        data_matrix = data_matrix.toarray()
    # 将数据矩阵转换为pandas数据框，并设置var_names作为行索引
    df = pd.DataFrame(data_matrix, index=var_names)
    # 保存数据框到txt文件，使用制表符作为分隔符
    df.to_csv(out_path + "/data_matrix_with_var_names.txt", sep='\t', index=True, header=True)
    df_obs = adata_combined_test.obs

    # 保存obs数据框到txt文件，使用制表符作为分隔符
    df_obs.to_csv(out_path + "/obs_data.txt", sep='\t', index=True)

def select_num_features(in_Path, out_path):
    adata = sc.read_h5ad(in_Path)
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
    num_features_to_keep = s * s
    print(num_features_to_keep)

    # 按照顺序选择要保留的特征
    indices_to_keep = np.arange(num_features_to_keep)
    # 使用索引数组来选择要保留的特征
    adata = adata[:, indices_to_keep]
    print(adata)

    # out_path = 'hscr/all_2_seurat_object_common_8/20000/train_dataset/scaden_1500'
    adata_filtered = out_path + '_' + str(num_features_to_keep) + '.h5ad'
    adata.write_h5ad(adata_filtered, compression='gzip')
    print(f'AnnData 已保存到 {adata_filtered}')
    return adata_filtered


if __name__ == "__main__":
    # file1 = 'hscr/all_2_seurat_object_common_8/20000/Z_S_HSCR2_dilated_2500_20000_train_2500.h5ad'
    # file2 = 'hscr/all_2_seurat_object_common_8/20000/Z_S_HSCR3_dilated_2500_20000_train_2500.h5ad'

    # file1 = 'LUSC_10_types/marge_4000/LUSC_IA3_BC3_10_2000_17956.h5ad'
    # file2 = 'LUSC_10_types/marge_4000/LUSC_IA3_BC8_10_2000_17956.h5ad'
    #
    # # 读取两个 .h5ad 文件
    # adata1 = ad.read_h5ad(file1)
    # adata2 = ad.read_h5ad(file2)
    #
    # # 合并两个 AnnData 对象
    # adata_combined_train = ad.concat([adata1, adata2], join='inner')
    #
    # # 手动合并 .uns 字典
    # adata_combined_train.uns = {**adata1.uns, **adata2.uns}
    #
    # # 保存为新的 .h5ad 文件
    # adata_combined_train.write('LUSC_10_types/marge_4000/' + 'combined_train.h5ad', compression='gzip')

    # file3 = 'hscr/all_2_seurat_object_common_8/20000/Z_S_HSCR2_dilated_2500_20000_pre_2500.h5ad'
    # file4 = 'hscr/all_2_seurat_object_common_8/20000/Z_S_HSCR3_dilated_2500_20000_pre_2500.h5ad'

    # file3 = 'LUSC_10_types/marge_4000/LUSC_IA3_BC3_10_1000_17956.h5ad'
    # file4 = 'LUSC_10_types/marge_4000/LUSC_IA3_BC8_10_1000_17956.h5ad'
    #
    # # 读取两个 .h5ad 文件
    # adata3 = ad.read_h5ad(file3)
    # adata4 = ad.read_h5ad(file4)
    #
    # # 合并两个 AnnData 对象
    # adata_combined_test = ad.concat([adata3, adata4], join='inner')
    #
    # # 手动合并 .uns 字典
    # adata_combined_test.uns = {**adata3.uns, **adata4.uns}
    #
    # # 保存为新的 .h5ad 文件
    # adata_combined_test.write('LUSC_10_types/marge_4000/' + 'combined_test.h5ad', compression='gzip')


    # ----------------获取占比及表达矩阵------------------------
    # in_path = 'hscr/all_2_seurat_object_common_8/20000//combined_test.h5ad'
    # out_path = 'hscr/all_2_seurat_object_common_8/20000/'
    # in_path = 'LUSC_10_types/marge_20000/train_dataset/scaden_1500/adata_LUSC_10_test_17956_1875_375.h5ad'
    # out_path = 'LUSC_10_types/marge_20000/train_dataset/scaden_1500'

    # in_path = 'hscr/all_2_seurat_object_common_8/20000/train_dataset/scaden_1500/adata_HSCR_8_train_2500_1875_1500.h5ad'
    # out_path = 'hscr/all_2_seurat_object_common_8/20000/train_dataset/scaden_1500'
    # in_path = 'LUSC_IA3_BC3_10/rare/LUSC_IA3_BC3_10_test_percent50_2000_17956.h5ad'
    # out_path = 'LUSC_IA3_BC3_10/rare'
    # in_path = 'LUSC_IA3_BC3_10/rare/200/LUSC_IA3_BC3_10_train_percent10_200_17956.h5ad'
    # out_path = 'LUSC_IA3_BC3_10/rare/200'

    # in_path = 'LUSC_IA3_BC3_10/property/200/LUSC_IA3_BC3_10_test_all_50_17956_new.h5ad'
    # out_path = 'LUSC_IA3_BC3_10/property/200'
    # # in_path = 'LUSC_IA3_BC3_10/rare/4000/LUSC_IA3_BC3_10_train_percent10_4000_17956.h5ad'
    # # out_path = 'LUSC_IA3_BC3_10/rare/4000'
    # getFractions(in_path, out_path)

    # 处理峰值为可开方的最大数
    # out_path = 'LUSC_IA3_BC3_10/property/LUSC_IA3_BC3_10_test_all_1000'
    # in_Path = out_path+'.h5ad'
    # # in_Path = 'LUSC_IA3_BC3_10/rare/LUSC_IA3_BC3_10_test_percent50_2000.h5ad'
    # # out_path = 'LUSC_IA3_BC3_10/rare'
    # select_num_features(in_Path, out_path)
    # print("over..........")

    # ------------使第二个h5ad文件的obs列顺序使其与第一个文件相同---------------------
    # file1_path='LUSC_IA3_BC3_10/20000/need/'
    # file1 = file1_path+'adata_LUSC_IA3_BC3_10_pre_17956.h5ad'
    # file2_path = 'LUSC_IA3_BC3_10/property/4000/'
    # file2 = file2_path+'LUSC_IA3_BC3_10_4000_17956.h5ad'
    #
    # # 读取第一个h5ad文件
    # adata1 = sc.read_h5ad(file1)
    # # 读取第二个h5ad文件
    # adata2 = sc.read_h5ad(file2)
    # # 调整第二个文件的obs列顺序使其与第一个文件相同
    # common_columns = adata1.obs.columns.intersection(adata2.obs.columns)
    # adata2.obs = adata2.obs[common_columns]
    # # 保存第二个文件为一个新的h5ad文件
    # adata2.write_h5ad(file2_path+'file2_new.h5ad', compression='gzip')
    # # -------------------------------------------------------------------------
    # # 处理峰值为可开方的最大数
    #
    # out_path = 'LUSC_IA3_BC3_10/property/4000'
    # file_name = '/LUSC_IA3_BC3_10_4000_17956_new'
    # # out_path = 'E-MTAB/10000/HP1525301T2D/'
    # # file_name = '/RNA_data_new'
    # out_path1 = out_path + file_name
    # in_Path = out_path1 + '.h5ad'
    # new_file= 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/test_2500_20000_4000.h5ad'
    # out_path = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/'
    new_file= 'LUSC_IA3_BC3_10/2500/LUSC_IA3_BC3_10_2500.h5ad'
    out_path = 'LUSC_IA3_BC3_10/2500/'
    # new_file = select_num_features(in_Path, out_path1)
    # 输出获取占比和矩阵数据
    getFractions(new_file, out_path)
    print("over..........")




