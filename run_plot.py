import anndata as ad
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

from Similarity_eval import *

def lins_ccc(x, y):
    """
    计算 Lin's Concordance Correlation Coefficient (CCC)
    参数:
    x: 第一组数据 (numpy array)
    y: 第二组数据 (numpy array)
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))
    pearson_corr = stats.pearsonr(x, y)[0]
    ccc = (2 * pearson_corr * np.sqrt(x_var) * np.sqrt(y_var)) / (x_var + y_var + (x_mean - y_mean) ** 2)
    return ccc

# 从txt文件中获取第2到第8列的数据
def get_data(file_path, lie):
    df = pd.read_csv(file_path, sep='\t')
    return df.iloc[:, 1:lie].values
    # return df.iloc[:, 1:5].values
    # return df.iloc[:, 1:14].values
    # return df.iloc[:, 1:23].values


def simil(file1_path, file2_path, col):
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep='\t')

    # 提取第2到第8列
    # columns_to_use = df1.columns[1:5]
    # columns_to_use = df1.columns[1:7]
    columns_to_use = df1.columns[1:col]

    data1 = df1[columns_to_use]
    data2 = df2[columns_to_use]

    # 将所有列的数据拼接成一个长向量
    long_vector1 = data1.to_numpy().flatten()
    long_vector2 = data2.to_numpy().flatten()

    # 计算similarity
    Pearson_val_all = PearsonCorrelationSimilarity(long_vector1, long_vector2)
    print(f"Pearson_val_all: {Pearson_val_all}")

    Dice_val_all = dice_coefficient(long_vector1.tolist(), long_vector2.tolist())
    print(f"Dice_val_all: {Dice_val_all}")

    # 计算余弦相似度
    cosine_val_all = python_cos(long_vector1.tolist(), long_vector2.tolist())
    print(f"cosine_val_all: {cosine_val_all}")

    fidelity_val_all = fidelity_similarity(long_vector1.tolist(), long_vector2.tolist())
    print(f"fidelity_val_all: {fidelity_val_all}")

    # euclidean_distance_val_all = euclidean_distance_similarity(long_vector1.tolist(), long_vector2.tolist())
    # print(f"euclidean_distance_val_all: {euclidean_distance_val_all}")

    # squared_val_all = squared_chord_similarity(long_vector1.tolist(), long_vector2.tolist())
    # print(f"squared_chord_val_all: {squared_val_all}")

    # jaccard_val_all = jaccard_similarity(long_vector1.tolist(), long_vector2.tolist())
    # print(f"jaccard_val_all: {jaccard_val_all}")

    return Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all


if __name__ == "__main__":
    # file = 'LUSC_IA3_BC3_10/20000/need/train_dataset'

    # file = 'hscr/all_2_seurat_object_common_8/20000/train_dataset/pred'

    # file = 'LUSC_IA3_BC3_10/rare/pred'

    # file = 'LUSC_IA3_BC3_10/20000/need/train_dataset/pred/pre_LUSC_IA3_BC8_10_types_17956'
    # file = 'LUSC_IA3_BC3_10/property/pred'
    # file = 'LUSC_IA3_BC3_10/2500/4000/pred'
    # file = 'LUSC_IA3_BC3_10/20000/pred'

    # file = 'LUSC_IA3_BC8_10/property/pred'
    # file = 'LUSC_IA3_BC8_10/property/others_pred_LUSC_IA3_BC8_10'

    # file = 'HP1504101T2D/real/pred'
    # file = 'HP1508501T2D/real/pred'
    # file = 'E-MTAB/10000/HP1507101/4000/pred'
    # file = 'E-MTAB/10000/all_7/pred'
    # file = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/train_dataset/pred'
    # file = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/need/pred'
    # filename = 'HSCR'

    file = 'LUSC_IA3_BC3_10/2500/pred'
    # file = 'LUSC_IA3_BC8/pred'
    # filename = 'LUSC'

    # file = 'mouse/4000/pred'
    # filename = 'Mouse'
    # file = 'GSE257541/10000/pbmc48h/4000/pred'

    # file = 'LUSC_10_types/marge_20000/train_dataset/pred'
    # file = 'seurat_pbmc/4000/pred'
    # file = 'seurat_pbmc/20000/train_dataset/pred'
    file1_path = file + '/truth_data.txt'
    # file1_path = file + '/truth_data_7.txt'
    # file1_path = file + '/truth_data_7_all.txt'
    # file1_path = file + '/truth_data_new.txt'
    # file2_path = file+'/out_predict_scaden.txt'
    # file2_path = file + '/out_predict_scaden_7.txt'
    # file2_path = file + '/out_predict_scaden_7_all.txt'
    # file2_path = file + '/prediction_CSx.txt'
    # file2_path = file + '/prediction_cmp.txt'
    # file2_path = file+'/prediction_music.txt'
    # file2_path = file + '/prediction_nnls.txt'
    # file2_path = file+'/prediction_res_8000.txt'
    # file2_path = file+'/prediction_res_6400.txt'
    # file2_path = file+'/prediction_res_4800.txt'
    file2_path = file + '/prediction_res_4000.txt'
    # file2_path = file + '/prediction_res_4000_7.txt'
    # file2_path = file + '/prediction_res_4000_7_all.txt'
    # file2_path = file + '/prediction_res_4000_new.txt'
    # file2_path = file+'/prediction_res_3200.txt'
    # file2_path = file+'/prediction_res_1600.txt'
    # file2_path = file+'/prediction_res_800.txt'
    # file2_path = file+'/prediction_res_400.txt'
    # file2_path = file + '/prediction_res_200.txt'
    # file2_path = file + '/prediction_res_200_1.txt'
    # file2_path = file + '/prediction_res_200_2.txt'
    # filename = 'PBMC'
    filename = 'Rescue'
    # filename = 'Scaden'

    # # # 读取文件，假设文件使用制表符作为分隔符
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep='\t')
    # #
    # # # 提取第2到第8列
    # columns_to_use = df1.columns[1:5]
    # columns_to_use = df1.columns[1:7]
    # columns_to_use = df1.columns[1:14]
    # columns_to_use = df1.columns[1:23]
    # columns_to_use = df1.columns[1:10]
    columns_to_use = df1.columns[1:11]
    # columns_to_use = df1.columns[1:9]
    # columns_to_use = df1.columns[8:9]
    # columns_to_use = df1.columns[1:5]

    data1 = df1[columns_to_use]
    data2 = df2[columns_to_use]

    # 将所有列的数据拼接成一个长向量
    long_vector1 = data1.to_numpy().flatten()
    long_vector2 = data2.to_numpy().flatten()
    # 计算总体的 Lin's CCC
    overall_ccc = lins_ccc(long_vector1, long_vector2)
    # print(f"Overall Lin's CCC: {overall_ccc}")
    # # 计算总体的 RMSE
    rmse_value_all = np.sqrt(mean_squared_error(long_vector1, long_vector2))
    print(f"Overall rmse: {rmse_value_all}")
    #
    Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all = simil(file1_path, file2_path, 5)

    # # 计算similarity
    # Pearson_val_all = PearsonCorrelationSimilarity(long_vector1, long_vector2)
    # print(f"Pearson_val_all: {Pearson_val_all}")
    #
    # Dice_val_all = dice_coefficient(long_vector1.tolist(), long_vector2.tolist())
    # print(f"Dice_val_all: {Dice_val_all}")
    #
    # # 计算余弦相似度
    # cosine_val_all = python_cos(long_vector1.tolist(), long_vector2.tolist())
    # print(f"cosine_val_all: {cosine_val_all}")
    #
    # fidelity_val_all = fidelity_similarity(long_vector1.tolist(), long_vector2.tolist())
    # print(f"fidelity_val_all: {fidelity_val_all}")
    #
    # # euclidean_distance_val_all = euclidean_distance_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"euclidean_distance_val_all: {euclidean_distance_val_all}")
    #
    # # squared_val_all = squared_chord_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"squared_chord_val_all: {squared_val_all}")
    #
    # jaccard_val_all = jaccard_similarity(long_vector1.tolist(), long_vector2.tolist())
    # print(f"jaccard_val_all: {jaccard_val_all}")
    #
    #
    # # 计算每列对应数据的 Lin's CCC
    # Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all, jaccard_val_all = simil(long_vector1, long_vector2)
    #
    # ccc_values = {}
    # for col in columns_to_use:
    #     ccc_values[col] = lins_ccc(data1[col].values, data2[col].values)
    #
    # # 输出结果
    # for col, ccc in ccc_values.items():
    #     print(f"Lin's CCC for column {col}: {ccc}")
    #
    # # # ------------------------------------------------------------
    # 计算每列对应数据的 Lin's CCC 并绘制散点图
    # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    # # fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    # # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    #
    # # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    #
    # for i, col in enumerate(columns_to_use):
    #     x = data1[col].values
    #     y = data2[col].values
    #     ccc = lins_ccc(x, y)
    #
    #     # ax = axes[i // 4, i % 4]
    #     # ax = axes[ i % 6]
    #     ax = axes[i // 3, i % 3]
    #     # ax = axes[i % 4]
    #     # ax = axes[i % 2]
    #     ax.scatter(x, y, alpha=0.5)
    #     ax.set_title(f"{col}\nLin's CCC: {ccc:.2f}")
    #     ax.set_xlabel('Truth')
    #     ax.set_ylabel('Prediction')
    #
    # # # 隐藏多余的子图
    # # for j in range(i + 1, 8):
    # #     fig.delaxes(axes[j // 4, j % 4])
    #
    # plt.tight_layout()
    # plt.savefig(file+'/plot_results/scatter_plots.png')
    # plt.show()
    # # # # # ------------------------------------------------------------
    # 定义颜色列表
    # colors = ['orange', 'cyan', 'red', 'purple', 'HotPink', 'MediumOrchid', 'blue', 'green', 'Tomato']
    # sim_val=[]
    # for i, col in enumerate(columns_to_use):
    #     x = data1[col].values
    #     y = data2[col].values
    #     # ccc = lins_ccc(x, y)
    #     # ccc = dice_coefficient(x, y)
    #     ccc = fidelity_similarity(x, y)
    #     # # # 计算余弦相似度
    #     # ccc = python_cos(x, y)
    #     sim_val.append(ccc)
    #
    #
    #     # 选择子图
    #     # ax = axes[i // 3, i % 3]
    #     ax = axes[i // 5, i % 5]
    #     # ax = axes[i % 4]
    #     # ax = axes[i // 4, i % 4]
    #     # ax = axes
    #
    #     # 使用不同颜色绘制散点图
    #     # ax.scatter(x, y, color=colors[i % len(colors)], alpha=0.5)
    #     # ax.scatter(x, y, color=colors[i % len(colors)], s=100, alpha=0.5)
    #     ax.scatter(x, y, s=100, alpha=0.5)
    #     # ax.scatter(x, y, s=100, alpha=0.5)
    #     # ax.set_title(f"{col}\nLin's CCC: {ccc:.2f}")
    #     # ax.set_xlabel('Truth')
    #     # ax.set_ylabel('Prediction')
    #     # ax.set_title(f"{col}\nCosine Similarity: {ccc:.2f}", fontsize=25)
    #     ax.set_title(f"{col}\nDice: {ccc:.2f}", fontsize=25)
    #     # ['Scaden', 'CSx', 'MuSiC', 'NNLS', 'Rescue']
    #     ax.set_xlabel('Truth\nRescue', fontsize=24)
    #     ax.set_ylabel('Prediction', fontsize=24)
    #     # 调整刻度字体大小
    #     ax.tick_params(axis='both', which='major', labelsize=24)
    #
    # print(sim_val)
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存并显示图像
    # plt.savefig(file+'/plot_results/scatter_plots.png')
    # # plt.savefig(file + '/plot_results/unk_scatter_plots.png')
    # # plt.savefig('seurat_pbmc_2000/scatter_plots.png')
    # plt.show()
    # ------------------------------------------------------------------
    # # 创建用于存储所有数据的 DataFrame
    # rmse_combined = pd.DataFrame()
    # for col in columns_to_use:
    #     x = data1[col].values
    #     y = data2[col].values
    #     # 计算每对 x 和 y 值之间的 RMSE
    #     rmses = np.sqrt((x - y) ** 2)
    #     # 创建临时 DataFrame，包含当前列所有的 RMSE 值
    #     temp_df = pd.DataFrame({
    #         'RMSE': rmses,
    #         'Category': col
    #     })
    #     # 将临时 DataFrame 添加到组合 DataFrame 中
    #     rmse_combined = pd.concat([rmse_combined, temp_df], axis=0)
    # # 设置图形大小
    # plt.figure(figsize=(24, 8))
    #
    # # 调整刻度字体大小
    # ax.tick_params(axis='both', which='major', labelsize=25)
    # # 使用 Seaborn 绘制箱型图，调整点的大小
    # sns.boxplot(x='Category', y='RMSE', data=rmse_combined, palette='Set3', fliersize=13)
    # # 设置标签和标题
    # plt.xlabel('Category', fontsize=25)
    # plt.ylabel('RMSE', fontsize=25)
    # plt.title('Boxplots of RMSE for Each Category', fontsize=25)
    # # 调整刻度字体大小
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存并显示图像
    # plt.savefig(file+'/plot_results/rmse_boxplots.png')
    # plt.show()
    # ------------------------------------------------------------------
    # # 创建用于存储所有数据的 DataFrame
    # rmse_combined = pd.DataFrame()
    #
    # for col in columns_to_use:
    #     x = data1[col].values
    #     y = data2[col].values
    #
    #     # 计算每对 x 和 y 值之间的 RMSE
    #     rmses = np.sqrt((x - y) ** 2)
    #
    #     # 创建临时 DataFrame，包含当前列所有的 RMSE 值
    #     temp_df = pd.DataFrame({
    #         'RMSE': rmses,
    #         'Category': col
    #     })
    #
    #     # 将临时 DataFrame 添加到组合 DataFrame 中
    #     rmse_combined = pd.concat([rmse_combined, temp_df], axis=0)
    #
    # # 设置图形大小
    # plt.figure(figsize=(12, 5))
    #
    # # 使用 Seaborn 绘制小提琴图
    # sns.violinplot(x='Category', y='RMSE', data=rmse_combined, palette='Set3')
    #
    # # 设置标签和标题
    # plt.xlabel('Category', fontsize=20)
    # plt.ylabel('RMSE', fontsize=20)
    # # plt.title('Violin plots of RMSE for Each Category', fontsize=25)
    #
    # # plt.xticks(fontsize=20)
    # plt.xticks(rotation=30, fontsize=20)
    # plt.yticks(fontsize=20)
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存并显示图像
    # plt.savefig(file+'/plot_results/rmse_violinplots.png')
    # plt.show()
    # ------------------------------------------------------------------
    #
    # 绘制散点图
    plt.figure(figsize=(5, 5))
    # plt.scatter(long_vector1, long_vector2, alpha=0.5)
    # 绘制散点图，指定点的大小和颜色
    plt.scatter(long_vector1, long_vector2, alpha=0.5, s=50, color='HotPink')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=50, color='DeepSkyBlue')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=70, color='MediumAquamarine')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=60, color='DarkOrange')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=80, color='Tomato')
    plt.title(f"{filename}\nScatter Plot with Lin's CCC: {overall_ccc:.2f}")
    # plt.title(f"{filename}\nScatter Plot with cosine_val: {cosine_val_all:.2f}")
    # plt.title(f"Scatter Plot with Lin's CCC: {overall_ccc:.2f}")
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    # plt.grid(True)
    plt.grid(False)

    # 保存散点图为 PNG 文件
    plt.savefig(file+'/plot_results/scatter_plot_all.png')
    plt.show()
    # # ##-----------------------------------------------------
    # data1 = get_data(file1_path)
    # data2 = get_data(file2_path)
    # ## # -----------------------------------------------------
    # # 计算每行数据的RMSE值
    # rmse_values = []
    # for row1, row2 in zip(data1, data2):
    #     rmse_values.append(np.sqrt(mean_squared_error(row1, row2)))
    #
    # # 绘制箱型图
    # plt.figure(figsize=(8, 6))
    # plt.boxplot(rmse_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    # # plt.title('GSE50244')
    # plt.ylabel('RMSE')
    # plt.grid(True)
    #
    # # 保存箱型图为PNG文件
    # plt.savefig('gene_1/rmse_boxplot.png')
    #
    # # 显示图形
    # plt.show()
    # # # # -----------------------------------------------------------------------------------
    # # # 计算每行数据的CCC值
    # # ccc_values = []
    # # for row1, row2 in zip(data1, data2):
    # #     ccc_values.append(lins_ccc(row1, row2))
    # #
    # # # 绘制箱线图
    # # plt.figure(figsize=(8, 6))
    # # plt.boxplot(ccc_values, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    # # # plt.title('GSE50244')
    # # plt.ylabel('CCC')
    # # plt.grid(True)
    # #
    # # # 保存箱线图为PNG文件
    # # plt.savefig('gene_1/ccc_boxplot.png')
    # #
    # # # 显示图形
    # # plt.show()
    # # # # # ----------------------------------------------------
    # file = 'LUSC_IA3_BC8_10/property/others_pred_LUSC_IA3_BC8_10'
    # file = 'LUSC_IA3_BC3_10/property/others_pred_LUSC_IA3_BC3_10'
    file = 'LUSC_IA3_BC3_10/20000/pred'
    # file = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/need/pred'

    file0_path = file + '/truth_data.txt'
    file1_path = file + '/out_predict_scaden.txt'
    file2_path = file + '/prediction_CSx.txt'
    file3_path = file + '/prediction_cmp.txt'
    file4_path = file + '/prediction_music.txt'
    file5_path = file + '/prediction_nnls.txt'
    file6_path = file + '/prediction_res_4000.txt'
    lie = 11

    # 文件路径列表
    file_paths = [file1_path, file2_path, file3_path, file4_path, file5_path, file6_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file4_path, file5_path, file6_path]
    # file_paths = [file1_path, file2_path, file3_path, file4_path, file5_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file3_path, file5_path, file6_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file3_path, file5_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file3_path, file4_path, file5_path]
    # file_paths = [file1_path, file3_path, file4_path, file5_path, file6_path]  # 根据需要添加更多文件路径

    # # 存储每个文件对比的RMSE值
    # all_rmse_values = []
    #
    # # 计算每个文件的RMSE值
    # data1 = get_data(file0_path, lie)
    # for file_path in file_paths:
    #     # data1 = get_data(file0_path)
    #     data2 = get_data(file_path, lie)  # 假设是对比同一文件的不同版本，如果是不同文件，更新为不同路径
    #     rmse_values = []
    #     for row1, row2 in zip(data1, data2):
    #         rmse_values.append(np.sqrt(mean_squared_error(row1, row2)))
    #     all_rmse_values.append(rmse_values)
    #
    # # 绘制箱型图
    # plt.figure(figsize=(8, 4))
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'CornflowerBlue', 'Pink']
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'CornflowerBlue']
    # boxprops = [dict(facecolor=color) for color in colors]
    # # 设置每个箱体的颜色
    # for patch, color in zip(plt.boxplot(all_rmse_values, patch_artist=True, flierprops=dict(marker='o', markersize=8))['boxes'], colors):
    #     patch.set_facecolor(color)
    # # plt.boxplot(all_rmse_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    # # plt.boxplot(all_rmse_values, patch_artist=True)
    #
    # # 设置X轴标签为文件名或其他合适的名称
    # file_name=['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # # file_name = ['Scaden', 'CSx', 'MuSiC', 'NNLS', 'Rescue']
    # # file_name = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # # file_name = ['scaden', 'cmp', 'music', 'nnls']
    # # plt.xticks(ticks=range(1, len(file_paths) + 1), labels=[f'File {i + 1}' for i in range(len(file_paths))])
    # plt.xticks(ticks=range(1, len(file_paths) + 1), labels=file_name, fontsize=15)
    # plt.yticks(fontsize=15)
    # # plt.title('mouse')
    # # plt.xlabel('Unknow_Celltype', fontsize=15)
    # plt.ylabel('RMSE', fontsize=15)
    # # 去掉网格线
    # plt.grid(False)
    #
    # # 保存箱型图为PNG文件
    # plt.savefig(file+'/plot_results/rmse_boxplot_all.png')
    # # plt.savefig(file + '/plot_results/rmse_boxplot_all_unknow.png')
    # # plt.savefig('dataset/rmse_boxplot_all.png')
    #
    # # 显示图形
    # plt.show()
    # # # ----------------------------------------------------
    # # 存储每个文件对比的CCC值
    # all_ccc_values = []
    #
    # # 计算每个文件的CCC值
    # data1 = get_data(file0_path, lie)
    # for file_path in file_paths:
    #     data2 = get_data(file_path, lie)
    #     ccc_values = []
    #     for row1, row2 in zip(data1, data2):
    #         ccc_values.append(lins_ccc(row1, row2))
    #     all_ccc_values.append(ccc_values)
    #
    # # 绘制箱型图
    # plt.figure(figsize=(8, 4))
    #
    # # 使用不同颜色填充箱体
    # # colors = ['LightSkyBlue', 'Lavender', 'LightGoldenrodYellow', 'LightSalmon']
    # colors = ['SkyBlue', 'Lavender', 'LightGoldenrodYellow', 'HotPink', 'MintCream', 'Magenta']
    # # colors = ['SkyBlue', 'Lavender', 'LightGoldenrodYellow', 'HotPink', 'MintCream']
    #
    # # 绘制每个算法的箱型图，并设置不同的填充颜色
    # boxplot = plt.boxplot(all_ccc_values, patch_artist=True, flierprops=dict(marker='o', markersize=8))
    #
    # # 设置每个箱体的颜色
    # for patch, color in zip(boxplot['boxes'], colors):
    #     patch.set_facecolor(color)
    #
    # # 设置X轴标签为文件名或其他合适的名称
    # # file_name = ['scaden', 'cmp', 'music', 'nnls']
    # file_name = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # # file_name = ['Scaden', 'CSx', 'MuSiC', 'NNLS', 'Rescue']
    # # file_name = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # plt.xticks(ticks=range(1, len(file_paths) + 1), labels=file_name, fontsize=15)
    # plt.yticks(fontsize=15)
    #
    # # plt.ylabel('CCC')
    # # plt.xlabel('Unknow_celltype', fontsize=15)
    # plt.ylabel('CCC', fontsize=15)
    # # 去掉网格线
    # plt.grid(False)
    #
    # # 保存箱型图为PNG文件
    # plt.savefig(file+'/plot_results/ccc_boxplot_all.png')
    # # plt.savefig(file + '/plot_results/ccc_boxplot_all_unknow.png')
    # # plt.savefig('dataset/ccc_boxplot_all.png')
    # # 显示图形
    # plt.show()
    # # # ----------------------------------------------------
    # plt.figure(figsize=(18, 10))
    # # colors = ['blue', 'green', 'red', 'orange', 'PowDerBlue']
    # # file_names = ['scaden', 'cmp', 'music', 'nnls']
    #
    # colors = ['blue', 'green', 'red', 'orange', 'DeepSkyBlue', 'HotPink']
    # # file_names = ['scaden', 'CSx', 'cmp', 'music', 'nnls']
    # # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # file_names = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # # file_names = ['Scaden', 'CSx', 'MuSiC', 'NNLS', 'Rescue']
    #
    # # 读取基准数据
    # data1 = get_data(file0_path, lie)
    # long_vector1 = np.concatenate(data1)
    #
    # # 创建子图
    # fig, axes = plt.subplots(2, (len(file_paths) + 1) // 2, figsize=(15, 10), sharex=True, sharey=True)
    # # fig, axes = plt.subplots(1, len(file_paths), figsize=(25, 5), sharex=True, sharey=True)
    #
    # # 将 axes 转换为 1D 数组以便于迭代
    # axes = axes.flatten()
    #
    # # 绘制每个算法的散点图
    # for i, (ax, file_path, file_name, color) in enumerate(zip(axes, file_paths, file_names, colors)):
    #     data2 = get_data(file_path, lie)
    #     long_vector2 = np.concatenate(data2)
    #
    #     overall_ccc = lins_ccc(long_vector1, long_vector2)
    #
    #     data1 = get_data(file0_path, lie)
    #     for file_path in file_paths:
    #         data2 = get_data(file_path, lie)
    #         ccc_values = []
    #         for row1, row2 in zip(data1, data2):
    #             ccc_values.append(lins_ccc(row1, row2))
    #         all_ccc_values.append(ccc_values)
    #
    #     ax.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color=color)
    #     ax.set_title(f"{file_name}\nCCC: {overall_ccc:.2f}", fontsize=20)
    #     ax.set_xlabel('Truth', fontsize=20)
    #     if i % 3 == 0:
    #     # if i == 0:
    #         ax.set_ylabel('Prediction', fontsize=20)
    #
    #     # 设置X轴和Y轴为黑色实线
    #     ax.spines['left'].set_color('black')
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_color('black')
    #     ax.spines['bottom'].set_linewidth(1.5)
    #
    #     # 隐藏顶部和右侧的边框
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.grid(False)
    #
    #     # 设置刻度字体大小
    #     ax.tick_params(axis='both', which='major', labelsize=20)
    #
    # # # 如果子图数量是奇数，移除最后一个子图
    # # if len(file_paths) % 2 != 0:
    # #     fig.delaxes(axes[-1])
    #
    # # plt.xlabel('Unknow_celltype', fontsize=15)
    # plt.tight_layout()
    # # 保存散点图为 PNG 文件
    # plt.savefig(file + '/plot_results/scatter_plot_all.png')
    # # plt.savefig(file + '/plot_results/scatter_plot_all_unknow.png')
    # # plt.savefig('dataset/scatter_plot_all.png')
    # plt.show()
    # # # ----------------------------------------------------
    # plt.figure(figsize=(18, 10))
    # colors = ['blue', 'green', 'red', 'orange', 'PowDerBlue']
    # # file_names = ['scaden', 'cmp', 'music', 'nnls']
    #
    # # colors = ['blue', 'green', 'red', 'orange', 'DeepSkyBlue', 'HotPink']
    # # file_names = ['scaden', 'CSx', 'cmp', 'music', 'nnls']
    # # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # # file_names = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # file_names = ['Scaden', 'CSx', 'MuSiC', 'NNLS', 'Rescue']
    #
    # # 读取基准数据
    # data1 = get_data(file0_path, lie)
    # long_vector1 = np.concatenate(data1)
    #
    # # 创建子图
    # # fig, axes = plt.subplots(2, (len(file_paths) + 1) // 2, figsize=(15, 10), sharex=True, sharey=True)
    # fig, axes = plt.subplots(1, len(file_paths), figsize=(25, 5), sharex=True, sharey=True)
    #
    # # 将 axes 转换为 1D 数组以便于迭代
    # axes = axes.flatten()
    #
    # # 绘制每个算法的散点图
    # for i, (ax, file_path, file_name, color) in enumerate(zip(axes, file_paths, file_names, colors)):
    #     data2 = get_data(file_path, lie)
    #     long_vector2 = np.concatenate(data2)
    #
    #     overall_ccc = lins_ccc(long_vector1, long_vector2)
    #
    #     ax.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color=color)
    #     ax.set_title(f"{file_name}\nCCC: {overall_ccc:.2f}", fontsize=20)
    #     ax.set_xlabel('Truth', fontsize=20)
    #     # if i % 2 == 0:
    #     if i == 0:
    #         ax.set_ylabel('Prediction', fontsize=20)
    #
    #     # 设置X轴和Y轴为黑色实线
    #     ax.spines['left'].set_color('black')
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_color('black')
    #     ax.spines['bottom'].set_linewidth(1.5)
    #
    #     # 隐藏顶部和右侧的边框
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.grid(False)
    #
    #     # 设置刻度字体大小
    #     ax.tick_params(axis='both', which='major', labelsize=20)
    #
    # # # 如果子图数量是奇数，移除最后一个子图
    # # if len(file_paths) % 2 != 0:
    # #     fig.delaxes(axes[-1])
    #
    # plt.xlabel('Unknow_celltype', fontsize=15)
    # plt.tight_layout()
    # # 保存散点图为 PNG 文件
    # # plt.savefig(file + '/plot_results/scatter_plot_all.png')
    # plt.savefig(file + '/plot_results/scatter_plot_all_unknow.png')
    # # plt.savefig('dataset/scatter_plot_all.png')
    # plt.show()
    # --------------------------------------------------------------------
    Pearson_val_list = []
    Dice_val_list = []
    cosine_val_list = []
    fidelity_val_list = []
    # jaccard_val_list = []
    for file_path in file_paths:
        Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all = simil(file0_path, file_path, lie)

        # Pearson_val_list.append(Pearson_val_all)
        Dice_val_list.append(Dice_val_all)
        # cosine_val_list.append(cosine_val_all)
        fidelity_val_list.append(fidelity_val_all)
        # jaccard_val_list.append(jaccard_val_all)

    # 设置指标名称和颜色
    # metrics = ['Pearson', 'Dice', 'Cosine', 'Fidelity']
    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'Pink']
    # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    #
    # metrics = ['Pearson', 'Dice', 'Cosine']
    # colors = ['lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'Pink']
    # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']

    # metrics = ['Pearson', 'Dice', 'Cosine']
    # metrics = ['Dice', 'Cosine']
    metrics = ['Fidelity', 'Dice']
    colors = ['red', 'lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'DarkOrange']
    file_names = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # file_names = ['scaden', 'cmp', 'music', 'nnls', 'res']

    # 准备数据
    # all_metrics = [Pearson_val_list, Dice_val_list, cosine_val_list, fidelity_val_list]
    # all_metrics = [Pearson_val_list, Dice_val_list, cosine_val_list]
    # all_metrics = [Dice_val_list, cosine_val_list]
    all_metrics = [fidelity_val_list, Dice_val_list]
    num_files = len(file_paths)
    num_metrics = len(metrics)

    # 设置柱状图的位置
    bar_width = 0.12
    # index = np.arange(num_files)
    index = np.arange(num_metrics)
    # 绘制柱状图
    plt.figure(figsize=(12, 6))

    for i in range(num_files):
        plt.bar(index + i * bar_width, [all_metrics[j][i] for j in range(num_metrics)], bar_width, label=file_names[i],
                color=colors[i % len(colors)])
    # for i in range(num_metrics):
    #     plt.bar(index + i * bar_width, all_metrics[i], bar_width, label=metrics[i], color=colors[i % len(colors)])

    # 设置X轴标签为相似性指标
    plt.xticks(index + bar_width * (num_files - 1) / 2, metrics, fontsize=15)
    plt.yticks(fontsize=15)

    # # # 设置X轴标签为文件名或其他合适的名称
    # plt.xticks(index + bar_width * (num_metrics - 1) / 2, file_names)

    # plt.xlabel('Files')
    plt.ylabel('Values', fontsize=17)
    # plt.title('Comparison of Different Metrics for Various Predictions')
    # plt.legend()
    # 将图例固定在左上角
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=False, fontsize=10)
    # 设置图例，固定在左上角，并自适应布局
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, frameon=False, fontsize=11)

    # 自动调整布局以防止重叠
    plt.tight_layout()

    # 保存柱状图为PNG文件
    # plt.savefig(file+'/plot_results/simil_barplot.png')
    # plt.savefig(file + '/plot_results/simil_barplot.png', dpi=300)
    # 保存柱状图为PNG文件，并设置较高的dpi以提高图像清晰度
    # plt.savefig(file + '/plot_results/simil_barplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(file + '/plot_results/simil_barplot.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
    # ---------------------------------------------
    print('over.....')

