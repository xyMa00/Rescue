import anndata as ad
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from Similarity_eval import *
import seaborn as sns
import glob
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap


# 读取文件数据并提取数值部分
def read_txt_file(file_path):
    # 使用 pandas 读取文件，跳过第一行的列名
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    return df.values.flatten()  # 将 DataFrame 转换为一维数组

# 将数据转换为布尔值
def convert_to_boolean(data, threshold=0.1):
    return data >= threshold

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, output_file):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True'])
    plt.figure(figsize=(5, 5))
    ax = disp.plot(cmap='Blues', values_format='d', colorbar=False).ax_  # 获取轴对象

    # 修改字体大小
    for text in ax.texts:  # 遍历混淆矩阵中的数字
        text.set_fontsize(16)  # 修改数字字体大小

    ax.set_xlabel('Predicted Label', fontsize=18)  # x轴标签字体
    ax.set_ylabel('True Label', fontsize=18)  # y轴标签字体
    plt.title(filename, fontsize=20)  # 标题字体

    # 修改x轴和y轴刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=16)  # x/y轴刻度字体
    ax.set_xticklabels(['False', 'True'], fontsize=16)  # 修改x轴标签字体
    ax.set_yticklabels(['False', 'True'], fontsize=16)  # 修改y轴标签字体

    # 保存图片时去除多余空白
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"混淆矩阵已保存到 {output_file}")

# 主程序
def confusion_matrix_plot(file_true, file_pred, output_png):
    # 读取数据
    y_true = read_txt_file(file_true)
    y_pred = read_txt_file(file_pred)

    # 转换为布尔值
    y_true = convert_to_boolean(y_true)
    y_pred = convert_to_boolean(y_pred)

    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, output_png)
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
    # file = 'LUSC_IA3_BC3_10/all/pred'
    # file = 'LUSC_IA3_BC3_10/all/others_pred_LUSC_IA3_BC3_10'
    # file = 'LUSC_IA3_BC3_10/2500/pred'
    # file = 'LUSC_IA3_BC3_10/20000/pred'
    # file = 'LUSC_IA3_BC8_10/property/pred'

    # file = 'HP1504101T2D/real/pred'
    # file = 'HP1508501T2D/real/pred'
    # file = 'E-MTAB/10000/HP1507101/4000/pred'
    # file = 'E-MTAB/10000/HP1504101T2D/pred/1000'
    # file = 'E-MTAB_1/10000/all_3/pred'
    # file = 'E-MTAB_1/10000/all_7/marge_7/pred'
    # file = 'E-MTAB_1/10000/all_7/marge_7/single_7/pred'
    # file = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/train_dataset/pred'
    # file = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/need/pred'
    # filename = 'HSCR'
    file = 'BBI/Intestine/marge/pred'

    # file = 'LUSC_IA3_BC3_10/2500/4000/pred'
    # file = 'mouse/4000/pred'
    # file = 'LUSC_IA3_BC8_10/others_pred_LUSC_IA3_BC8_10/heatmap'
    # file = 'LUSC_IA3_BC3_10/20000/pred'
    # file = 'LUSC_IA3_BC3_10/all/pred'
    # filename = 'LUSC'
    # file = 'hscr/20000/pred'
    # file = 'mouse/4000/pred'
    # filename = 'Mouse'
    # file = 'GSE257541/10000/pbmc48h/4000/pred'

    # file = 'LUSC_10_types/marge_20000/train_dataset/pred'
    # file = 'seurat_pbmc/4000/pred'
    # file = 'seurat_pbmc/20000/train_dataset/pred'
    # file = 'seurat_pbmc/4000/10000/pred'
    # file = 'seurat_pbmc/pbmc_3k_8_2/pred/pred_all'
    # file = 'seurat_pbmc/pbmc_3k_8_2/pred/pred_01'
    file1_path = file + '/truth_data.txt'
    # file1_path = file + '/truth_data_4.txt'
    # file1_path = file + '/truth_data_7_all.txt'
    # file1_path = file + '/truth_data_new.txt'
    # file2_path = file+'/out_predict_scaden.txt'
    # file2_path = file + '/out_predict_scaden_7.txt'
    # file2_path = file + '/out_predict_scaden_7_all.txt'
    # file2_path = file + '/prediction_CSx.txt'
    # file2_path = file + '/prediction_cmp.txt'
    # file2_path = file+'/prediction_music.txt'
    # file2_path = file + '/prediction_nnls.txt'
    # file2_path = file + '/prediction_CSx_7_all.txt'
    # file2_path = file + '/prediction_cmp_7_all.txt'
    # file2_path = file+'/prediction_music_7_all.txt'
    # file2_path = file + '/prediction_nnls_7_all.txt'
    # file2_path = file+'/prediction_res_8000.txt'
    # file2_path = file+'/prediction_res_6400.txt'
    # file2_path = file+'/prediction_res_4800.txt'
    # file2_path = file + '/prediction_res_4000.txt'
    # file2_path = file + '/prediction_res_4000_4.txt'
    # file2_path = file + '/prediction_res_4000_7_all.txt'
    # file2_path = file + '/prediction_res_4000_new.txt'
    # file2_path = file+'/prediction_res_3200.txt'
    # file2_path = file+'/prediction_res_1600.txt'
    # file2_path = file+'/prediction_res_800.txt'
    # file2_path = file+'/prediction_res_400.txt'
    # file2_path = file + '/prediction_res_200.txt'
    # file2_path = file + '/prediction_res_200_1.txt'
    # file2_path = file + '/prediction_res_200_2.txt'
    # file2_path = file + '/prediction_DWLS_7_all.txt'
    # file2_path = file + '/prediction_BayesPrism_7_all.txt'
    # file2_path = file + '/out_predict_scaden_01.txt'
    # file2_path = file + '/prediction_CSx_01.txt'
    # file2_path= file + '/prediction_cmp_01.txt'
    # file2_path = file + '/prediction_music_01.txt'
    # file2_path = file + '/prediction_nnls_01.txt'
    # file2_path = file + '/prediction_res_4000_01.txt'
    # file2_path = file + '/prediction_DWLS_01.txt'
    file2_path= file + '/prediction_BayesPrism.txt'
    # file1_path = file + '/truth_data_HP1504101T2D.txt'
    # file2_path = file + '/prediction_res_HP1504101T2D.txt'
    # file2_path = file + '/out_predict_scaden_HP1504101T2D.txt'
    # file2_path = file + '/prediction_cmp_HP1504101T2D.txt'
    # file2_path = file + '/prediction_CSx_HP1504101T2D.txt'
    # file2_path = file + '/prediction_music_HP1504101T2D.txt'
    # file2_path = file + '/prediction_nnls_HP1504101T2D.txt'
    # file2_path = file + '/prediction_BayesPrism_HP1504101T2D.txt'


    # filename = 'PBMC'
    # filename = 'LUSC'
    # filename = 'Mouse'
    # filename = 'HSCR'
    # filename = 'Rescue'
    # filename = 'Scaden'
    # filename = 'CPM'
    # filename = 'CSx'
    # filename = 'MuSiC'
    # filename = 'NNLS'
    # filename = 'DWLS'
    filename = 'BayesPrism'


    # filename = 'Pearson'
    # filename = 'Dice'
    # filename = 'Fidelity'
    # filename = 'Cosine'
    # filename = 'Pancreas'
    # filename = 'Real'
    # bulkname = 'HP1504101T2D'
    bulkname = 'all_6'


    # # # 读取文件，假设文件使用制表符作为分隔符
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep='\t')
    #
    # # # 提取第2到第8列
    # columns_to_use = df1.columns[1:5]
    # columns_to_use = df1.columns[1:7]
    # columns_to_use = df1.columns[1:14]
    # columns_to_use = df1.columns[1:23]
    # columns_to_use = df1.columns[1:10]
    # columns_to_use = df1.columns[1:11]
    columns_to_use = df1.columns[1:9]
    # columns_to_use = df1.columns[0:1]
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
    #
    # # # 计算similarity
    # # Pearson_val_all = PearsonCorrelationSimilarity(long_vector1, long_vector2)
    # # print(f"Pearson_val_all: {Pearson_val_all}")
    # #
    # # Dice_val_all = dice_coefficient(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"Dice_val_all: {Dice_val_all}")
    # #
    # # # 计算余弦相似度
    # # cosine_val_all = python_cos(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"cosine_val_all: {cosine_val_all}")
    # #
    # # fidelity_val_all = fidelity_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"fidelity_val_all: {fidelity_val_all}")
    # #
    # # # euclidean_distance_val_all = euclidean_distance_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # # print(f"euclidean_distance_val_all: {euclidean_distance_val_all}")
    # #
    # # # squared_val_all = squared_chord_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # # print(f"squared_chord_val_all: {squared_val_all}")
    # #
    # # jaccard_val_all = jaccard_similarity(long_vector1.tolist(), long_vector2.tolist())
    # # print(f"jaccard_val_all: {jaccard_val_all}")
    # #
    # #
    # # # 计算每列对应数据的 Lin's CCC
    # # Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all, jaccard_val_all = simil(long_vector1, long_vector2)
    # #
    # # ccc_values = {}
    # # for col in columns_to_use:
    # #     ccc_values[col] = lins_ccc(data1[col].values, data2[col].values)
    # #
    # # # 输出结果
    # # for col, ccc in ccc_values.items():
    # #     print(f"Lin's CCC for column {col}: {ccc}")
    # #
    # # # # ------------------------------------------------------------
    # # 计算每列对应数据的 Lin's CCC 并绘制散点图
    # # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    # # # fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))
    # # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    # # # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # #
    # # # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    # #
    # # for i, col in enumerate(columns_to_use):
    # #     x = data1[col].values
    # #     y = data2[col].values
    # #     ccc = lins_ccc(x, y)
    # #
    # #     # ax = axes[i // 4, i % 4]
    # #     # ax = axes[ i % 6]
    # #     ax = axes[i // 3, i % 3]
    # #     # ax = axes[i % 4]
    # #     # ax = axes[i % 2]
    # #     ax.scatter(x, y, alpha=0.5)
    # #     ax.set_title(f"{col}\nLin's CCC: {ccc:.2f}")
    # #     ax.set_xlabel('Truth')
    # #     ax.set_ylabel('Prediction')
    # #
    # # # # 隐藏多余的子图
    # # # for j in range(i + 1, 8):
    # # #     fig.delaxes(axes[j // 4, j % 4])
    # #
    # # plt.tight_layout()
    # # plt.savefig(file+'/plot_results/scatter_plots.png')
    # # plt.show()
    # # # # # # ------------------------------------------------------------
    # # 定义颜色列表
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
    #     ax = axes[i // 3, i % 3]
    #     # ax = axes[i // 5, i % 5]
    #     # ax = axes[i % 4]
    #     # ax = axes[i // 4, i % 4]
    #     # ax = axes
    #
    #     # 使用不同颜色绘制散点图
    #     ax.scatter(x, y, color=colors[i % len(colors)], alpha=0.5)
    #     # ax.scatter(x, y, color=colors[i % len(colors)], s=100, alpha=0.5)
    #     # ax.scatter(x, y, s=100, alpha=0.5)
    #     # ax.scatter(x, y, s=100, alpha=0.5)
    #     # ax.set_title(f"{col}\nLin's CCC: {ccc:.2f}")
    #     # ax.set_xlabel('Truth')
    #     # ax.set_ylabel('Prediction')
    #     # ax.set_title(f"{col}\nCosine Similarity: {ccc:.2f}", fontsize=25)
    #     # ax.set_title(f"{col}\nDice: {ccc:.2f}", fontsize=25)
    #     ax.set_title(f"{col}\n", fontsize=25)
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
    # plt.savefig(file+'/plot_results/scatter_plots.png', dpi=600)
    # # plt.savefig(file + '/plot_results/unk_scatter_plots.png')
    # # plt.savefig('seurat_pbmc_2000/scatter_plots.png')
    # plt.show()
    # # ------------------------------------------------------------------
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
    # # ax.tick_params(axis='both', which='major', labelsize=25)
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
    # # ------------------------------------------------------------------
    # 创建用于存储所有数据的 DataFrame
    # rmse_combined = pd.DataFrame()
    # for col in columns_to_use:
    #     x = data1[col].values
    #     y = data2[col].values
    #     # 计算每对 x 和 y 值之间的 RMSE
    #     rmses = np.sqrt((x - y) ** 2)
    #
    #     # 创建临时 DataFrame，包含当前列所有的 RMSE 值
    #     temp_df = pd.DataFrame({
    #         'RMSE': rmses,
    #         'Category': col
    #     })
    #     # 将临时 DataFrame 添加到组合 DataFrame 中
    #     rmse_combined = pd.concat([rmse_combined, temp_df], axis=0)
    # # 设置图形大小
    # plt.figure(figsize=(13, 5))
    # # 使用 Seaborn 绘制小提琴图
    # sns.violinplot(x='Category', y='RMSE', data=rmse_combined, palette='Set3')
    #
    # # 设置标签和标题
    # plt.xlabel('Category', fontsize=20)
    # plt.ylabel('RMSE', fontsize=20)
    # # plt.title('Using all genes', fontsize=25)
    # # plt.title('using few highly variable features', fontsize=25)
    # plt.title('RMSE for Each Category', fontsize=25)
    # # plt.xticks(fontsize=20)
    # plt.xticks(rotation=30, fontsize=20)
    # plt.yticks(fontsize=20)
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存并显示图像
    # plt.savefig(file+'/plot_results/rmse_violinplots.png')
    # plt.show()
    # # ------------------------------------------------------------------
    # #
    # # # 绘制散点图
    # # plt.figure(figsize=(5, 5))
    # # # plt.scatter(long_vector1, long_vector2, alpha=0.5)
    # # # 绘制散点图，指定点的大小和颜色
    # # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='HotPink')
    # # # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='DeepSkyBlue')
    # # # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='MediumAquamarine')
    # # # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='DarkOrange')
    # # # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='Tomato')
    # # plt.title(f"{filename}\nScatter Plot with Lin's CCC: {overall_ccc:.2f}")
    # # # plt.title(f"{filename}\nScatter Plot with cosine_val: {cosine_val_all:.2f}")
    # # # plt.title(f"Scatter Plot with Lin's CCC: {overall_ccc:.2f}")
    # # plt.xlabel('Truth')
    # # plt.ylabel('Prediction')
    # # # plt.grid(True)
    # # plt.grid(False)
    # #
    # # # 保存散点图为 PNG 文件
    # # plt.savefig(file+'/plot_results/scatter_plot_all.png')
    # # plt.show()
    # # --------------------------------逐行计算CC并绘制散点图----------------------------
    # 确保两个文件的行数一致
    # assert df1.shape == df2.shape, "两个文件的行数或列数不一致"
    # # 提取数据部分，不指定列，使用所有数值列
    # data1 = df1.select_dtypes(include=[np.number])
    # data2 = df2.select_dtypes(include=[np.number])
    # # 逐行计算 Lin's CCC
    # row_cccs = []
    # for row1, row2 in zip(data1.to_numpy(), data2.to_numpy()):
    #     ccc = lins_ccc(row1, row2)  # 假设 lins_ccc 可以接受两个向量作为输入
    #     row_cccs.append(ccc)
    # # 将行 CCC 结果输出为 DataFrame
    # row_ccc_df = pd.DataFrame({'Row_Index': np.arange(len(row_cccs)), 'CCC': row_cccs})
    # # 计算总体的 Lin's CCC
    # long_vector1 = data1.to_numpy().flatten()
    # long_vector2 = data2.to_numpy().flatten()
    # overall_ccc = lins_ccc(long_vector1, long_vector2)
    # # 打印每行的CCC和整体CCC
    # print(f"每行的CCC值：\n{row_ccc_df}")
    # print(f"整体的CCC值：{overall_ccc:.2f}")
    # #-----------------------------------------------------------------
    # 绘制散点图
    plt.figure(figsize=(5, 5))
    # 散点图，表示 truth 和 prediction 数据的关系
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#E64B35')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#4DBBD5')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#00A087')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#3C5488')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#F39B7F')
    # 黄色颜色有点浅
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#F5D86A')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#E6A5D7')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=100, color='#C48B9F')

    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='HotPink')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#E64B35')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#4DBBD5')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#00A087')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#3C5488')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#F39B7F')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#E6A5D7')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#C48B9F')

    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#E64B35')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#4DBBD5')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#00A087')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#3C5488')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#F39B7F')
    plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#E6A5D7')
    # plt.scatter(long_vector1, long_vector2, alpha=0.5, s=200, color='#C48B9F')
    # 绘制从零点 (0,0) 开始的对角线
    plt.plot([0, 1], [0, 1], color=(126/255, 126/255, 126/255), linestyle='--', linewidth=3)
    # plt.title(f"{filename}\nLin's CCC: {overall_ccc:.2f}", fontsize=20)
    plt.title(f"{filename}", fontsize=20)
    # 设置 X 和 Y 轴的标签和刻度
    plt.xlabel('Truth', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # 自适应布局和网格设置
    plt.tight_layout()
    plt.grid(False)
    # 保存散点图为 PNG 文件
    # plt.savefig(file + '/plot_results/' + filename + '_scatter_plot_all_7.png', dpi=600)
    plt.savefig(file + '/plot_results/' + filename + '_scatter_plot_' + bulkname + '.png', dpi=600)
    plt.show()
    # 保存每行的CCC结果为 CSV 文件
    # row_ccc_df.to_csv(file + '/plot_results/' + filename + '_row_cccs.csv', index=False)
    # # # ##-----------------------------------------------------
    # # data1 = get_data(file1_path)
    # # data2 = get_data(file2_path)
    # # ## # -----------------------------------------------------
    # # # 计算每行数据的RMSE值
    # # rmse_values = []
    # # for row1, row2 in zip(data1, data2):
    # #     rmse_values.append(np.sqrt(mean_squared_error(row1, row2)))
    # #
    # # # 绘制箱型图
    # # plt.figure(figsize=(8, 6))
    # # plt.boxplot(rmse_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    # # # plt.title('GSE50244')
    # # plt.ylabel('RMSE')
    # # plt.grid(True)
    # #
    # # # 保存箱型图为PNG文件
    # # plt.savefig('gene_1/rmse_boxplot.png')
    # #
    # # # 显示图形
    # # plt.show()
    # # # # # -----------------------------------------------------------------------------------
    # # # # 计算每行数据的CCC值
    # # # ccc_values = []
    # # # for row1, row2 in zip(data1, data2):
    # # #     ccc_values.append(lins_ccc(row1, row2))
    # # #
    # # # # 绘制箱线图
    # # # plt.figure(figsize=(8, 6))
    # # # plt.boxplot(ccc_values, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    # # # # plt.title('GSE50244')
    # # # plt.ylabel('CCC')
    # # # plt.grid(True)
    # # #
    # # # # 保存箱线图为PNG文件
    # # # plt.savefig('gene_1/ccc_boxplot.png')
    # # #
    # # # # 显示图形
    # # # plt.show()
    # # # ------------------------------------plot_heatmap-----------------------------------------
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # 读取数据
    # data = pd.read_csv(file2_path, sep='\t', index_col=0)
    # # 绘制热图
    # plt.figure(figsize=(6, 12))
    # # sns.heatmap(data, cmap='coolwarm', annot=False)
    # sns.heatmap(data, cmap='coolwarm', annot=False,vmin=0, vmax=1)
    # plt.title(filename, fontsize=22)
    # # 设置坐标轴名称
    # plt.xlabel('Celltype', fontsize=18)  # 横坐标名称
    # plt.ylabel('Sample', fontsize=18)  # 纵坐标名称
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # # 保存为png文件
    # plt.savefig(file + '/heatmap.png', dpi=600)
    # # 展示图像（可选）
    # plt.show()
    # --------------------极坐标柱状图----------------------------------
    # 数据
    # categories = ['Scaden', 'Music', 'NNLS', 'CPM', 'CSx', 'Rescue']
    # # values = [0.9871, 0.842, 0.6682, 0.6337, 0.8059, 0.9932]    # Pearson
    # # values = [0.9879, 0.8641, 0.7339, 0.5741, 0.8506, 0.9936]    #Dice
    # # values = [0.9556, 0.9063, 0.7229, 0.7368, 0.8988, 0.9728]    #Fidelity
    # values = [0.99, 0.8855, 0.7388, 0.725, 0.8554, 0.9948]  # Cosine
    # # 角度设置，将类别映射到角度
    # num_vars = len(categories)
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # # 使角度周期闭合
    # angles += angles[:1]
    # values += values[:1]
    # # 创建极坐标图
    # fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    # # 背景颜色划分 (左半边和右半边)
    # theta = np.linspace(0, 2 * np.pi, 100)
    # r = np.ones(100)
    # # # 填充左半边背景色
    # # ax.fill_between(theta, 0, r, where=(theta >= 0) & (theta < np.pi), facecolor='#FFFEEC', zorder=-1)
    # # # 填充右半边背景色
    # # ax.fill_between(theta, 0, r, where=(theta >= np.pi) & (theta <= 2 * np.pi), facecolor='#E5EAF3', zorder=-1)
    # # 设置中心的最小值为0.3，半径范围为0.3到1
    # min_val = 0.4
    # ax.set_ylim(min_val, 1)
    # # 绘制柱状图
    # base_value = min_val  # 中心的基线值
    # adjusted_values = [v - base_value for v in values]  # 每个值相对基线的变化
    # # bars = ax.bar(angles[:-1], adjusted_values[:-1], bottom=base_value,
    # #               color='#FFFFFF', edgecolor='black', linewidth=2, width=0.4, align='edge')
    # # colors = plt.cm.viridis(values)
    # colors = ['#CCE5FF', '#CCFFCC', '#FFF5CC', '#FFD6E7', '#E6CCFF', '#FFEACC']
    # # 淡蓝色#CCE5FF 淡绿色#CCFFCC 淡黄色：#FFF5CC 淡粉色：#FFD6E7 淡紫色：#E6CCFF 淡橙色：#FFEACC
    # bars = ax.bar(angles[:-1], adjusted_values[:-1], bottom=base_value,
    #               color=colors, linewidth=2, width=0.4, align='edge')
    # # bars = ax.bar(angles[:-1], adjusted_values[:-1], bottom=base_value,
    # #               color='#CCE5FF',  linewidth=2, width=0.4, align='edge')
    # # bars = ax.bar(angles[:-1], values[:-1], color='#FFFFFF', edgecolor='black', linewidth=2, width=0.4, align='edge')
    # # 设置刻度标签（类别）
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories, color='black', fontsize=14, horizontalalignment='center')
    # # 将类别标签移到图形外面
    # ax.tick_params(axis='x', pad=15)  # 调整pad值增加标签与图形边界的距离
    #
    # # 取消数据刻度显示
    # ax.set_yticks([])  # 隐藏 y 轴刻度
    # ax.xaxis.set_visible(False)
    #
    # # 去掉圆圈内部的实线
    # ax.spines['polar'].set_visible(False)
    #
    # # ax.spines['polar'].set_visible(True)
    # # # # 设置最外层轮廓的颜色和宽度
    # # ax.spines['polar'].set_edgecolor('#6A5ACD')  # 选择轮廓颜色（如红色）
    # # ax.spines['polar'].set_linewidth(2)  # 轮廓线宽度
    #
    # # 在每个柱子上方显示数值，并留一定距离
    # for i, bar in enumerate(bars):
    #     height = bar.get_height() + base_value  # 高度是柱子的实际值
    #     # ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,  # 0.05 是与柱子的距离
    #     #         f'{values[i]:.3f}', ha='center', va='bottom', fontsize=8, color='black')
    #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,  # 0.05 是与柱子的距离
    #             f'{values[i]:.3f}', ha='center', va='center_baseline', fontsize=8, color='black')
    #
    #     # ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,  # 0.05 是与柱子的距离
    #     #         f"{categories[i]}\n{values[i]:.3f}", ha='center', va='bottom', fontsize=12, color='black')
    #
    # # 设置图例，颜色对应每个类别
    # legend_labels = categories  # 图例标签
    # legend_patches = [plt.Line2D([0], [0], color=colors[i], marker='o', markersize=10, linestyle='None') for i in
    #                   range(len(categories))]
    # # 在右下角显示图例
    # # plt.legend(legend_patches, legend_labels, loc='lower right', fontsize=10, title_fontsize='10')
    # plt.legend(legend_patches, legend_labels, loc='upper right', fontsize=9)
    #
    # # # 在左上方添加新的y轴
    # # # 创建一个新的坐标轴
    # # new_ax = plt.axes([0.15, 0.85, 0.01, 0.15])  # [左, 下, 宽, 高]
    # # new_ax.set_ylim(min_val, 1)  # 设置y轴范围
    # # # # 绘制竖线
    # # # new_ax.vlines(x=0, ymin=0.5, ymax=1, color='black', linewidth=2)
    # # # 设置y轴刻度和标签
    # # new_ax.set_yticks(np.arange(min_val, 1, 0.1))  # 设置y轴刻度
    # # new_ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(min_val, 1, 0.1)], fontsize=8)  # 设置y轴标签
    # # # 隐藏x轴
    # # new_ax.xaxis.set_visible(False)
    # # # 添加y轴标签
    # # new_ax.set_ylabel(filename, fontsize=15)
    # # ax.set_yticklabels([])  # 隐藏 y 轴刻度标签
    #
    # # # 增大图例字体，并将图例移到左上角
    # # ax.legend(['Dice'], loc='upper left', fontsize=16)  # 增大字体大小
    # plt.title(filename, fontsize=17, loc='left')
    # # 调整起始角度，使刻度从左上角开始
    # ax.set_theta_offset(np.pi / 2)
    # ax.set_theta_direction(-1)
    #
    # plt.tight_layout()
    # # 保存为png文件
    # # plt.savefig(file + '/plot_results/Polarbar_'+filename+'_2.png', dpi=600)
    # # 显示图形
    # plt.show()
    # #---------------------------------------
    # from matplotlib.colors import LinearSegmentedColormap, Normalize
    # import matplotlib.colorbar as colorbar
    # # 已知数据
    # categories = ['Scaden', 'Music', 'NNLS', 'CPM', 'CSx', 'Rescue']
    # values = [0.9871, 0.842, 0.6682, 0.6337, 0.8059, 0.9932]    # Pearson
    # # # values = [0.9879, 0.8641, 0.7339, 0.5741, 0.8506, 0.9936]    #Dice
    # # # values = [0.9556, 0.9063, 0.7229, 0.7368, 0.8988, 0.9728]    #Fidelity
    # # values = [0.99, 0.8855, 0.7388, 0.725, 0.8554, 0.9948]  # Cosine
    # # 取每个值的 -log10
    # # log_values = [-np.log2(v) for v in values]
    # # # log_values = [-np.log(v) / np.log(5) for v in values]
    # # log_values.append(log_values[0])  # 闭合值
    #
    # values.append(values[0])  # 闭合值
    #
    # # # 取每个值的 -log5
    # # log_values = [-np.log(v) / np.log(2) for v in values]
    # # log_values.append(log_values[0])  # 闭合值
    #
    # # 角度设置，将类别映射到角度
    # num_vars = len(categories)
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # angles += angles[:1]  # 使角度周期闭合
    # # 创建极坐标图
    # fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    # # 设置中心的最小值和半径范围
    # min_val = 0.4
    # ax.set_ylim(min_val, 1)
    #
    # # # 计算颜色的深浅变化，使用Blues渐变
    # # norm_values = (log_values - np.min(log_values)) / (np.max(log_values) - np.min(log_values))  # 归一化
    # # colors = plt.cm.Blues(norm_values)  # 颜色深浅变化
    #
    # # # 创建自定义淡蓝色渐变色彩映射
    # # colors_list = ['#CCE5FF', '#0066CC']  # 最浅到较深的蓝色
    # # custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors_list, N=100)
    # # # 归一化 log 值以应用到颜色渐变
    # # norm_values = (log_values - np.min(log_values)) / (np.max(log_values) - np.min(log_values))
    # # colors = custom_cmap(norm_values)  # 使用淡蓝色渐变
    #
    # # 创建自定义淡蓝色渐变色彩映射，最浅颜色调整为更明显的淡蓝色
    # colors_list = ['#99CCFF', '#0066CC']  # 从较浅的蓝色到较深的蓝色
    # custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors_list, N=100)
    # # 归一化 log 值以应用到颜色渐变
    # norm = Normalize(vmin=min(values), vmax=max(values))
    # colors = custom_cmap(norm(values))  # 使用新的淡蓝色渐变
    #
    # # 绘制柱状图
    # base_value = min_val  # 中心的基线值
    # # adjusted_values = [v - base_value for v in log_values]  # 相对基线的变化量
    # adjusted_values = [v - base_value for v in values]  # 相对基线的变化量
    # bars = ax.bar(angles[:-1], adjusted_values[:-1], bottom=base_value,
    #               color=colors[:-1], linewidth=2, width=0.4, align='edge')
    #
    # # 设置类别标签
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories, color='black', fontsize=14, horizontalalignment='center')
    # ax.tick_params(axis='x', pad=15)
    #
    # # 在每个柱子上方显示取 -log10 后的值，并留一定距离
    # for i, bar in enumerate(bars):
    #     height = bar.get_height() + base_value
    #     # ax.text(bar.get_x() + bar.get_width() / 2, height + 0.06,
    #     #         f'{log_values[i]:.3f}', ha='center', va='bottom', fontsize=8, color='black')
    #     # ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
    #     #         f'{categories[i]}\n{values[i]:.3f}', ha='center', va='bottom', fontsize=7, color='black')
    #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
    #             f'{categories[i]}\n{values[i]:.3f}', ha='center', va='center_baseline', fontsize=8, color='black')
    #
    # # 取消数据刻度显示
    # ax.set_yticks([])  # 隐藏 y 轴刻度
    # ax.xaxis.set_visible(False)
    # # 去掉圆圈内部的实线
    # ax.spines['polar'].set_visible(False)
    #
    # # # 设置图例为浅蓝色渐变对应的每个类别
    # # legend_patches = [plt.Line2D([0], [0], color=plt.cm.Blues(0.2 + 0.8 * i / (len(categories) - 1)),
    # #                              marker='o', markersize=10, linestyle='None') for i in range(len(categories))]
    # # plt.legend(legend_patches, categories, loc='upper right', fontsize=9)
    # plt.title(filename, fontsize=17, loc='left')
    #
    # # 设置图例为淡蓝色渐变代表值大小
    # # legend_vals = np.linspace(min(log_values), max(log_values), 5)
    # # legend_colors = [custom_cmap((val - min(log_values)) / (max(log_values) - min(log_values))) for val in legend_vals]
    #
    # # 创建一个颜色条来显示颜色变化表示的数值范围
    # sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    # # cbar.set_label('-log5(Value)', fontsize=10)
    # cbar.ax.tick_params(labelsize=8)
    #
    # # legend_vals = np.linspace(min(values), max(values), 5)
    # # legend_colors = [custom_cmap((val - min(values)) / (max(values) - min(values))) for val in legend_vals]
    # # legend_patches = [plt.Line2D([0], [0], color=color, marker='o', markersize=10, linestyle='None')
    # #                   for color in legend_colors]
    # # plt.legend(legend_patches, [f'{val:.2f}' for val in legend_vals], loc='upper right',
    # #            fontsize=10, title="-log5", title_fontsize='13')
    # # 调整起始角度，使刻度从左上角开始
    # ax.set_theta_offset(np.pi / 2)
    # ax.set_theta_direction(-1)
    #
    # plt.tight_layout()
    # # 保存图形
    # plt.savefig(file + '/plot_results/Polarbar_' + filename + '_log10.png', dpi=600)
    # plt.show()
    # --------------绘制混淆矩阵-----------------------------------
    # 文件路径配置
    # file_true_path = file1_path  # 替换为你的真实值文件路径
    # file_pred_path = file2_path  # 替换为你的预测值文件路径
    # output_png_path = file + '/plot_results/confusion_matrix' + filename + '_1.png'
    #
    # confusion_matrix_plot(file_true_path, file_pred_path, output_png_path)
    # # # # # ----------------------------------------------------
    # file = 'LUSC_IA3_BC3_10/property/others_pred_LUSC_IA3_BC3_10'
    # file = 'LUSC_IA3_BC3_10/20000/pred'
    # file = 'BBI/Intestine/marge/pred'
    file = 'LUSC_IA3_BC8_10/others_pred_LUSC_IA3_BC8_10/heatmap'
    # file = 'seurat_pbmc/pbmc_3k_8_2/pred/pred_all'
    # file = 'seurat_pbmc/pbmc_3k_8_2/pred/pred_03'
    # file = 'LUSC_IA3_BC3_10/all/others_pred_LUSC_IA3_BC3_10'

    file0_path = file + '/truth_data.txt'
    file1_path = file + '/out_predict_scaden.txt'
    file2_path = file + '/prediction_CSx.txt'
    # file3_path = file + '/prediction_cmp.txt'
    file4_path = file + '/prediction_music.txt'
    file5_path = file + '/prediction_nnls.txt'
    file6_path = file + '/prediction_res_4000.txt'
    # file7_path = file + '/prediction_DWLS.txt'
    file8_path = file + '/prediction_BayesPrism.txt'

    # file0_path = file + '/truth_data.txt'
    # file1_path = file + '/out_predict_scaden_01.txt'
    # file2_path = file + '/prediction_CSx_01.txt'
    # file3_path = file + '/prediction_cmp_01.txt'
    # file4_path = file + '/prediction_music_01.txt'
    # file5_path = file + '/prediction_nnls_01.txt'
    # file6_path = file + '/prediction_res_4000_01.txt'
    # file7_path = file + '/prediction_DWLS_01.txt'
    # file8_path = file + '/prediction_BayesPrism_01.txt'

    # file0_path = file + '/truth_data.txt'
    # file1_path = file + '/out_predict_scaden_02.txt'
    # file2_path = file + '/prediction_CSx_02.txt'
    # file3_path = file + '/prediction_cmp_02.txt'
    # file4_path = file + '/prediction_music_02.txt'
    # file5_path = file + '/prediction_nnls_02.txt'
    # file6_path = file + '/prediction_res_4000_02.txt'
    # file7_path = file + '/prediction_DWLS_02.txt'
    # file8_path = file + '/prediction_BayesPrism_02.txt'

    # file0_path = file + '/truth_data.txt'
    # file1_path = file + '/out_predict_scaden_03.txt'
    # file2_path = file + '/prediction_CSx_03.txt'
    # file3_path = file + '/prediction_cmp_03.txt'
    # file4_path = file + '/prediction_music_03.txt'
    # file5_path = file + '/prediction_nnls_03.txt'
    # file6_path = file + '/prediction_res_4000_03.txt'
    # file7_path = file + '/prediction_DWLS_03.txt'
    # file8_path = file + '/prediction_BayesPrism_03.txt'
    lie = 11

    # 文件路径列表
    # file_paths = [file1_path, file2_path, file3_path, file4_path, file5_path, file6_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file4_path, file5_path, file6_path]
    # file_paths = [file1_path, file2_path, file3_path, file4_path, file5_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file3_path, file5_path, file6_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file2_path, file4_path, file6_path]  # 根据需要添加更多文件路径
    # file_paths = [file1_path, file3_path, file4_path, file5_path]
    # file_paths = [file1_path, file3_path, file4_path, file5_path, file6_path]  # 根据需要添加更多文件路径
    file_paths = [file6_path, file1_path, file2_path, file4_path, file5_path]
    # file_paths = [file1_path, file2_path, file3_path, file4_path, file5_path, file6_path, file7_path, file8_path]
    # file_paths = [file1_path, file2_path, file3_path, file4_path, file7_path, file8_path, file6_path]
    # file_paths = [file1_path, file2_path, file4_path, file7_path, file8_path, file6_path]

    # file_paths = [file6_path, file1_path, file2_path, file4_path, file5_path, file8_path]
    # file_paths = [file6_path, file1_path, file2_path, file4_path, file5_path]

    # 存储每个文件对比的RMSE值
    all_rmse_values = []

    # 计算每个文件的RMSE值
    data1 = get_data(file0_path, lie)
    for file_path in file_paths:
        # data1 = get_data(file0_path)
        data2 = get_data(file_path, lie)  # 假设是对比同一文件的不同版本，如果是不同文件，更新为不同路径
        rmse_values = []
        for row1, row2 in zip(data1, data2):
            rmse_values.append(np.sqrt(mean_squared_error(row1, row2)))
        all_rmse_values.append(rmse_values)

    # # 绘制箱型图
    # plt.figure(figsize=(8, 4))
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'CornflowerBlue', 'Pink']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
    # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#E6A5D7', '#F5D86A']
    # # colors = ['#E64B35', '#4DBBD5', '#3C5488', '#F39B7F', '#E6A5D7', '#F5D86A']
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'CornflowerBlue']
    # boxprops = [dict(facecolor=color) for color in colors]
    # # 设置每个箱体的颜色
    # for patch, color in zip(plt.boxplot(all_rmse_values, patch_artist=True, flierprops=dict(marker='o', markersize=8))['boxes'], colors):
    #     patch.set_facecolor(color)
    # # plt.boxplot(all_rmse_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    # # plt.boxplot(all_rmse_values, patch_artist=True)
    #
    # # 设置X轴标签为文件名或其他合适的名称
    # # file_name=['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # file_name = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS']
    # # file_name = ['Scaden', 'CSx', 'MuSiC', 'Rescue']
    # # file_name = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # # file_name = ['scaden', 'cmp', 'music', 'nnls']
    # # file_name = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'DWLS', 'BayesPrism', 'Rescue']
    # # file_name = ['Scaden', 'CSx', 'MuSiC', 'DWLS', 'BayesPrism', 'Rescue']
    # # file_name = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS', 'BayesPrism']
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
    # plt.savefig(file+'/plot_results/rmse_boxplot_all_5.png', dpi=600)
    # # plt.savefig(file + '/plot_results/rmse_boxplot_all_unknow.png')
    # # plt.savefig('dataset/rmse_boxplot_all.png')
    #
    # # 显示图形
    # plt.show()
    # # # # ----------------------------------------------------
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
    # # colors = ['SkyBlue', 'Lavender', 'LightGoldenrodYellow', 'HotPink', 'MintCream', 'Magenta']
    # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
    # # colors = ['#E64B35', '#4DBBD5', '#3C5488', '#F39B7F', '#E6A5D7', '#F5D86A']
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
    # # file_name = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # file_name = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS']
    # # file_name = ['Scaden', 'CSx', 'MuSiC', 'Rescue']
    # # file_name = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS', 'BayesPrism']
    #
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
    # plt.savefig(file+'/plot_results/ccc_boxplot_all_5.png', dpi=600)
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
    # Pearson_val_list = []
    # Dice_val_list = []
    # cosine_val_list = []
    # fidelity_val_list = []
    # # jaccard_val_list = []
    # ccc_val_list = []
    #
    # for file_path in file_paths:
    #     Pearson_val_all, Dice_val_all, cosine_val_all, fidelity_val_all = simil(file0_path, file_path, lie)
    #
    #     Pearson_val_list.append(Pearson_val_all)
    #     Dice_val_list.append(Dice_val_all)
    #     cosine_val_list.append(cosine_val_all)
    #     fidelity_val_list.append(fidelity_val_all)
    #     # jaccard_val_list.append(jaccard_val_all)
    #
    #     df1 = pd.read_csv(file0_path, sep='\t')
    #     df2 = pd.read_csv(file_path, sep='\t')
    #     columns_to_use = df1.columns[1:lie]
    #     data1 = df1[columns_to_use]
    #     data2 = df2[columns_to_use]
    #     long_vector1 = data1.to_numpy().flatten()
    #     long_vector2 = data2.to_numpy().flatten()
    #     # 计算总体的 Lin's CCC
    #     overall_ccc = lins_ccc(long_vector1, long_vector2)
    #     ccc_val_list.append(overall_ccc)
    #
    # # 设置指标名称和颜色
    # # metrics = ['Pearson', 'Dice', 'Cosine', 'Fidelity']
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'Pink']
    # # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    # #
    # # metrics = ['Pearson', 'Dice', 'Cosine']
    # # colors = ['lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'Pink']
    # # file_names = ['scaden', 'CSx', 'cmp', 'nnls', 'res']
    #
    # # metrics = ['Pearson', 'Dice', 'Cosine']
    # print(ccc_val_list)
    # # metrics = ['CCC', 'Pearson', 'Dice', 'Fidelity', 'Cosine']
    # metrics = ['Pearson', 'Dice', 'Fidelity', 'Cosine']
    # # metrics = ['Dice', 'Fidelity']
    # # metrics = ['Dice', 'Cosine']
    # # metrics = ['Dice']
    # # colors = ['red', 'lightblue', 'lightgreen', 'lightcoral', 'CornflowerBlue', 'DarkOrange']
    # # colors = ['#E889BD', '#67C2A3', '#FC8A61', '#8EA0C9']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
    # # colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#E6A5D7', '#F5D86A']
    # colors = ['#E64B35', '#4DBBD5', '#3C5488', '#F39B7F', '#E6A5D7', '#F5D86A']
    # # file_names = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS']
    # # file_names = ['Scaden', 'CSx', 'MuSiC', 'Rescue']
    # # file_names = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'DWLS', 'BayesPrism', 'Rescue']
    # # file_names = ['Scaden', 'CSx', 'MuSiC', 'DWLS', 'BayesPrism', 'Rescue']
    # file_names = ['Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS', 'BayesPrism']
    # # file_names = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # # file_names = ['scaden', 'cmp', 'music', 'nnls', 'res']
    #
    # # 准备数据
    # # all_metrics = [Pearson_val_list, Dice_val_list, cosine_val_list, fidelity_val_list]
    # # all_metrics = [Pearson_val_list, Dice_val_list, cosine_val_list]
    # # all_metrics = [ccc_val_list, Pearson_val_list, Dice_val_list, fidelity_val_list, cosine_val_list]
    # all_metrics = [Pearson_val_list, Dice_val_list, fidelity_val_list, cosine_val_list]
    # # all_metrics = [Dice_val_list, fidelity_val_list]
    # # all_metrics = [Dice_val_list, cosine_val_list]
    # # all_metrics =[Dice_val_list]
    # num_files = len(file_paths)
    # num_metrics = len(metrics)
    #
    # # 设置柱状图的位置
    # # bar_width = 0.12
    # bar_width = 0.13
    # # index = np.arange(num_files)
    # index = np.arange(num_metrics)
    # # 绘制柱状图
    # # plt.figure(figsize=(18, 9))
    # # plt.figure(figsize=(10, 5))
    # plt.figure(figsize=(16, 7))
    #
    # for i in range(num_files):
    #     plt.bar(index + i * bar_width, [all_metrics[j][i] for j in range(num_metrics)], bar_width, label=file_names[i],
    #             color=colors[i % len(colors)])
    #
    # # 设置X轴标签为相似性指标
    # plt.xticks(index + bar_width * (num_files - 1) / 2, metrics, fontsize=15)
    # plt.yticks(fontsize=15)
    #
    # # # # 设置X轴标签为文件名或其他合适的名称
    # # plt.xticks(index + bar_width * (num_metrics - 1) / 2, file_names)
    #
    # # plt.xlabel('Files')
    # plt.ylabel('Values', fontsize=17)
    # # plt.title('Comparison of Different Metrics for Various Predictions')
    #
    # # 去掉图的外框
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # # plt.gca().spines['left'].set_visible(False)
    # # plt.gca().spines['bottom'].set_visible(False)
    #
    # # 将图例固定在左上角
    # # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=False, fontsize=10)
    # # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=False, fontsize=15)
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, frameon=False, fontsize=12)
    # # 设置图例，固定在左上角，并自适应布局
    # # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, frameon=False, fontsize=11)
    #
    # # # 设置图例位置在图表右侧
    # # plt.legend(loc='upper right', ncol=1, frameon=False, fontsize=11)
    #
    # # plt.xlim(-0.5)
    # plt.xlim(-0.6)
    #
    # # 自动调整布局以防止重叠
    # plt.tight_layout()
    #
    # # 保存柱状图为PNG文件，并设置较高的dpi以提高图像清晰度
    # plt.savefig(file + '/plot_results/simil_barplot_6.png', dpi=600, bbox_inches='tight')
    #
    # # 显示图形
    # plt.show()
    #-----------------------------------------------------------
    # # 数据和标签
    # # ccc_val_list = [0.9190093395894668, 0.81097207780171, 0.6585521176835647, 0.7992411578699008, 0.8218512587243072]
    # # labels = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
    #
    # # Nature 配色方案
    # # nature_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    # # 创建柱状图
    # fig, ax = plt.subplots(figsize=(16, 8))
    # bars = ax.bar(file_names, ccc_val_list, color=colors, edgecolor='black')
    #
    # # 设置标题和轴标签
    # # ax.set_title('CCC Values by Method', fontsize=16, fontweight='bold')
    # ax.set_ylabel('CCC Value', fontsize=17)
    # ax.set_ylim(0, 1)  # 设置 y 轴范围为 [0, 1]
    #
    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f'{height:.2f}', ha='center', fontsize=15)
    #
    # # 调整 x 轴标签旋转角度（如果需要）
    # # plt.xticks(rotation=45, fontsize=12)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    #
    # # 去掉图的外框
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    #
    # # 保存图像为 PNG 文件
    # plt.tight_layout()
    #
    # plt.savefig(file + '/plot_results/ccc_bar_chart.png', dpi=600, bbox_inches='tight')
    #
    # # 显示图像
    # plt.show()

    # -------------------------heatmap----------------------
    # 获取多个文件路径（假设你的文件是以 .txt 结尾的）
    # file_paths = glob.glob(file + '/*.txt')  # 替换为实际的文件目录
    # file_paths = [file0_path, file1_path, file2_path, file4_path, file5_path, file6_path]
    # file_paths = [file0_path, file6_path, file1_path, file2_path, file4_path, file5_path]
    # # file_paths = [file0_path, file6_path, file1_path, file2_path, file4_path]
    # # file_paths = [file0_path, file6_path, file1_path, file2_path, file4_path, file5_path, file8_path]
    # # file_name = ['Scaden', 'CSx', 'CPM', 'MuSiC', 'NNLS', 'Rescue']
    # file_name = ['Real', 'Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS']
    # # file_name = ['Real', 'Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS', 'BayesPrism']
    # # 设置子图网格大小，假设每行放 2 张热图
    # num_files = len(file_paths)
    # # ncols = 6  # 每行2张热图
    # ncols = 3  # 每行2张热图
    # nrows = (num_files + 1) // ncols  # 确定行数
    #
    # # 创建子图
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, nrows * 9))  # 每张图宽度6，高度6
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    #
    # # 循环读取文件并绘制热图
    # for i, file_path in enumerate(file_paths):
    #     data = pd.read_csv(file_path, sep='\t', index_col=0)
    #     row = i // ncols
    #     col = i % ncols
    #     ax = axes[row, col] if nrows > 1 else axes[col]  # 选择当前子图
    #
    #     # 获取文件名作为标题
    #     # filename = file_path.split('/')[-1].split('.')[0]
    #     filename = file_name[i]
    #
    #     # 判断是否是最后一个子图，控制colorbar的显示
    #     show_cbar = (i == num_files - 1)
    #
    #     # from matplotlib.colors import LinearSegmentedColormap
    #     # colors = ['lightblue', 'red']  # 蓝色到红色渐变
    #     # cmap = LinearSegmentedColormap.from_list('blue_red', colors)
    #     # sns.heatmap(data, cmap=cmap, annot=False, vmin=0, vmax=0.4, ax=ax, cbar=show_cbar)
    #
    #     # sns.heatmap(data, cmap='coolwarm', annot=False, vmin=0, vmax=0.48, ax=ax, cbar=show_cbar)
    #     # sns.heatmap(data, cmap='coolwarm', annot=True, vmin=0, vmax=0.6, ax=ax, cbar=show_cbar)
    #     sns.heatmap(data, cmap='coolwarm', annot=True, ax=ax, cbar=show_cbar)
    #
    #     # 绘制热图
    #     # sns.heatmap(data, cmap='coolwarm', annot=False, vmin=0, vmax=0.7, ax=ax, cbar=show_cbar)
    #     # sns.heatmap(data, cmap='coolwarm', annot=False, vmin=0, vmax=1)
    #
    #     # 设置标题
    #     ax.set_title(filename, fontsize=18)
    #
    #     # 只在第一张子图上显示纵坐标轴名称（Sample）
    #     if i == 0:
    #         ax.set_ylabel('Sample', fontsize=16)
    #     else:
    #         ax.set_ylabel('')  # 其他子图不显示纵坐标轴名称
    #         ax.set_yticklabels([])
    #
    #     # 为所有子图设置横坐标轴名称（Celltype），但是不显示 Sample 标签
    #     # ax.set_xlabel('Celltype', fontsize=14)
    #
    #     # 控制刻度大小
    #     ax.tick_params(axis='x', labelsize=12)
    #     ax.tick_params(axis='y', labelsize=14)
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存成一个PNG文件
    # plt.savefig(file + '/plot_results/combined_heatmaps_5_1.png', dpi=600)
    #
    # # 展示图像（可选）
    # plt.show()

    # -----------------------------heatmap--------------------------------------------------
    # 文件路径和文件名
    file_paths = [file0_path, file6_path, file1_path, file2_path, file4_path, file5_path]
    file_name = ['Real', 'Rescue', 'Scaden', 'CSx', 'MuSiC', 'NNLS']

    # 设置子图网格大小，假设每行放 3 张热图
    num_files = len(file_paths)
    ncols = 3  # 每行3张热图
    nrows = (num_files + ncols - 1) // ncols  # 向上取整计算行数

    # 创建子图，启用自动布局
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, nrows * 5))

    # 存储所有热图的色图对象，用于添加一个整体的 colorbar
    heatmaps = []

    # 循环读取文件并绘制热图
    for i, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path, sep='\t', index_col=0)
        data = data.T  # 转置数据

        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        filename = file_name[i]

        # 设置渐变色
        colors = ['lightyellow', 'red']  # 从浅黄色到红色渐变
        cmap = LinearSegmentedColormap.from_list('yellow_red', colors)

        # 判断是否为每行的最后一个子图，控制是否显示 colorbar
        show_cbar = (col == ncols - 1)

        # 绘制热图
        heatmap = sns.heatmap(data, cmap=cmap, annot=True, vmin=0, vmax=0.7, ax=ax, cbar=show_cbar)
        heatmaps.append(heatmap)

        ax.set_title(filename, fontsize=18)

        # 纵坐标标签：只在每行第一个图显示
        if col == 0:
            ax.set_ylabel('Celltype', fontsize=16)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # 横坐标标签：只在最后一行显示
        if row == nrows - 1:
            ax.set_xlabel('Sample', fontsize=16)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)


    # 使用 constrained_layout 自动调整布局
    plt.tight_layout()

    # 保存成一个PNG文件
    plt.savefig(file + '/plot_results/combined_heatmaps_5_1.png', dpi=600)

    # 展示图像（可选）
    plt.show()

    print('over.....')

