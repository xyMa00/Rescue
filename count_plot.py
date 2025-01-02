import matplotlib.pyplot as plt
import numpy as np
import re
# # # 文件路径
# # # file_nam = 'seurat_pbmc'
# # file_nam ='seurat_pbmc_2000'
# # file_path = file_nam+'/log/output_adata_pbmc_13689_00001.log'
# #
# # # 存储浮点数的数组
# # louvain_ari_values = []
# #
# # # 打开文件并逐行读取，使用 UTF-8 编码
# # with open(file_path, 'r', encoding='utf-8') as file:
# #     for line in file:
# #         if 'ccc_value_max:' in line:
# #             # 提取浮点数部分
# #             # louvain_ari_str = line.split('louvain_ari:')[1].split('.')[0].strip()
# #             louvain_ari_str = line.split('cccc_value_max:')[1].strip().rstrip('.')
# #             # 转换为浮点数，并存储在数组中
# #             louvain_ari_value = float(louvain_ari_str)
# #             louvain_ari_values.append(louvain_ari_value)
# #
# # # 生成 X 轴坐标数组，每个坐标是当前数据点数量的10倍
# # data_points = len(louvain_ari_values)
# # x_coordinates = np.arange(0, data_points * 5, 5)
# #
# #
# # # 绘制折线图，使用新生成的 X 轴坐标数组
# # # plt.plot(x_coordinates, louvain_ari_values, label='louvain_ari')
# # plt.plot(x_coordinates, louvain_ari_values)
# #
# #
# # # 设置标签和标题
# # plt.xlabel('epoch')
# # plt.ylabel('CCC Value')
# # # plt.title('ARI Values with different algorithm ')
# #
# # plt.title('ResPreP')
# # # plt.title('Scale ')
# #
# # plt.legend(['louvain_ari'], loc='lower right')
# # # plt.legend()
# #
# # # 保存折线图到当前目录
# # plt.savefig(file_nam+'/log/ccc_line_plot.png')
# #
# # # 显示折线图
# # plt.show()
# # ---------------------------------------------------------------------
# # 定义文件名和路径
# # file_names = ['seurat_pbmc', 'seurat_pbmc_2000', 'seurat_pbmc_5000']
# # # file_names = ['seurat_pbmc', 'seurat_pbmc_2000']
# # file_paths = [file_name + '/log/output_adata_pbmc_13689_00001.log' for file_name in file_names]
#
#
# # file_paths = ['LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_250_200.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_500_400.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_1000_800.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_2000_1600.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_4000_3200.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_6000_4800.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_8000_6400.log',
# #               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_10000_8000.log']
#
# # file_paths = ['LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_250_200.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_500_400.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_1000_800.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_2000_1600.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_4000_3200.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_6000_4800.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_8000_6400.log',
# #               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_10000_8000.log']
#
# # file_paths = ['hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_250_200.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_500_400.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_1000_800.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_2000_1600.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_4000_3200.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_5000_4000.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_6000_4800.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_8000_6400.log',
# #               'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_8000_6400.log']
#
# log_path = 'D:/cooperation/dataset/pbmc_4/'
# # file_paths = [log_path+'data6k/log/output_data_A_C_8_train_6000.log',
# #               log_path+'data8k/log/output_data_A_C_6_train_6000.log',
# #               log_path+'donorA/log/output_data_6_8_C_train_6000.log',
# #               log_path+'donorC/log/output_data_6_8_A_train_6000.log']
# # file_paths = [log_path+'sdy67/pred/use_1/log/output_combined_train_6_8_A_C_1.log']
# file_paths = [log_path+'data_6_8_A_C/log/output_combined_train_6_8_A_C_1.log']
# # file_paths = [log_path+'data8k/log/output_data_A_C_6_train_6000.log']
#
# # 初始化颜色列表
# colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'Tomato', 'HotPink', 'MediumOrchid', 'Lavender']
#
# # key = 'ccc_value_min: '
# # key = 'min_loss: '
# key = 'loss_test:'
# # key = 'loss:'
# get_num=110
#
# # 初始化图像
# plt.figure(figsize=(10, 4))
#
# # 循环处理每个文件
# for idx, file_path in enumerate(file_paths):
#     # 存储浮点数的数组
#     louvain_ari_values = []
#
#     # 打开文件并逐行读取，使用 UTF-8 编码
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             if key in line:
#                 # 提取浮点数部分
#                 # louvain_ari_str = line.split(key)[1].strip().rstrip('.')
#                 louvain_ari_str = line.split(key)[1].split(',')[0].strip()
#                 # # 使用正则表达式提取 'loss:' 后面的值
#                 # match = re.search(r'loss:(\d+\.\d+)', line)
#                 # if match:
#                 #     louvain_ari_str = match.group(1)
#                 #     # print(louvain_ari_str)
#                 # else:
#                 #     print("未找到 'loss:' 的值")
#             # if 'test_ccc_value:' in line:
#             #     # 提取浮点数部分
#             #     louvain_ari_str = line.split('test_ccc_value:')[1].strip().rstrip('.')
#                 # 转换为浮点数，并存储在数组中
#                 louvain_ari_value = float(louvain_ari_str)
#                 # louvain_ari_value = round(float(louvain_ari_str), 2)
#                 if louvain_ari_value ==0:
#                     continue
#                 louvain_ari_values.append(louvain_ari_value)
#             if len(louvain_ari_values)==get_num:
#                 break
#             # if file_path=='seurat_pbmc_5000/log/output_adata_pbmc_13689_00001.log' and len(louvain_ari_values)==151:
#             #     break
#
#         # for i in range(len(louvain_ari_values), 1000):
#         #     louvain_ari_values.append(louvain_ari_value)
#
#
#     # 生成 X 轴坐标数组，每个坐标是当前数据点数量的5倍
#     data_points = len(louvain_ari_values)
#
#     x_coordinates = np.arange(0, data_points * 5, 5)
#     # x_coordinates = np.arange(0, data_points)
#
#     # 绘制折线图，使用新生成的 X 轴坐标数组
#     # plt.plot(x_coordinates, louvain_ari_values, label=file_names[idx], color=colors[idx % len(colors)])
#     # label_names = ['sampling_2000', 'sampling_5000', 'sampling_10000']
#     # label_names = ['200', '400', '800', '1600', '3200', '4000', '4800', '6400', '8000']
#     # label_names = ['data6k_2000', 'data8k_2000', 'donorA_2000', 'donorC_2000']
#     label_names = ['data_6_8_A_C']
#     # label_names = ['data8k_2000']
#     # plt.plot(x_coordinates, louvain_ari_values, label=file_names[idx], color=colors[idx % len(colors)], linewidth=3)
#     plt.plot(x_coordinates, louvain_ari_values, label=label_names[idx], color=colors[idx % len(colors)], linewidth=3)
#
# # 设置标签和标题
# plt.xlabel('epoch')
# # plt.ylabel('CCC Value')
# plt.ylabel('Loss Value')
# plt.title('Rescue')
#
# # 显示图例，位置设为右下角
# # plt.legend(loc='lower right')
# plt.legend(loc='upper right')
#
# # 保存折线图到当前目录
# plt.savefig(log_path+'loss_line_plot_data_6_8_A_C.png')
# # plt.savefig(log_path+'loss_line_plot_pbmc_4_4.png')
# # plt.savefig(log_path+'ccc_line_plot_pbmc_4_4.png')
#
# # 显示折线图
# plt.show()


# -------------------------------
# 数据和标签
# x_data = ['10%', '20%', '30%']
# scaden_data = [0.9262, 0.5811, 0.179]
# csx_data = [0.6586, 0.4078, 0.1056]
# music_data = [0.877, 0.5031, 0.0964]
# nnls_data = [0.8375, 0.4228, 0.0174]
# rescue_data = [0.97, 0.9577, 0.7625]

# scaden_data = [0.0607, 0.1567, 0.2531]
# csx_data = [0.0505, 0.1448, 0.256]
# music_data = [0.0806, 0.1772, 0.274]
# nnls_data = [0.098, 0.1938, 0.2853]
# rescue_data = [0.0284, 0.1145, 0.2024]

# x_data = ['10%', '20%', '30%']
# scaden_data = [0.91, 0.56, 0.14]
# csx_data = [0.60, 0.37, 0.1]
# music_data = [0.87, 0.48, 0.11]
# # nnls_data = [0.8375, 0.4228, 0.0174]
# rescue_data = [0.93, 0.74, 0.44]

# scaden_data = [0.03, 0.06, 0.1]
# csx_data = [0.08, 0.1, 0.13]
# music_data = [0.04, 0.08, 0.12]
# # nnls_data = [0.098, 0.1938, 0.2853]
# rescue_data = [0.03, 0.06, 0.09]


x_data = ['100', '400', '800', '1600', '2000', '2400', '4000']
scaden_data = [0.9353, 0.9241, 0.9394, 0.934, 0.9274, 0.9274, 0.9249]
csx_data = [0.7915, 0.8594, 0.855, 0.8369, 0.8497, 0.8514, 0.8423]
music_data = [0.8365, 0.8639, 0.8776, 0.8818, 0.8947, 0.8863, 0.889]
rescue_data = [0.9411, 0.9584, 0.9594, 0.9583, 0.9618, 0.9584, 0.9563]

val='CCC'
# val='RMSE'

# 创建图表
plt.figure(figsize=(8, 5))

# 绘制折线图
# plt.plot(x_data, scaden_data, label='Scaden', color='#E64B35', linewidth=4, marker='o', markersize=10)
# plt.plot(x_data, csx_data, label='CSx', color='#4DBBD5', linewidth=4, marker='o', markersize=10)
# plt.plot(x_data, music_data, label='MuSiC', color='#00A087', linewidth=4, marker='o', markersize=10)
# # plt.plot(x_data, nnls_data, label='NNLS', color='#d62728', linewidth=4, marker='o', markersize=10)
# plt.plot(x_data, rescue_data, label='Rescue', color='#3C5488', linewidth=4, marker='o', markersize=10)

plt.plot(x_data, scaden_data, label='data6k', color='#E64B35', linewidth=4, marker='o', markersize=10)
plt.plot(x_data, csx_data, label='data8k', color='#4DBBD5', linewidth=4, marker='o', markersize=10)
plt.plot(x_data, music_data, label='donorA', color='#00A087', linewidth=4, marker='o', markersize=10)
plt.plot(x_data, rescue_data, label='donorC', color='#3C5488', linewidth=4, marker='o', markersize=10)


# 设置标签和标题
# plt.xlabel('Proportion of unknown cell types', fontsize=15, color='black', labelpad=20)
plt.xlabel('Size', fontsize=15, color='black', labelpad=20)
plt.ylabel(val, fontsize=15, color='black', labelpad=20)
plt.xticks(fontsize=15, color='black')
plt.yticks(fontsize=15, color='black')

# 显示图例
plt.legend(fontsize=12, loc='best')

# 调整边距
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)

# 保存图表为 PNG 文件
# plt.savefig('unk_plot_'+val+'_1.png', dpi=600)
plt.savefig('PBMC_plot_'+val+'_1.png', dpi=600)

# 显示图表
plt.show()



print("over.......")