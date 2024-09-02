import matplotlib.pyplot as plt
import numpy as np

# # 文件路径
# # file_nam = 'seurat_pbmc'
# file_nam ='seurat_pbmc_2000'
# file_path = file_nam+'/log/output_adata_pbmc_13689_00001.log'
#
# # 存储浮点数的数组
# louvain_ari_values = []
#
# # 打开文件并逐行读取，使用 UTF-8 编码
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         if 'ccc_value_max:' in line:
#             # 提取浮点数部分
#             # louvain_ari_str = line.split('louvain_ari:')[1].split('.')[0].strip()
#             louvain_ari_str = line.split('cccc_value_max:')[1].strip().rstrip('.')
#             # 转换为浮点数，并存储在数组中
#             louvain_ari_value = float(louvain_ari_str)
#             louvain_ari_values.append(louvain_ari_value)
#
# # 生成 X 轴坐标数组，每个坐标是当前数据点数量的10倍
# data_points = len(louvain_ari_values)
# x_coordinates = np.arange(0, data_points * 5, 5)
#
#
# # 绘制折线图，使用新生成的 X 轴坐标数组
# # plt.plot(x_coordinates, louvain_ari_values, label='louvain_ari')
# plt.plot(x_coordinates, louvain_ari_values)
#
#
# # 设置标签和标题
# plt.xlabel('epoch')
# plt.ylabel('CCC Value')
# # plt.title('ARI Values with different algorithm ')
#
# plt.title('ResPreP')
# # plt.title('Scale ')
#
# plt.legend(['louvain_ari'], loc='lower right')
# # plt.legend()
#
# # 保存折线图到当前目录
# plt.savefig(file_nam+'/log/ccc_line_plot.png')
#
# # 显示折线图
# plt.show()
# ---------------------------------------------------------------------

# 定义文件名和路径
# file_names = ['seurat_pbmc', 'seurat_pbmc_2000', 'seurat_pbmc_5000']
# # file_names = ['seurat_pbmc', 'seurat_pbmc_2000']
# file_paths = [file_name + '/log/output_adata_pbmc_13689_00001.log' for file_name in file_names]


# file_paths = ['LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_250_200.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_500_400.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_1000_800.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_2000_1600.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_4000_3200.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_6000_4800.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_8000_6400.log',
#               'LUSC_10_types/marge_20000/train_dataset/log/output_adata_LUSC_10_train_17956_10000_8000.log']

# file_paths = ['LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_250_200.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_500_400.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_1000_800.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_2000_1600.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_4000_3200.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_6000_4800.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_8000_6400.log',
#               'LUSC_IA3_BC3_10/20000/need/train_dataset/log/output_adata_LUSC_IA3_BC3_10_train_17956_10000_8000.log']

file_paths = ['hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_250_200.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_500_400.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_1000_800.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_2000_1600.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_4000_3200.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_5000_4000.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_6000_4800.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_8000_6400.log',
              'hscr/all_2_seurat_object_common_8/20000/train_dataset/log/output_adata_HSCR_8_train_2500_8000_6400.log']

# 初始化颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'Tomato', 'HotPink', 'MediumOrchid', 'Lavender']

# 初始化图像
plt.figure(figsize=(10, 4))

# 循环处理每个文件
for idx, file_path in enumerate(file_paths):
    # 存储浮点数的数组
    louvain_ari_values = []

    # 打开文件并逐行读取，使用 UTF-8 编码
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if 'ccc_value_max:' in line:
                # 提取浮点数部分
                louvain_ari_str = line.split('ccc_value_max:')[1].strip().rstrip('.')
            # if 'test_ccc_value:' in line:
            #     # 提取浮点数部分
            #     louvain_ari_str = line.split('test_ccc_value:')[1].strip().rstrip('.')
                # 转换为浮点数，并存储在数组中
                louvain_ari_value = float(louvain_ari_str)
                # louvain_ari_value = round(float(louvain_ari_str), 2)
                if louvain_ari_value ==0:
                    continue
                louvain_ari_values.append(louvain_ari_value)
            if len(louvain_ari_values)==1000:
                break
            # if file_path=='seurat_pbmc_5000/log/output_adata_pbmc_13689_00001.log' and len(louvain_ari_values)==151:
            #     break

        for i in range(len(louvain_ari_values), 1000):
            louvain_ari_values.append(louvain_ari_value)


    # 生成 X 轴坐标数组，每个坐标是当前数据点数量的5倍
    data_points = len(louvain_ari_values)

    # x_coordinates = np.arange(0, data_points * 5, 5)
    x_coordinates = np.arange(0, data_points)

    # 绘制折线图，使用新生成的 X 轴坐标数组
    # plt.plot(x_coordinates, louvain_ari_values, label=file_names[idx], color=colors[idx % len(colors)])
    # label_names = ['sampling_2000', 'sampling_5000', 'sampling_10000']
    label_names = ['200', '400', '800', '1600', '3200', '4000', '4800', '6400', '8000']
    # plt.plot(x_coordinates, louvain_ari_values, label=file_names[idx], color=colors[idx % len(colors)], linewidth=3)
    plt.plot(x_coordinates, louvain_ari_values, label=label_names[idx], color=colors[idx % len(colors)], linewidth=3)

# 设置标签和标题
plt.xlabel('epoch')
plt.ylabel('CCC Value')
plt.title('Rescue')

# 显示图例，位置设为右下角
plt.legend(loc='lower right')

# 保存折线图到当前目录
plt.savefig('ccc_line_plot_HSCR_8_9.png')

# 显示折线图
plt.show()


print("over.......")