import os
import shutil

# 定义文件夹路径
gt_folder = 'data/ground_truth'
geo_folder = 'data/geo'
output_folder = 'data/geo_s'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取GT文件夹中的所有文件名
gt_files = os.listdir(gt_folder)

# 遍历GT文件夹中的文件
for gt_file in gt_files:
    # 构建geo文件夹中对应文件的路径
    geo_file_path = os.path.join(geo_folder, gt_file)
    
    # 检查geo文件夹中是否存在同名文件
    if os.path.exists(geo_file_path):
        # 构建输出文件的路径
        output_file_path = os.path.join(output_folder, gt_file)
        
        # 将geo文件夹中的文件复制到输出文件夹
        shutil.copy(geo_file_path, output_file_path)
        print(f'Copied {gt_file} to {output_folder}')
    else:
        print(f'{gt_file} does not exist in {geo_folder}')

print('Done!')