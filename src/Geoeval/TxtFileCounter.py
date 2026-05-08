import os

def count_txt_files_in_folder(folder_path):
    # 初始化计数器
    txt_count = 0
    
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        # 获取文件扩展名
        file_extension = os.path.splitext(filename)[1].lower()
        
        # 检查是否是 .txt 文件
        if file_extension == '.txt':
            txt_count += 1
    
    return txt_count

# 示例：查询文件夹中的 .txt 文件数量
folder_path = 'data/im2gps3k_geos'  # 替换为你的文件夹路径
txt_count = count_txt_files_in_folder(folder_path)
print(f"文件夹 '{folder_path}' 中共有 {txt_count} 个 .txt 文件。")