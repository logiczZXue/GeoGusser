import os

def calculate_column_means(directory):
    column_sums = [0, 0, 0, 0]
    row_count = 0

    # 遍历文件夹中的所有 .txt 文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                # 确保数据格式正确
                if content:
                    values = list(map(float, content.split(',')))
                    if len(values) == 4:
                        for i in range(4):
                            column_sums[i] += values[i]
                        row_count += 1

    # 计算平均值
    column_means = [s / row_count for s in column_sums]
    return column_means

if __name__ == "__main__":
    directory = "data/llama_eval"  # 替换为你的文件夹路径
    averages = calculate_column_means(directory)
    print(f"各列的平均值分别为: {averages}")
