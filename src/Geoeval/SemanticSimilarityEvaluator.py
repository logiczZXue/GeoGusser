import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 使用Sentence-BERT模型（你可以选择不同的预训练模型）
model = SentenceTransformer('all-MiniLM-L6-v2')  # 这个是一个常用的轻量级模型

def calculate_similarity(text1, text2):
    """
    使用Sentence-BERT计算两个文本的余弦相似度
    """
    # 编码文本
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    
    # 计算余弦相似度
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

def compare_folders(groundtruth_folder, test_folder):
    groundtruth_files = set(os.listdir(groundtruth_folder))
    test_files = set(os.listdir(test_folder))

    # 找到同名的txt文件
    common_files = groundtruth_files.intersection(test_files)
    common_files = [f for f in common_files if f.endswith(".txt")]

    similarities = []
    for filename in tqdm(common_files,total=len(common_files),desc="Calculating Similarity"):
        # 读取文件内容
        with open(os.path.join(groundtruth_folder, filename), 'r', encoding='utf-8') as f:
            groundtruth_text = f.read()
        
        with open(os.path.join(test_folder, filename), 'r', encoding='utf-8') as f:
            test_text = f.read()
        
        # 计算相似度
        similarity = calculate_similarity(groundtruth_text, test_text)
        similarities.append(similarity)

    if similarities:
        # 计算平均相似度
        avg_similarity = np.mean(similarities)
        print(f"Average Similarity Score for all compared files: {avg_similarity:.4f}")
    else:
        print("No common text files to compare.")

# 指定文件夹路径
groundtruth_folder = 'data/gt_description'
test_folder = 'data/llamas'

# 调用比较函数
compare_folders(groundtruth_folder, test_folder)
