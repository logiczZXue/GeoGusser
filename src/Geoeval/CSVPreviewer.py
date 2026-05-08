# 预览csv文件
import csv
import sys
import json

def preview_csv(file_path, num_rows=5):
    """预览 CSV 文件的前几行，格式化为 CSV 风格"""
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # 打印前 num_rows 行数据
            preview_data = []
            for i, row in enumerate(reader):
                if i < num_rows:
                    preview_data.append(row)
                else:
                    break
            
            # 打印格式化后的 CSV 风格数据
            writer = csv.DictWriter(sys.stdout, fieldnames=preview_data[0].keys())
            writer.writeheader()
            writer.writerows(preview_data)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python preview_csv.py <csv_file_path> [num_rows]")
    else:
        csv_file_path = sys.argv[1]
        rows_to_preview = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        preview_csv(csv_file_path, rows_to_preview)