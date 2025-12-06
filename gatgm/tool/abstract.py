import csv
import argparse
import os

def extract_first_column(input_path: str, output_path: str, has_header: bool = True):
    """
    读取 CSV 文件，提取其第一列数据，并写入一个新的 CSV 文件。
    
    Args:
        input_path (str): 输入 CSV 文件路径 (例如: 1w.csv)。
        output_path (str): 输出 CSV 文件路径 (例如: 1w_smiles_only.csv)。
        has_header (bool): 输入文件是否包含标题行（第一行）。
    """
    first_column_data = []
    
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # 跳过表头
            if has_header:
                try:
                    # 读取第一行（表头）
                    header = next(reader)
                    print(f"检测到表头: {header}")
                    # 如果原文件有表头，新文件只保留 SMILES 相关的表头
                    first_column_data.append([header[0]]) 
                except StopIteration:
                    print(f"警告: 输入文件 {input_path} 为空。")
                    return
            
            # 读取数据行
            for row in reader:
                if row:  # 确保行不为空
                    first_column_data.append([row[0]])
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 写入新的文件
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(first_column_data)
        
    print(f"\n 成功提取第一列数据 ({len(first_column_data)} 行) 到 {output_path}")
    print(f"您现在可以使用 {output_path} 作为 --predict_path 进行预测。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="提取 CSV 文件的第一列数据。")
    parser.add_argument('input_path', type=str, 
                        help="要读取的输入 CSV 文件路径 (例如: 1w.csv)。")
    parser.add_argument('output_path', type=str, nargs='?', default='smiles_only.csv',
                        help="输出 CSV 文件路径。默认为 smiles_only.csv。")
    parser.add_argument('--no_header', action='store_true',
                        help="如果输入文件没有表头（标题行），请添加此参数。")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    extract_first_column(args.input_path, args.output_path, has_header=not args.no_header)