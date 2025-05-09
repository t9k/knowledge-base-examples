import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import matplotlib.pyplot as plt

# 配置
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32


# 读取 jsonl 文件，每行一个 dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=overlap,
                                                   separators=["。", "；", "，"],
                                                   keep_separator="end")
    return text_splitter.split_text(text)


def process_file(file_path):
    lens = []
    total_chunks = 0
    long_chunks = 0
    for item in read_jsonl(file_path):
        fact = item["fact"].strip().replace(",", "，").replace(";", "；")
        if len(fact) <= 30:
            continue
        chunks = chunk_text(fact) if len(fact) > CHUNK_SIZE else [fact]
        total_chunks += len(chunks)
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len <= 3:
                continue
            lens.append(chunk_len)
            if chunk_len > CHUNK_SIZE:
                long_chunks += 1
            if chunk_len == 1361:
                print(chunk)
    return lens, total_chunks, long_chunks


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunk_data.py <directory_path>")
        return
        
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory")
        return
        
    all_lens = []
    total_facts = 0
    total_chunks = 0
    total_long_chunks = 0
    
    # Process all jsonl files in the directory
    for file_path in Path(directory_path).glob('**/*.json'):
        print(f"Processing {file_path}")
        file_lens, file_chunks, file_long_chunks = process_file(file_path)
        all_lens.extend(file_lens)
        total_chunks += file_chunks
        total_long_chunks += file_long_chunks
        total_facts += 1
        
    if not all_lens:
        print("No data found in any files")
        return
        
    # 计算统计数据
    lens_array = np.array(all_lens)
    min_len = np.min(lens_array)
    max_len = np.max(lens_array)
    percentiles = np.percentile(lens_array, [1, 10, 50, 90, 99])

    # 打印统计信息
    print(f"处理文件总数: {len(list(Path(directory_path).glob('**/*.json')))}")
    print(f"处理事实总数: {total_facts}")
    print(f"处理数据块总数: {total_chunks}")
    print(f"平均每个事实的数据块数: {total_chunks/total_facts:.2f}")
    print(f"长度大于{CHUNK_SIZE}的数据块数量: {total_long_chunks}")
    print(f"长度大于{CHUNK_SIZE}的数据块百分比: {total_long_chunks/total_chunks*100:.2f}%")
    print(f"最小值: {min_len}")
    print(f"1% 分位数: {percentiles[0]}")
    print(f"10% 分位数: {percentiles[1]}")
    print(f"50% 分位数: {percentiles[2]}")
    print(f"90% 分位数: {percentiles[3]}")
    print(f"99% 分位数: {percentiles[4]}")
    print(f"最大值: {max_len}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    # 设置横轴每25一个柱子，从0到1000
    bins = np.arange(0, 1025, 25)  # 0, 25, 50, ..., 1000
    plt.hist(lens_array,
             bins=bins,
             alpha=0.7,
             color='skyblue',
             edgecolor='black')
    plt.title('Text chunk length distribution')
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(0, 1000)  # 设置横轴上限为 1000

    plt.tight_layout()
    plt.savefig('chunk_length_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()
