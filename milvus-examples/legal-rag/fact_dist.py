import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 读取 jsonl 文件，每行一个 dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def process_file(file_path):
    lens = []
    long_facts_count = 0  # 长度大于8192的fact计数
    total_length = 0  # 总长度
    for item in read_jsonl(file_path):
        fact = item["fact"].strip()
        fact_len = len(fact)
        lens.append(fact_len)
        total_length += fact_len
        if fact_len > 8192:  # 修正: 计数长度大于8192的fact
            long_facts_count += 1
        # import re
        # # 匹配前面是汉字，后面是汉字或空格的 "."
        # match = re.search(r'[\u4e00-\u9fff]\.[\u4e00-\u9fff\s]', fact)
        # if match:
        #     start = max(0, match.start() - 10)
        #     end = min(len(fact), match.end() + 10)
        #     print(f"Found Chinese character with decimal point in: ...{fact[start:end]}...")
    return lens, long_facts_count, total_length

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fact_dist.py <directory_path>")
        return
        
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory")
        return
        
    all_lens = []
    total_long_facts = 0  # 所有长度大于8192的fact计数
    total_length = 0  # 所有fact的总长度
    
    # Process all jsonl files in the directory
    for file_path in Path(directory_path).glob('**/*.json'):
        print(f"Processing {file_path}")
        file_lens, file_long_facts, file_total_length = process_file(file_path)
        all_lens.extend(file_lens)
        total_long_facts += file_long_facts
        total_length += file_total_length
        
    if not all_lens:
        print("No data found in any files")
        return
        
    # 计算统计数据
    lens_array = np.array(all_lens)
    min_len = np.min(lens_array)
    max_len = np.max(lens_array)
    percentiles = np.percentile(lens_array, [1, 10, 50, 90, 99])
    average_length = total_length / len(all_lens)  # 计算平均长度

    # 打印统计信息
    print(f"处理文件总数: {len(list(Path(directory_path).glob('**/*.jsonl')))}")
    print(f"处理事实总数: {len(all_lens)}")
    print(f"事实总长度: {total_length}")
    print(f"事实平均长度: {average_length:.2f}")
    print(f"长度大于8192的事实数量: {total_long_facts}")
    print(f"长度大于8192的事实百分比: {total_long_facts/len(all_lens)*100:.2f}%")
    print(f"最小值: {min_len}")
    print(f"1% 分位数: {percentiles[0]}")
    print(f"10% 分位数: {percentiles[1]}")
    print(f"50% 分位数: {percentiles[2]}")
    print(f"90% 分位数: {percentiles[3]}")
    print(f"99% 分位数: {percentiles[4]}")
    print(f"最大值: {max_len}")

    # 设置每个柱子宽度为50
    bin_width = 50
    x_upper_limit = max(2000, np.percentile(lens_array, 99) * 1.2)
    bins = np.arange(0, x_upper_limit + bin_width, bin_width)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lens_array,
             bins=bins,
             alpha=0.7,
             color='skyblue',
             edgecolor='black')
    plt.title('Fact length distribution')
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.75)
    
    plt.xlim(0, x_upper_limit)
    plt.tight_layout()
    plt.savefig('fact_length_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()
