#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 数据处理脚本
功能：
1. 读取当前目录下的 CSV 文件
2. 删除"来源"列
3. 保留"案件类型编码"为 1（民事）的行
4. 删除"全文"中包含"撤诉"的行
5. 仅保留"案件名称"中包含"判决书"的行
6. 移除"全文"末尾的广告信息（马克数据网相关）
7. 将HTML实体编码替换为对应字符（包括 &#xa0;、&times;、&divide;、&hellip;、&ldquo;、&rdquo;、&middot;、&permil;、&mdash;）
8. 移除后面不为https的换行符，保留网址前的换行符
支持多进程并行处理多个文件
"""
import pandas as pd
import glob
import re
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
# HTML实体映射常量
HTML_ENTITIES = {
    "&#xa0;": " ",      # 不换行空格
    "&times;": "×",     # 乘号
    "&divide;": "÷",    # 除号
    "&hellip;": "……",    # 省略号
    "&ldquo;": "“",     # 左双引号
    "&rdquo;": "”",     # 右双引号
    "&middot;": "·",    # 中点
    "&permil;": "‰",    # 千分号
    "&mdash;": "——"      # 长破折号
}
def remove_newlines_except_before_https(text):
    """
    移除后面不为 https 的换行符
    
    Args:
        text (str): 输入文本
    
    Returns:
        str: 处理后的文本
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 使用正则表达式：匹配换行符，但不匹配后面紧跟 https 的换行符
    # (?!\s*https) 是负向先行断言，表示后面不跟 https（可能有空白字符）
    cleaned_text = re.sub(r'\n(?!\s*https)', '', text)
    
    return cleaned_text
def remove_advertisements(text):
    """
    移除文本末尾的广告信息
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 定义广告模式的正则表达式
    # 匹配包含马克数据网相关信息的广告
    ad_patterns = [
        # 匹配各种前缀 + 马克数据网的模式
        r'\s*(百度搜索|来源：?|关注公众号|来自：?|微信公众号|关注微信公众号|更多数据：?搜索).*?(马\s*克\s*数\s*据\s*网|www\.macrodatas\.cn).*?$',
        # 匹配带引号的马克数据网
        r'\s*[""]马\s*克\s*[\s数]*据\s*网[""].*?$',
        # 匹配网址
        r'\s*来自：?www\.macrodatas\.cn.*?$',
        # 匹配复杂的组合模式（如：更多数据：搜索"马克数据网"来源：www.macrodatas.cn）
        r'\s*更多数据：?.*?马\s*克\s*数\s*据\s*网.*?来源：?.*?www\.macrodatas\.cn.*?$',
        # 匹配单独的微信公众号模式
        r'\s*(微信公众号|关注微信公众号).*?马\s*克\s*[\s数]*据\s*网.*?$',
        # 匹配单独的马克数据网（无前缀）
        r'\s*马\s*克\s*数\s*据\s*网\s*$',
        # 匹配简单的搜索模式
        r'\s*搜索.*?马\s*克\s*数\s*据\s*网.*?$',
        # 匹配更多数据 + 网址模式
        r'\s*更多数据：?\s*www\.macrodatas\.cn.*?$'
    ]
    
    # 逐个尝试匹配和移除广告
    cleaned_text = text
    for pattern in ad_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text.strip()
def check_column_and_log(df, column_name, csv_file, process_prefix):
    """
    检查列是否存在并记录日志
    
    Args:
        df: DataFrame
        column_name: 列名
        csv_file: 文件名
        process_prefix: 进程前缀
    
    Returns:
        bool: 列是否存在
    """
    if column_name in df.columns:
        return True
    else:
        print(f"{process_prefix} {csv_file} - 未找到'{column_name}'列")
        return False
def preprocess_single_csv_file(csv_file_and_id):
    """
    预处理单个CSV文件
    
    Args:
        csv_file_and_id (tuple): (CSV文件路径, 进程ID)
    
    Returns:
        dict: 处理结果统计信息
    """
    csv_file, process_id = csv_file_and_id
    result = {
        'file': csv_file,
        'success': False,
        'original_rows': 0,
        'final_rows': 0,
        'error': None,
        'process_id': process_id
    }
    
    try:
        process_prefix = f"[Process-{process_id}]" if process_id is not None else ""
        print(f"{process_prefix} 正在预处理文件: {csv_file}")
        
        # 读取 CSV 文件
        df = pd.read_csv(csv_file, encoding='utf-8')
        result['original_rows'] = len(df)
        print(f"{process_prefix} {csv_file} - 原始数据形状: {df.shape}")
        print(f"{process_prefix} {csv_file} - 列名: {list(df.columns)}")
        
        # 删除"来源"列（如果存在）
        if check_column_and_log(df, "来源", csv_file, process_prefix):
            df = df.drop("来源", axis=1)
            print(f"{process_prefix} {csv_file} - 已删除'来源'列")
        
        # 保留"案件类型编码"为 1（民事）的行
        if check_column_and_log(df, "案件类型编码", csv_file, process_prefix):
            initial_rows = len(df)
            df = df[df["案件类型编码"] == 1]
            removed_rows = initial_rows - len(df)
            print(f"{process_prefix} {csv_file} - 删除了 {removed_rows} 行（案件类型编码!=1）")
        
        # 删除"全文"中包含"撤诉"的行
        if check_column_and_log(df, "全文", csv_file, process_prefix):
            initial_rows = len(df)
            df = df[~df["全文"].str.contains("撤诉", na=False)]
            removed_rows = initial_rows - len(df)
            print(f"{process_prefix} {csv_file} - 删除了 {removed_rows} 行（全文包含'撤诉'）")
        
        # 仅保留"案件名称"中包含"判决书"的行
        if check_column_and_log(df, "案件名称", csv_file, process_prefix):
            initial_rows = len(df)
            df = df[df["案件名称"].str.contains("判决书", na=False)]
            remaining_rows = len(df)
            removed_rows = initial_rows - remaining_rows
            print(f"{process_prefix} {csv_file} - 仅保留包含'判决书'的案件，删除了 {removed_rows} 行，保留了 {remaining_rows} 行")
        
        # 移除"全文"末尾的广告信息和替换HTML实体编码
        if "全文" in df.columns:
            print(f"{process_prefix} {csv_file} - 正在移除'全文'中的广告信息...")
            original_lengths = df["全文"].str.len()
            df["全文"] = df["全文"].apply(remove_advertisements)
            new_lengths = df["全文"].str.len()
            
            # 统计移除了广告的行数
            ads_removed = (original_lengths != new_lengths).sum()
            print(f"{process_prefix} {csv_file} - 从 {ads_removed} 行中移除了广告信息")
            
            # 替换HTML实体编码
            print(f"{process_prefix} {csv_file} - 正在替换HTML实体编码...")
            
            # 逐个替换HTML实体
            total_replacements = 0
            for entity, replacement in HTML_ENTITIES.items():
                before_replace = df["全文"].copy()
                df["全文"] = df["全文"].str.replace(entity, replacement, regex=False)
                entity_replacements = (before_replace != df["全文"]).sum()
                if entity_replacements > 0:
                    total_replacements += entity_replacements
                    print(f"{process_prefix} {csv_file} - 在 {entity_replacements} 行中替换了 {entity}")
            
            print(f"{process_prefix} {csv_file} - 共在 {total_replacements} 行中替换了HTML实体编码")
            
            # 移除后面不为 https 的换行符
            print(f"{process_prefix} {csv_file} - 正在移除后面不为https的换行符...")
            before_newline_remove = df["全文"].copy()
            df["全文"] = df["全文"].apply(remove_newlines_except_before_https)
            newline_changes = (before_newline_remove != df["全文"]).sum()
            print(f"{process_prefix} {csv_file} - 在 {newline_changes} 行中移除了不必要的换行符")
        else:
            print(f"{process_prefix} {csv_file} - 未找到'全文'列，跳过广告移除、实体编码替换和换行符处理")
        
        result['final_rows'] = len(df)
        print(f"{process_prefix} {csv_file} - 预处理后数据形状: {df.shape}")
        
        # 保存处理后的文件
        output_file = "preprocessed_" + csv_file[0:4] + "_" + csv_file[5:7] + ".csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"{process_prefix} {csv_file} - 预处理后的文件已保存为: {output_file}")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f"{process_prefix} 预处理文件 {csv_file} 时出错: {str(e)}")
    
    return result
def preprocess_csv_files_parallel(max_workers=None):
    """
    并行预处理当前目录下的所有 CSV 文件
    
    Args:
        max_workers (int): 最大进程数，默认为None（由ProcessPoolExecutor自动决定）
    """
    # 查找当前目录下的所有 CSV 文件
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("当前目录下没有找到 CSV 文件")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件: {csv_files}")
    print(f"使用 {max_workers if max_workers else '自动'} 个进程进行并行处理")
    
    # 用于收集所有结果
    all_results = []
    
    # 为每个文件创建 (文件路径, 进程ID) 的元组
    file_tasks = [(csv_file, i+1) for i, csv_file in enumerate(csv_files)]
    
    # 使用进程池执行器
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(preprocess_single_csv_file, task): task[0] 
            for task in file_tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            csv_file = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                print(f'文件 {csv_file} 产生了异常: {exc}')
                all_results.append({
                    'file': csv_file,
                    'success': False,
                    'error': str(exc),
                    'original_rows': 0,
                    'final_rows': 0,
                    'process_id': None
                })
    
    # 打印总结信息
    print("\n" + "="*50)
    print("预处理完成！总结信息：")
    print("="*50)
    
    successful_files = [r for r in all_results if r['success']]
    failed_files = [r for r in all_results if not r['success']]
    
    print(f"总文件数: {len(csv_files)}")
    print(f"成功预处理: {len(successful_files)}")
    print(f"预处理失败: {len(failed_files)}")
    
    if successful_files:
        total_original_rows = sum(r['original_rows'] for r in successful_files)
        total_final_rows = sum(r['final_rows'] for r in successful_files)
        print(f"总原始行数: {total_original_rows}")
        print(f"总处理后行数: {total_final_rows}")
        print(f"总删除行数: {total_original_rows - total_final_rows}")
    
    if failed_files:
        print("\n失败的文件:")
        for result in failed_files:
            print(f"  - {result['file']}: {result['error']}")
def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="并行预处理CSV文件的数据清洗脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明：
1. 删除"来源"列
2. 保留"案件类型编码"为 1（民事）的行
3. 删除"全文"中包含"撤诉"的行
4. 仅保留"案件名称"中包含"判决书"的行
5. 移除"全文"末尾的广告信息（马克数据网相关）
6. 将HTML实体编码替换为对应字符（包括 &#xa0;、&times;、&divide;、&hellip;、&ldquo;、&rdquo;、&middot;、&permil;、&mdash;）
7. 移除后面不为https的换行符，保留网址前的换行符
使用示例：
  python preprocess.py                # 使用默认进程数
  python preprocess.py --processes 4  # 使用4个进程
  python preprocess.py -j 8           # 使用8个进程
        """
    )
    
    parser.add_argument(
        '--processes', '-j',
        type=int,
        default=None,
        help='并行预处理的进程数量 (默认: 自动根据CPU核心数确定)'
    )
    
    args = parser.parse_args()
    
    print("开始并行预处理 CSV 文件...")
    preprocess_csv_files_parallel(max_workers=args.processes)
if __name__ == "__main__":
    main()
