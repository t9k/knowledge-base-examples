import os
import re
import sys


def process_markdown_file(file_path):
    """处理Markdown文件，应用所有格式化要求。"""
    # 从文件名中提取数字（例如，"1-总则.md" -> 1）
    filename = os.path.basename(file_path)
    match = re.match(r'(\d+)-', filename)
    if not match:
        print(f"无法从文件名中提取数字：{filename}")
        return False

    number = int(match.group(1))
    chinese_number = arabic_to_chinese(number)

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 检查文件是否以要求的标题开头
    if not content.strip().startswith('# 中华人民共和国民法典'):
        print(f"文件不是以要求的标题开头：{file_path}")
        return False

    # 使用所有格式化要求处理内容
    modified_content = content

    # 1. 格式化第二个一级标题
    modified_content = format_second_heading(modified_content, chinese_number)

    # 2. 规范化"章"和"节"的标题级别
    modified_content = normalize_heading_levels(modified_content)

    # 3. 清理标题中的空格
    modified_content = clean_heading_spaces(modified_content)

    # 4. 合并"分编"与前面的"编"标题
    modified_content = merge_fenbian_headings(modified_content)

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

    print(f"成功格式化文件：{file_path}")
    return True


def arabic_to_chinese(num):
    """将阿拉伯数字转换为中文汉字。"""
    chinese_nums = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }

    # 对于1-9的简单数字
    if 1 <= num <= 9:
        return chinese_nums[str(num)]
    else:
        # 这是一个简化的转换，完整实现应处理更大的数字
        return ''.join(chinese_nums[d] for d in str(num))


def format_second_heading(content, chinese_number):
    """格式化第二个一级标题。"""
    # 查找第二个一级标题
    headings = re.findall(r'^# (.+)$', content, re.MULTILINE)
    if len(headings) < 2:
        return content

    # 将第二个一级标题替换为二级标题，并添加所需前缀
    second_heading = headings[1]
    old_heading = f"# {second_heading}"
    new_heading = f"## 第{chinese_number}编 {second_heading}"

    return content.replace(old_heading, new_heading, 1)


def normalize_heading_levels(content):
    """规范化"章"和"节"的标题级别。"""
    # 处理所有行
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # 匹配任何标题级别（例如，#，##，###等）
        heading_match = re.match(r'^(#+)\s+(.+)$', line)
        if heading_match:
            level_markers = heading_match.group(1)
            heading_text = heading_match.group(2)

            # 将包含"章"的标题规范化为三级标题
            if "章" in heading_text:
                lines[i] = f"### {heading_text}"
            # 将包含"节"的标题规范化为四级标题
            elif "节" in heading_text:
                lines[i] = f"#### {heading_text}"

    return '\n'.join(lines)


def clean_heading_spaces(content):
    """清理标题中的空格，仅在特定模式（第x编/章/节）周围保留空格。"""

    def process_heading(line):
        match = re.match(r'^(#+)\s+(.+)$', line)
        if not match:
            return line

        level = match.group(1)
        text = match.group(2)

        # 去掉全部空格
        text = re.sub(r'\s+', '', text)

        # 在特定模式周围添加空格（如果前后不是空格就加）
        text = re.sub(r'(第[一二三四五六七八九十百千万分]+编)', r' \1 ', text)
        text = re.sub(r'(第[一二三四五六七八九十百千万]+章)', r' \1 ', text)
        text = re.sub(r'(第[一二三四五六七八九十百千万]+节)', r' \1 ', text)

        # 清理多余空格
        text = re.sub(r'\s+', ' ', text.strip())

        return f'{level} {text}'

    lines = content.split('\n')
    lines = [process_heading(line) for line in lines]
    return '\n'.join(lines)


def merge_fenbian_headings(content):
    """合并"分编"标题与前面的"编"标题。"""
    lines = content.split('\n')
    i = 0

    # 存储最新的"编"标题文本和行号
    bian_heading = None
    bian_line_num = -1
    merged_flag = False

    while i < len(lines):
        # 检查这是否是标题
        heading_match = re.match(r'^(#+)\s+(.+)$', lines[i])
        if heading_match:
            level_markers = heading_match.group(1)
            heading_text = heading_match.group(2)

            # 如果这是一个"编"标题（但不是"分编"标题）
            if "编" in heading_text and "分编" not in heading_text:
                # 将其存储为最新的"编"标题
                bian_heading = heading_text
                bian_line_num = i

            # 如果这是一个"分编"标题，并且我们有之前的"编"标题
            elif "分编" in heading_text and bian_heading is not None:
                # 创建一个新的合并标题
                merged_heading = f"{level_markers} {bian_heading} {heading_text}"

                # 用合并的标题替换"分编"标题
                lines[i] = merged_heading

                # 设置合并标志
                merged_flag = True

        i += 1

    # 若至少合并一次，则删除"编"标题
    if merged_flag:
        lines[bian_line_num] = ''

    # 删除空行（我们删除原始"编"标题的地方）
    lines = [line for line in lines if line.strip() != '']

    return '\n\n'.join(lines)


def main():
    """主函数，用于处理文件。"""
    if len(sys.argv) < 2:
        print("用法：python formatter.py <文件路径或目录路径>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path) and path.endswith('.md'):
        process_markdown_file(path)
    elif os.path.isdir(path):
        # 处理目录中的所有markdown文件
        for filename in os.listdir(path):
            if filename.endswith('.md'):
                file_path = os.path.join(path, filename)
                process_markdown_file(file_path)
    else:
        print(f"无效的路径或不是markdown文件：{path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
