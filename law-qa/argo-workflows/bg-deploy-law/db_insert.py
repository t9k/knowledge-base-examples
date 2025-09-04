#!/usr/bin/env python3
"""
Script to insert law data into Milvus.
"""
import os
import uuid
import re
import sys
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import OpenAI

# Set up logging
# Check if LOG_FILE environment variable is set
log_file = os.environ.get("LOG_FILE")
if log_file:
    # When running in the workflow with tee redirecting output to a log file,
    # only use StreamHandler to avoid duplicate logs
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
else:
    # Default logging setup
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("/env-config/env.properties")
MILVUS_URI = os.environ.get("MILVUS_URI")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
FILE_PATH = os.environ.get("FILE_PATH")

# Log environment variables
logger.info("=== insert_data_cases.py environment ===")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"COLLECTION_NAME: {COLLECTION_NAME}")
logger.info(f"EMBEDDING_BASE_URL: {EMBEDDING_BASE_URL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"FILE_PATH: {FILE_PATH}")
logger.info(f"LOG_FILE: {log_file}")

BATCH_SIZE = 1000



# 读取 markdown 文件
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# 将中文数字转换为阿拉伯数字
def chinese_to_arabic(chinese_str):
    chinese_numerals = {
        '零': 0,
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
        '百': 100,
        '千': 1000
    }

    # 如果是"第xx条"格式
    if '第' in chinese_str and '条' in chinese_str:
        # 提取中文数字部分
        match = re.search(r'第([零一二三四五六七八九十百千]+)条', chinese_str)
        if not match:
            return None

        chinese_num = match.group(1)
    # 如果是"xx、"格式
    elif '、' in chinese_str:
        match = re.search(r'([一二三四五六七八九十百千]+)、', chinese_str)
        if not match:
            return None

        chinese_num = match.group(1)
    else:
        return None

    # 转换逻辑
    result = 0
    temp = 0

    # 处理"十"开头的特殊情况，如"十一"应该是11
    if chinese_num.startswith('十'):
        result = 10
        chinese_num = chinese_num[1:]

    for char in chinese_num:
        if char in ['十', '百', '千']:
            # 处理单位
            if temp == 0:
                temp = 1
            result += temp * chinese_numerals[char]
            temp = 0
        else:
            # 处理数字
            temp = chinese_numerals[char]

    # 处理最后一个数字
    if temp != 0:
        result += temp

    return result


def chunk_text(text):
    # 判断是否存在“编”层级：有则按 law/part/chapter/section；无则按 law/chapter/section
    has_part_level = re.search(r"## 第[一二三四五六七八九十]编", text) is not None

    if has_part_level:
        headers_to_split_on = [
            ("#", "law"),
            ("##", "part"),
            ("###", "chapter"),
            ("####", "section"),
        ]
    else:
        headers_to_split_on = [
            ("#", "law"),
            ("##", "chapter"),
            ("###", "section"),
        ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=True)

    md_header_splits = md_splitter.split_text(text)

    # 手动实现基于分隔符的文本分割
    chunks = []

    # 正则表达式分隔符
    pattern = r'(第[零一二三四五六七八九十百千]+条 |[一二三四五六七八九十百千]+、|<!-- INFO END -->|<!-- FORCE BREAK -->)'

    for md_header_split in md_header_splits:
        text_content = md_header_split.page_content
        metadata = md_header_split.metadata.copy()

        # 查找所有匹配的分隔符位置
        matches = list(re.finditer(pattern, text_content))

        if not matches:
            # 如果没有匹配到分隔符，将整个文本作为一个块
            chunks.append(
                Document(page_content=text_content, metadata=metadata))
            continue

        # 处理第一个分隔符之前的文本（如果有）
        if matches[0].start() > 0:
            prefix_text = text_content[:matches[0].start()]
            chunks.append(Document(page_content=prefix_text,
                                   metadata=metadata))

        # 处理每个分隔符之间的文本
        for i in range(len(matches)):
            start_pos = matches[i].start()
            # 找出当前块的结束位置（下一个分隔符的起始位置或文本结束）
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(
                text_content)

            # 提取文本块（包含分隔符）
            chunk_text = text_content[start_pos:end_pos]

            # 如果是 <!-- INFO END --> 分隔符，不包含在文本中
            if chunk_text.startswith("<!-- INFO END -->"):
                chunk_text = chunk_text[len("<!-- INFO END -->"):].strip()

            # 提取法条编号
            article = 0
            article_amended = 0
            # 段首“第…条 ”作为当前条文编号
            leading_article_match = re.match(r'^第([零一二三四五六七八九十百千]+)条\s', chunk_text)
            if leading_article_match:
                article = chinese_to_arabic(leading_article_match.group(0))
            else:
                # 如果没找到法条编号，尝试查找列表项目符号
                item_match = re.search(r'([一二三四五六七八九十]+)、', chunk_text)
                if item_match:
                    article = chinese_to_arabic(item_match.group(0))
            # 全段扫描其他“第…条”作为被修正/引用的条文，忽略段首自身条号
            for m in re.finditer(r'第([零一二三四五六七八九十百千]+)条', chunk_text):
                if leading_article_match and m.start() == 0:
                    continue
                candidate = chinese_to_arabic(m.group(0))
                if candidate != article:
                    article_amended = candidate
                    break

            # 创建新的元数据，包含原始元数据和法条编号
            new_metadata = metadata.copy()
            new_metadata["article"] = article
            new_metadata["article_amended"] = article_amended

            chunks.append(
                Document(page_content=chunk_text, metadata=new_metadata))

    return chunks


def embed_with_embedding_model(chunks):
    client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
    response = client.embeddings.create(input=chunks, model=EMBEDDING_MODEL)
    
    return [data.embedding for data in response.data]


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=DATABASE_NAME)
    fields = [
        FieldSchema(name="chunk_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=12288),
        FieldSchema(name="law", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="part", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="article", dtype=DataType.INT64),
        FieldSchema(name="article_amended", dtype=DataType.INT64),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()
    col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 24,
            "efConstruction": 400
        }
    }
    col.create_index("dense_vector", dense_index)
    col.load()
    return col


def record_generator(md_path):
    text = read_markdown(md_path).strip()
    pages = chunk_text(text)

    for page in pages:
        chunk = page.page_content
        if len(chunk) <= 3:
            continue

        metadata = page.metadata
        metadata = {
            "law": metadata["law"],
            "part": metadata.get("part", ""),
            "chapter": metadata.get("chapter", ""),
            "section": metadata.get("section", ""),
            "article": metadata.get("article", 0),
            "article_amended": metadata.get("article_amended", 0),
        }

        yield {"chunk_id": str(uuid.uuid4()), "chunk": chunk, **metadata}


def insert_batch(col, buffer, ef):
    """插入一批数据到Milvus集合中"""
    if not buffer:
        return 0

    chunks = [r["chunk"] for r in buffer]
    sparse_embeddings = ef(chunks)["sparse"]
    dense_embeddings = embed_with_embedding_model(chunks)
    
    to_insert = [
        [r["chunk_id"] for r in buffer],
        [r["chunk"] for r in buffer],
        [r["law"] for r in buffer],
        [r["part"] for r in buffer],
        [r["chapter"] for r in buffer],
        [r["section"] for r in buffer],
        [r["article"] for r in buffer],
        [r["article_amended"] for r in buffer],
        sparse_embeddings,
        dense_embeddings,
    ]
    col.insert(to_insert)
    return len(buffer)


def insert_data_streaming(col, record_iter, ef, batch_size=BATCH_SIZE):
    buffer = []
    total = 0

    for record in tqdm(record_iter, desc="Inserting records"):
        buffer.append(record)
        if len(buffer) >= batch_size:
            total += insert_batch(col, buffer, ef)
            buffer = []
    # 插入剩余部分
    if buffer:
        total += insert_batch(col, buffer, ef)

    print(f"Inserted {total} records.")


def main():
    path = "/workspace/law/" + FILE_PATH.strip("/")

    md_files = []

    if os.path.isdir(path):
        # 如果是目录，递归处理目录及子目录下所有的md文件
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.md') and "法" in file:
                    md_files.append(os.path.join(root, file))

        if not md_files:
            print(
                f"No Markdown files found in directory or subdirectories: {path}"
            )
            return
        print(f"Found {len(md_files)} Markdown files to process")
    elif os.path.isfile(path):
        # 如果是文件，直接处理
        md_files = [path]
    else:
        print(f"Path not found: {path}")
        return

    import torch
    import torch_gcu
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
    print("Using Enflame GCU for embedding")
    dense_dim = ef.dim["dense"]
    col = setup_milvus_collection(dense_dim)

    # 处理每个文件
    for md_path in md_files:
        print(f"Processing file: {md_path}")
        insert_data_streaming(col, record_generator(md_path), ef)

    print("All files processed successfully")


if __name__ == "__main__":
    main()
