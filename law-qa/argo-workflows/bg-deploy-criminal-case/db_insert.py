#!/usr/bin/env python3
"""
Script to insert criminal cases data into Milvus.
"""
import os
import json
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

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
PARENT_COLLECTION_NAME = os.environ.get("PARENT_COLLECTION_NAME")
PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", 4096))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", 0))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 32))
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
CHAT_BASE_URL = os.environ.get("CHAT_BASE_URL")
CHAT_MODEL = os.environ.get("CHAT_MODEL")
IS_PARENT_CHILD = os.environ.get("IS_PARENT_CHILD", True)
IS_LLM_EXTRACT = os.environ.get("IS_LLM_EXTRACT", True)
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", 4))

# Log environment variables
logger.info("=== insert_data_cases.py environment ===")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"PARENT_COLLECTION_NAME: {PARENT_COLLECTION_NAME}")
logger.info(f"PARENT_CHUNK_SIZE: {PARENT_CHUNK_SIZE}")
logger.info(f"PARENT_CHUNK_OVERLAP: {PARENT_CHUNK_OVERLAP}")
logger.info(f"COLLECTION_NAME: {COLLECTION_NAME}")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
logger.info(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
logger.info(f"EMBEDDING_BASE_URL: {EMBEDDING_BASE_URL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"CHAT_BASE_URL: {CHAT_BASE_URL}")
logger.info(f"CHAT_MODEL: {CHAT_MODEL}")
logger.info(f"IS_PARENT_CHILD: {IS_PARENT_CHILD}")
logger.info(f"IS_LLM_EXTRACT: {IS_LLM_EXTRACT}")
logger.info(f"LLM_WORKERS: {LLM_WORKERS}")
logger.info(f"LOG_FILE: {log_file}")

BATCH_SIZE = 1000



# 读取 jsonl 文件，每行一个 dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\r\n", "\n", "。", "；", "，", "、"],
        keep_separator="end")
    return text_splitter.split_text(text)


def extract_metadata_with_llm(chunk, client):
    system_prompt = "你是一名精确的法律信息提取助手，擅长从刑事案件描述中提取关键元素。"

    # 定义所有prompt模板
    prompts = {
        "dates":
        f"""
你的任务是从案件描述片段中提取所有出现的日期。你的回答应为一个字符串，其中包含所有日期，用换行符分隔：

- 格式："YYYYMMDD\nYYYYMM\nYYYY"
- 示例："20190101\n20191231\n202001"

要求：

- 同一日期出现多次的，只记录一次
- 如果在案件描述片段中找不到任何一个具体的日期，请返回："<none>"
- 不要提供推理过程，直接给出最终结果

案件描述片段如下：

<案件描述片段>
{chunk}
</案件描述片段>
""",
        "locations":
        f"""
你的任务是从案件描述片段中提取所有出现的地点。你的回答应为一个字符串，其中包含所有地点，用换行符分隔：

- 格式："地点1\n地点2\n地点3"
- 示例："杭州市淳安县千岛湖镇新安东路\n达州市通川区\n博罗县下河村"

要求：

— 提供尽量完整的地址，如有可能，精确到道路名、街道名、镇名、村名
- 禁止提取更具体的地标，如：门牌号、楼栋号、单位/企业名称（如工厂、超市、学校）、楼层信息、房间号等
- 如果在案件描述片段中找不到任何一个具体的地点，请返回："<none>"
- 不要提供推理过程，直接给出最终结果

案件描述片段如下：

<案件描述片段>
{chunk}
</案件描述片段>
""",
        "people":
        f"""
你的任务是从案件描述片段中提取所有出现的人物（自然人）。你的回答应为一个字符串，其中包含所有人物的姓名、职业和在当次审判中的角色，人物之间用换行符分隔，姓名、职业、角色之间用半角分号分隔：

- 格式："人物1;职业1;角色1\n人物2;职业2;角色2\n人物3;职业3;角色3"
- 示例："秦成然;幼儿园董事长;上诉人法定代表人\n苏利强;律师;原告委托诉讼代理人\n郭恒军;法官;审判长\n顾春;村委会主任;被上诉人法定代表人"

要求：

- 提取的人物必须是自然人，而不能是法人（公司企业、事业单位、政府部门等）
- 如果某项信息缺失，请用 "<unk>" 占位
- 如果在案件描述片段中找不到任何一个具体的人物，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

案件描述片段如下：

<案件描述片段>
{chunk}
</案件描述片段>
""",
        "numbers":
        f"""
你的任务是从案件描述片段中提取所有出现的数额。你的回答应为一个字符串，其中包含所有数额，用换行符分隔：

- 格式："数额1\n数额2\n数额3"
- 示例："168000元\n16.8万元\n98.95平方米"

要求：

- 保留万、亿等数量单位
- 保留元、米、平方米等度量单位
- 不保留千位分隔符
- 如果在案件描述片段中找不到任何一个具体的数额，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

案件描述片段如下：

<案件描述片段>
{chunk}
</案件描述片段>
""",
        "criminals_llm":
        f"""
你的任务是从案件描述片段中提取所有被告人。你的回答应为一个字符串，其中包含所有被告人的姓名，用换行符分隔：

- 格式："被告人姓名1\n被告人姓名2\n被告人姓名3"
- 示例1："王某甲"
- 示例2："胡某\n谢某"

要求：

- 不添加"被告"、"被告人"等前缀
- 提取的被告人必须是自然人，而不能是法人（公司企业、事业单位、政府部门等）
- 如果在案件描述片段中找不到任何一个被告人，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

案件描述片段如下：

<案件描述片段>
{chunk}
</案件描述片段>
"""
    }

    def extract_single_type(prompt_data):
        field_name, prompt = prompt_data
        max_retries = 0  # 总共5次尝试（初始1次 + 重试4次）
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": True
                        },
                    },
                    temperature=0.0)
                # data = response.choices[0].message.content.strip()
                data = response.choices[0].message.content.split("</think>\n\n")[-1].strip()

                if len(data) > 100:
                    data = data.split("：\n")[-1].split("\n\n")[-1]

                return field_name, data
            except Exception as e:
                if attempt < max_retries:
                    # 不是最后一次尝试，等待后重试
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s
                    print(
                        f"Attempt {attempt + 1} failed for {field_name}, retrying in {wait_time}s..."
                    )
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败，打印错误并返回默认值
                    print(
                        f"Error requesting LLM for {field_name} after {max_retries + 1} attempts: {e}"
                    )
                    return field_name, "<none>"

    # 并行执行所有5个LLM调用
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(extract_single_type, item): item
            for item in prompts.items()
        }

        for future in as_completed(futures):
            field_name, data = future.result()
            results[field_name] = data

        if len(results["locations"]) > 200:
            results["locations"] = "<none>"

        if len(results["people"]) > 600:
            results["people"] = "<none>"

        if len(results["numbers"]) > 400:
            results["numbers"] = "<none>"

        if len(results["criminals_llm"]) > 200:
            results["criminals_llm"] = "<none>"

    return results


def embed_with_embedding_model(chunks):
    client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
    response = client.embeddings.create(input=chunks, model=EMBEDDING_MODEL)
    return [data.embedding for data in response.data]


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=DATABASE_NAME)

    parent_col = None
    if IS_PARENT_CHILD:
        parent_fields = [
            FieldSchema(name="parent_id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=36),
            FieldSchema(name="fact_id", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR,
                        max_length=30000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,
                        dim=2)  # unused
        ]

        parent_schema = CollectionSchema(parent_fields)
        if utility.has_collection(PARENT_COLLECTION_NAME):
            Collection(PARENT_COLLECTION_NAME).drop()
        parent_col = Collection(PARENT_COLLECTION_NAME,
                                parent_schema,
                                consistency_level="Strong")
        index = {"index_type": "FLAT", "metric_type": "IP"}
        parent_col.create_index("vector", index)
        parent_col.load()

    fields = [
        FieldSchema(name="chunk_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=36),
        FieldSchema(name="fact_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="relevant_articles",
                    dtype=DataType.ARRAY,
                    element_type=DataType.INT64,
                    max_capacity=9),
        FieldSchema(name="accusation", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="punish_of_money", dtype=DataType.INT64),
        FieldSchema(name="criminals", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="imprisonment", dtype=DataType.INT64),
        FieldSchema(name="life_imprisonment", dtype=DataType.BOOL),
        FieldSchema(name="death_penalty", dtype=DataType.BOOL),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dense_dim),
    ]
    if IS_PARENT_CHILD:
        fields.insert(
            1,
            FieldSchema(name="parent_id",
                        dtype=DataType.VARCHAR,
                        max_length=36))
    if IS_LLM_EXTRACT:
        fields.extend([
            FieldSchema(name="dates", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="locations",
                        dtype=DataType.VARCHAR,
                        max_length=600),
            FieldSchema(name="people", dtype=DataType.VARCHAR,
                        max_length=1800),
            FieldSchema(name="numbers", dtype=DataType.VARCHAR,
                        max_length=500),
            FieldSchema(name="criminals_llm",
                        dtype=DataType.VARCHAR,
                        max_length=600)
        ])
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

    return col, parent_col


def record_generator(jsonl_path):
    for item in read_jsonl(jsonl_path):
        # 只在逗号前后都不是数字时才替换为中文逗号，避免影响数字格式（如1,000）
        fact = re.sub(r'(?<!\d),(?!\d)', '，', item["fact"]).replace(";", "；")
        meta = item["meta"]
        fact_id = str(uuid.uuid4())

        metadata = {
            "relevant_articles":
            [int(i) for i in set(meta["relevant_articles"])],
            "accusation": "\n".join(meta["accusation"]),
            "punish_of_money": meta["punish_of_money"],
            "criminals": "\n".join(meta["criminals"]),
            "imprisonment": meta["term_of_imprisonment"]["imprisonment"],
            "life_imprisonment":
            meta["term_of_imprisonment"]["life_imprisonment"],
            "death_penalty": meta["term_of_imprisonment"]["death_penalty"]
        }

        if IS_PARENT_CHILD:
            parent_chunks = chunk_text(
                fact, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP) if len(
                    fact) > PARENT_CHUNK_SIZE else [fact]
            for parent_chunk in parent_chunks:
                if len(parent_chunk) <= 3:
                    continue
                parent_id = str(uuid.uuid4())
                yield {
                    "parent_id": parent_id,
                    "fact_id": fact_id,
                    "chunk": parent_chunk,
                    "is_parent": True
                }

                chunks = chunk_text(
                    parent_chunk, CHUNK_SIZE, CHUNK_OVERLAP) if len(
                        parent_chunk) > CHUNK_SIZE else [parent_chunk]
                for chunk in chunks:
                    if len(chunk) <= 3:
                        continue

                    yield {
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": parent_id,
                        "fact_id": fact_id,
                        "chunk": chunk,
                        **metadata
                    }

        else:
            chunks = chunk_text(fact) if len(fact) > CHUNK_SIZE else [fact]
            for chunk in chunks:
                if len(chunk) <= 3:
                    continue

                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "fact_id": fact_id,
                    "chunk": chunk,
                    **metadata
                }


def process_llm_metadata_parallel(records, llm_client, max_workers=4):
    """
    并行处理LLM元数据提取
    
    Args:
        records: 需要处理的记录列表
        llm_client: OpenAI client
        max_workers: 最大线程数
    
    Returns:
        处理后的记录列表
    """

    def process_single_record(record):
        chunk = record["chunk"]
        extracted_metadata = extract_metadata_with_llm(chunk, llm_client)
        record.update(extracted_metadata)
        return record

    # 分离parent和非parent记录
    parent_records = [r for r in records if r.get("is_parent", False)]
    non_parent_records = [r for r in records if not r.get("is_parent", False)]

    # 如果没有非parent记录需要处理，直接返回原记录
    if not non_parent_records:
        return records

    # 并行处理非parent记录 - 使用map更简洁
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_non_parent = list(
            executor.map(process_single_record, non_parent_records))

    # 简单合并：parent记录 + 处理后的非parent记录
    return parent_records + processed_non_parent


def insert_batch(col, buffer, ef, parent_col):
    """插入一批数据到Milvus集合中"""
    if not buffer:
        return 0
    
    chunks = [r["chunk"] for r in buffer]
    sparse_embeddings = ef(chunks)["sparse"]
    dense_embeddings = embed_with_embedding_model(chunks)
        
    to_insert = [
        [r["chunk_id"] for r in buffer],
        [r["fact_id"] for r in buffer],
        [r["chunk"] for r in buffer],
        [r["relevant_articles"] for r in buffer],
        [r["accusation"] for r in buffer],
        [r["punish_of_money"] for r in buffer],
        [r["criminals"] for r in buffer],
        [r["imprisonment"] for r in buffer],
        [r["life_imprisonment"] for r in buffer],
        [r["death_penalty"] for r in buffer],
        sparse_embeddings,
        dense_embeddings,
    ]
    if parent_col:
        to_insert.insert(1, [r["parent_id"] for r in buffer])
    if IS_LLM_EXTRACT:
        to_insert.extend([[r["dates"] for r in buffer],
                          [r["locations"] for r in buffer],
                          [r["people"] for r in buffer],
                          [r["numbers"] for r in buffer],
                          [r["criminals_llm"] for r in buffer]])
    
    col.insert(to_insert)
    return len(buffer)


def insert_parent_batch(parent_col, parent_buffer):
    """
    插入parent记录批次
    
    Args:
        parent_col: parent集合
        parent_buffer: parent记录缓冲区
    
    Returns:
        int: 插入的记录数量
    """
    if not parent_buffer:
        return 0

    parent_col.insert([
        [r["parent_id"] for r in parent_buffer],  # parent_id
        [r["fact_id"] for r in parent_buffer],  # fact_id
        [r["chunk"] for r in parent_buffer],  # chunk
        [[0.0, 0.0] for _ in parent_buffer]  # vector
    ])
    return len(parent_buffer)


def insert_data_streaming(col,
                          parent_col,
                          record_iter,
                          ef,
                          llm_client,
                          max_workers=4):
    buffer = []
    parent_buffer = []
    total = 0
    total_parent = 0

    for record in tqdm(record_iter, desc="Inserting records"):
        if record.get("is_parent", False):
            parent_buffer.append(record)
            if len(parent_buffer) >= BATCH_SIZE:
                total_parent += insert_parent_batch(parent_col, parent_buffer)
                parent_buffer = []
        else:
            buffer.append(record)
            if len(buffer) >= BATCH_SIZE:
                # 并行处理LLM元数据提取
                if IS_LLM_EXTRACT:
                    print(
                        f"Processing LLM metadata for {len(buffer)} chunks with {max_workers} workers..."
                    )
                    buffer = process_llm_metadata_parallel(
                        buffer, llm_client, max_workers)

                total += insert_batch(col, buffer, ef, parent_col)
                buffer = []
    # 插入剩余部分
    if buffer:
        # 并行处理LLM元数据提取
        if IS_LLM_EXTRACT:
            print(
                f"Processing LLM metadata for {len(buffer)} chunks with {max_workers} workers..."
            )
            buffer = process_llm_metadata_parallel(buffer, llm_client,
                                                   max_workers)

        total += insert_batch(col, buffer, ef, parent_col)

    if parent_buffer:
        total_parent += insert_parent_batch(parent_col, parent_buffer)

    print(f"Inserted {total} records and {total_parent} parent records.")


def main():
    path = "/workspace/criminal-case/"

    jsonl_files = []

    if os.path.isdir(path):
        # 如果是目录，递归处理目录及子目录下所有的json文件
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('rest_data.json'):
                    continue
                if file.endswith('.json'):
                    jsonl_files.append(os.path.join(root, file))

        if not jsonl_files:
            print(
                f"No JSON files found in directory or subdirectories: {path}")
            return
        print(f"Found {len(jsonl_files)} JSON files to process")
    elif os.path.isfile(path):
        # 如果是文件，直接处理
        jsonl_files = [path]
    else:
        print(f"Path not found: {path}")
        return

    llm_client = OpenAI(base_url=CHAT_BASE_URL, api_key="dummy", timeout=60)

    import torch
    import torch_gcu
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
    dense_dim = ef.dim["dense"]
    col, parent_col = setup_milvus_collection(dense_dim)

    # 处理每个文件
    for jsonl_path in jsonl_files:
        print(f"Processing file: {jsonl_path}")
        insert_data_streaming(col, parent_col, record_generator(jsonl_path),
                              ef, llm_client, LLM_WORKERS)

    print("All files processed successfully")


if __name__ == "__main__":
    main()
