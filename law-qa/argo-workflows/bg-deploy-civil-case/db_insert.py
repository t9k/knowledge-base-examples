#!/usr/bin/env python3
"""
Script to insert civil cases data into Milvus.
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import pandas as pd
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



# 流式读取 csv 文件
def read_csv_streaming(file_path, chunksize):
    """
    流式读取CSV文件，避免大文件内存问题
    
    Args:
        file_path: CSV文件路径
        chunksize: 每次读取的行数
    
    Yields:
        dict: 每一行的数据字典
    """
    try:
        # 使用chunksize参数进行分块读取
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # 将每个chunk转换为字典列表，然后逐行yield
            for record in chunk.to_dict('records'):
                yield record
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return


# 解析法律依据文本，提取法律名称和条款
def parse_legal_basis(legal_text):
    if pd.isna(legal_text) or not legal_text:
        return {}

    result = {}
    # 分割多个法律依据
    legal_items = legal_text.replace("\n", "").split("；")

    for item in legal_items:
        item = item.strip()
        if not item:
            continue

        # 提取法律名称（去掉书名号）
        law_match = re.search(r'《([^》]+)》', item)
        if law_match:
            law_name = law_match.group(1)

            # 提取条款数字
            articles = []
            # 匹配"第X条"或"第X款"或"第X项"等模式
            article_matches = re.findall(r'第([零一二三四五六七八九十百千\d]+)条', item)
            for match in article_matches:
                article_num = chinese_to_arabic(match) if any(
                    c in match for c in '零一二三四五六七八九十百千') else int(match)
                if article_num:
                    articles.append(article_num)

            if articles:
                if law_name in result:
                    result[law_name].extend(articles)
                else:
                    result[law_name] = articles

    # 去重 + 排序
    for law_name in result:
        result[law_name] = sorted(list(set(result[law_name])))

    return result


# 将中文数字转换为阿拉伯数字
def chinese_to_arabic(chinese_num):
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


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\r\n", "\n", "。", "；", "，", "、", "："],
        keep_separator="end")
    return text_splitter.split_text(text)


def extract_metadata_with_llm(chunk, client):
    system_prompt = "你是一名精确的法律信息提取助手，擅长从刑事案件描述中提取关键元素。"

    # 定义所有prompt模板
    prompts = {
        "dates":
        f"""
你的任务是从裁判文书片段中提取所有出现的日期。你的回答应为一个字符串，其中包含所有日期，用换行符分隔：

- 格式："YYYYMMDD\nYYYYMM\nYYYY"
- 示例："20190101\n20191231\n202001"

要求：

- 同一日期出现多次的，只记录一次
- 如果在裁判文书片段中找不到任何一个具体的日期，请返回："<none>"
- 不要提供推理过程，直接给出最终结果

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
""",
        "locations":
        f"""
你的任务是从裁判文书片段中提取所有出现的地点。你的回答应为一个字符串，其中包含所有地点，用换行符分隔：

- 格式："地点1\n地点2\n地点3"
- 示例："杭州市淳安县千岛湖镇新安东路\n达州市通川区\n博罗县下河村"

要求：

— 提供尽量完整的地址，如有可能，精确到道路名、街道名、镇名、村名
- 禁止提取更具体的地标，如：门牌号、楼栋号、单位/企业名称（如工厂、超市、学校）、楼层信息、房间号等
- 如果在裁判文书片段中找不到任何一个具体的地点，请返回："<none>"
- 不要提供推理过程，直接给出最终结果

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
""",
        "people":
        f"""
你的任务是从裁判文书片段中提取所有出现的人物（自然人）。你的回答应为一个字符串，其中包含所有人物的姓名、职业和在当次审判中的角色，人物之间用换行符分隔，姓名、职业、角色之间用半角分号分隔：

- 格式："人物1;职业1;角色1\n人物2;职业2;角色2\n人物3;职业3;角色3"
- 示例："秦成然;幼儿园董事长;上诉人法定代表人\n苏利强;律师;原告委托诉讼代理人\n郭恒军;法官;审判长\n顾春;村委会主任;被上诉人法定代表人"

要求：

- 提取的人物必须是自然人，而不能是法人（公司企业、事业单位、政府部门等）
- 如果某项信息缺失，请用 "<unk>" 占位
- 如果在裁判文书片段中找不到任何一个具体的人物，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
""",
        "numbers":
        f"""
你的任务是从裁判文书片段中提取所有出现的数额。你的回答应为一个字符串，其中包含所有数额，用换行符分隔：

- 格式："数额1\n数额2\n数额3"
- 示例："168000元\n16.8万元\n98.95平方米"

要求：

- 保留万、亿等数量单位
- 保留元、米、平方米等度量单位
- 不保留千位分隔符
- 如果在裁判文书片段中找不到任何一个具体的数额，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
""",
        "parties_llm":
        f"""
你的任务是从裁判文书片段中提取所有出现的当事人。你的回答应为一个字符串，其中包含原告（上诉人）、被告（被上诉人）、原审原告、原审被告，用换行符分隔：

- 格式："原告1\n被告1\n原审原告1\n原审被告1"
- 示例："北京西联东升石材有限公司\n北京盛宇兴业建筑装饰工程有限公司\n北京西联东升石材有限公司\n北京盛宇兴业建筑装饰工程有限公司"

要求：

- 注意上诉人即为原告，被上诉人即为被告。上诉人（或被上诉人）可能是原审原告，也可能是原审被告
- 注意原告（或被告）可能是自然人，也可能是法人（公司、事业单位、政府部门等）
- 如果某项信息（如原审原告）缺失，请用 "<unk>" 占位
- 如果在裁判文书片段中找不到任何一个当事人，请直接返回："<none>"
- 不要提供推理过程，直接给出最终结果

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
"""
    }

    def extract_single_type(prompt_data):
        field_name, prompt = prompt_data
        max_retries = 4  # 总共5次尝试（初始1次 + 重试4次）
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
                            "enable_thinking": False
                        },
                    },
                    temperature=0.0)
                data = response.choices[0].message.content.strip()

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

    if len(results["people"]) > 500:
        results["people"] = "<none>"

    if len(results["numbers"]) > 400:
        results["numbers"] = "<none>"

    if len(results["parties_llm"]) > 200:
        results["parties_llm"] = "<none>"

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
            FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=36),
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
        FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="case_number", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="case_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="court", dtype=DataType.VARCHAR, max_length=90),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=45),
        FieldSchema(name="judgment_date",
                    dtype=DataType.VARCHAR,
                    max_length=17),
        FieldSchema(name="parties", dtype=DataType.VARCHAR, max_length=400),
        FieldSchema(name="case_cause", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="legal_basis", dtype=DataType.JSON),
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
            FieldSchema(name="dates", dtype=DataType.VARCHAR, max_length=700),
            FieldSchema(name="locations",
                        dtype=DataType.VARCHAR,
                        max_length=600),
            FieldSchema(name="people", dtype=DataType.VARCHAR, max_length=1500),
            FieldSchema(name="numbers", dtype=DataType.VARCHAR,
                        max_length=500),
            FieldSchema(name="parties_llm",
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


def record_generator(csv_path, chunksize):
    """
    生成处理后的记录，支持流式读取大CSV文件
    
    Args:
        csv_path: CSV文件路径
        chunksize: 每次读取的行数，用于控制内存使用
    """
    for item in read_csv_streaming(csv_path, chunksize=chunksize):
        if pd.isna(item["全文"]):
            continue
        # 只在逗号前后都不是数字时才替换为中文逗号，避免影响数字格式（如1,000）
        full_text = re.sub(r'(?<!\d),(?!\d)', '，',
                           item["全文"]).replace(";", "；")
        # 将连续的空白字符替换为单个空格
        full_text = re.sub(r'\s+', ' ', full_text)

        def get_metadata(key):
            if pd.isna(item[key]):
                return "<unk>"
            value = item[key]
            if "；" in value:
                return value.split("；")[0]
            return value

        # 解析元数据
        case_id = item["原始链接"].split("=")[-1]
        case_number = item["案号"]
        case_name = item["案件名称"]
        court = item["法院"]
        region = get_metadata("所属地区")
        judgment_date = "{}年{}月{}日".format(*map(int, item["裁判日期"].split("-")))
        parties = get_metadata("当事人").replace("；", "\n")
        case_cause = get_metadata("案由")
        legal_basis = parse_legal_basis(item["法律依据"])

        metadata = {
            "case_number": case_number,
            "case_name": case_name,
            "court": court,
            "region": region,
            "judgment_date": judgment_date,
            "parties": parties,
            "case_cause": case_cause,
            "legal_basis": legal_basis
        }

        if IS_PARENT_CHILD:
            parent_chunks = chunk_text(
                full_text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP) if len(
                    full_text) > PARENT_CHUNK_SIZE else [full_text]
            for parent_chunk in parent_chunks:
                if len(parent_chunk) <= 3:
                    continue
                parent_id = str(uuid.uuid4())
                yield {
                    "parent_id": parent_id,
                    "case_id": case_id,
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
                        "case_id": case_id,
                        "chunk": chunk,
                        **metadata
                    }

        else:
            chunks = chunk_text(full_text) if len(
                full_text) > CHUNK_SIZE else [full_text]
            for chunk in chunks:
                if len(chunk) <= 3:
                    continue

                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "case_id": case_id,
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
        [r["case_id"] for r in buffer],
        [r["chunk"] for r in buffer],
        [r["case_number"] for r in buffer],
        [r["case_name"] for r in buffer],
        [r["court"] for r in buffer],
        [r["region"] for r in buffer],
        [r["judgment_date"] for r in buffer],
        [r["parties"] for r in buffer],
        [r["case_cause"] for r in buffer],
        [r["legal_basis"] for r in buffer],
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
                          [r["parties_llm"] for r in buffer]])

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
        [r["case_id"] for r in parent_buffer],  # case_id
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
    path = "/workspace/civil-case/preprocessed_2021_07.csv"

    csv_files = []

    if os.path.isdir(path):
        # 如果是目录，递归处理目录及子目录下所有的csv文件
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        if not csv_files:
            print(f"No CSV files found in directory or subdirectories: {path}")
            return
        print(f"Found {len(csv_files)} CSV files to process")
    elif os.path.isfile(path):
        # 如果是文件，直接处理
        csv_files = [path]
    else:
        print(f"Path not found: {path}")
        return

    llm_client = OpenAI(base_url=CHAT_BASE_URL, api_key="dummy", timeout=60)

    import torch
    import torch_gcu
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
    print("Using Enflame GCU for embedding")
    dense_dim = ef.dim["dense"]
    col, parent_col = setup_milvus_collection(dense_dim)

    # 处理每个文件
    for csv_path in csv_files:
        print(f"Processing file: {csv_path}")
        insert_data_streaming(col, parent_col,
                              record_generator(csv_path, 100000),
                              ef, llm_client, LLM_WORKERS)

    print("All files processed successfully")


if __name__ == "__main__":
    main()
