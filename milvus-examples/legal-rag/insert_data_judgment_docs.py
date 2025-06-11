import argparse
import os
import json
import uuid
import re
from tqdm import tqdm
import pandas as pd
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

# 配置
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
PARENT_COLLECTION_NAME = "civil_cases_parent"
PARENT_CHUNK_SIZE = 4096
PARENT_CHUNK_OVERLAP = 0
COLLECTION_NAME = "civil_cases"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 1000
CHAT_BASE_URL = "http://app-vllm-enflame-xxxxxxxx.namespace.svc.cluster.local/v1"
CHAT_MODEL = "Qwen2.5-72B-Instruct"
SYSTEM_PROMPT = "你是一名精确的法律信息提取助手，擅长从裁判文书中提取关键元素。"


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


# 解析当事人信息
def parse_parties(parties_text):
    if pd.isna(parties_text) or not parties_text:
        return []

    # 按分号分割当事人
    parties = [
        party.strip() for party in str(parties_text).split('；')
        if party.strip()
    ]
    return parties


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
        separators=["\r\n", "\n", "。", "；", "，", "、"],
        keep_separator="end")
    return text_splitter.split_text(text)


def extract_metadata_with_llm(chunk, client):
    """
    使用LLM从案件文本中提取时间、地点、人物和被告人信息
    """
    user_prompt = f"""
请从下面的裁判文书片段中提取关键信息，并严格按照要求返回。

裁判文书片段如下：

<裁判文书片段>
{chunk}
</裁判文书片段>
""" + """
提取项的详细说明如下：

1. dates（日期）：
   - 提取裁判文书片段中出现的所有明确日期。
   - 如有可能，精确到"日"级别，否则精确到"月"或"季节"级别。
   - 舍弃比"日"更精确的时间粒度，如时、分、秒、下午、晚上、左右。
   - 格式示例：["2019年1月1日", "2019年1月", "2019年春季"]。

2. locations（地点）：
   - 提取案件中提及的所有地点。
   - 提取层级要求如下：
     - 如有可能，保留至街道名、镇名、村名，否则保留至县名、区名或市名；
     - 禁止提取更具体的地标，如：门牌号、楼栋号、单位/企业名称（如工厂、超市、学校）、楼层信息、房间号等；
     - 举例说明：
       - 错误示例："南通市通州区五接镇江苏某某船舶重工有限公司1号宿舍楼"
       - 正确示例："南通市通州区五接镇"
       - 错误示例："杭州市淳安县千岛湖镇新安东路1120号美美食品商行"
       - 正确示例："杭州市淳安县千岛湖镇新安东路"
       - 错误示例："广州市南沙区大岗镇人民路宇航网吧门口"
       - 正确示例："广州市南沙区大岗镇人民路"
   - 格式示例：["杭州市淳安县千岛湖镇新安东路", "达州市通川区", "博罗县下河村", "桥东岸路"]。

3. people（人物）：
   - 提取所有在裁判文书片段中出现的人物（自然人）信息。
   - 尽可能提取：
     - 姓名（如有）；
     - 职业（如有，如董事长、股东、公司职员、村委会主任、律师、法律工作者、法官等）。
     - 在当次审判中扮演的角色（如有，如原告、原告负责人、原告法定代表人、原告委托诉讼代理人、被告、被告负责人、被告法定代表人、被告委托诉讼代理人、上诉人、上诉人法定代表人、被上诉人、被上诉人法定代表人等），注意这里不是人物的职业；
   - 如果某项信息缺失，请用特殊字符串 "<unknown>" 占位。
   - 注意提取的人物必须是自然人，而不能是法人（公司、事业单位、政府部门等）
   - 格式示例：
     [
       {"name": "秦成然", "role": "上诉人法定代表人", "occupation": "幼儿园董事长"},
       {"name": "苏利强", "role": "原告委托诉讼代理人", "occupation": "律师"},
       {"name": "郭恒军", "role": "审判长", "occupation": "法官"},
       {"name": "顾春", "role": "被上诉人法定代表人", "occupation": "村委会主任"}
     ]
    - 错误示例：
     [
       {"name": "长沙市建筑安装工程公司新疆分公司", "role": "上诉人", "occupation": "公司"}
     ]

4. numbers（数额）：
   - 提取裁判文书片段中出现的所有明确数额。
   - 不提取编号、日期等的数字，不提取文件的数量或页码
   - 允许保留万、亿等数量单位，允许保留元、米、平方米等衡量单位，不允许保留千位分隔符
   - 格式示例：["168000元", "16.8万元", "98.95平方米"]。

5. parties_llm（当事人）
   - 提取裁判文书片段中出现的所有明确当事人信息。
   - 注意上诉人即为原告，被上诉人即为被告。上诉人（或被上诉人）可能是原审原告，也可能是原审被告。
   - 原告（或被告）可能是自然人，也可能是法人（公司、事业单位、政府部门等）。
   - 如果某项信息缺失，请用特殊字符串 "<unknown>" 占位。
   - 格式示例：
    {
      "plaintiff": ["北京西联东升石材有限公司"],
      "defendant": ["北京盛宇兴业建筑装饰工程有限公司"],
      "original_plaintiff": ["北京西联东升石材有限公司"],
      "original_defendant": ["北京盛宇兴业建筑装饰工程有限公司"]
    }


请严格按照以下 JSON 格式返回提取结果（不要有任何其他文字解释）：

```json
{
  "dates": ["..."],
  "locations": ["..."],
  "people": [
    {"name": "...", "role": "...", "occupation": "..."}
  ],
  "numbers": ["..."],
  "parties_llm": {
    "plaintiff": ["...", "..."],
    "defendant": ["...", "..."],
    "original_plaintiff": ["...", "..."],
    "original_defendant": ["...", "..."]
  }
}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
    )

    # 提取JSON部分
    response_text = response.choices[0].message.content
    json_start = response_text.find(
        "```json") + 7 if "```json" in response_text else response_text.find(
            "{")
    json_end = response_text.rfind("}") + 1
    json_str = response_text[json_start:json_end].strip()

    # 解析JSON
    extracted_data = json.loads(json_str)
    people = extracted_data["people"]
    if people:
        people = {
            "names": [p["name"] for p in people],
            "roles": [p["role"] for p in people],
            "occupations": [p["occupation"] for p in people]
        }
    else:
        people = {
            "names": ["<unknown>"],
            "roles": ["<unknown>"],
            "occupations": ["<unknown>"]
        }

    return {
        "dates":
        extracted_data.get("dates", ["<unknown>"]),
        "locations":
        extracted_data.get("locations", ["<unknown>"]),
        "people":
        people,
        "numbers":
        extracted_data.get("numbers", ["<unknown>"]),
        "parties_llm":
        extracted_data.get(
            "parties_llm", {
                "plaintiff": ["<unknown>"],
                "defendant": ["<unknown>"],
                "original_plaintiff": ["<unknown>"],
                "original_defendant": ["<unknown>"]
            })
    }


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token="root:Milvus", db_name="default")

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
        FieldSchema(name="case_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="case_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="court", dtype=DataType.VARCHAR, max_length=90),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=45),
        FieldSchema(name="judgment_date",
                    dtype=DataType.VARCHAR,
                    max_length=17),
        FieldSchema(name="parties",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=100,
                    max_capacity=100),
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
            FieldSchema(name="dates",
                        dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=40,
                        max_capacity=20),
            FieldSchema(name="locations",
                        dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=150,
                        max_capacity=20),
            FieldSchema(name="people", dtype=DataType.JSON),
            FieldSchema(name="numbers",
                        dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=40,
                        max_capacity=20),
            FieldSchema(name="parties_llm", dtype=DataType.JSON)
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
        full_text = item["全文"].replace(",", "，").replace(";", "；")
        # 将连续的空白字符替换为单个空格
        full_text = re.sub(r'\s+', ' ', full_text)

        def get_metadata(key):
            if pd.isna(item[key]):
                return "<unknown>"
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
        parties = parse_parties(item["当事人"])
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
        chunk_id = record.get('chunk_id', 'unknown')
        max_retries = 4  # 总共5次尝试（初始1次 + 重试4次）
        
        for attempt in range(max_retries + 1):
            try:
                extracted_metadata = extract_metadata_with_llm(chunk, llm_client)
                record.update(extracted_metadata)
                return record
            except Exception as e:
                if attempt < max_retries:
                    # 不是最后一次尝试，等待后重试
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s
                    print(f"Chunk {chunk_id}: Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败，打印错误并返回默认值
                    print(f"Error processing chunk {chunk_id} after {max_retries + 1} attempts: {e}")
                    # 返回带有默认值的记录
                    record.update({
                        "dates": ["<unknown>"],
                        "locations": ["<unknown>"],
                        "people": {
                            "names": ["<unknown>"],
                            "roles": ["<unknown>"],
                            "occupations": ["<unknown>"]
                        },
                        "numbers": ["<unknown>"],
                        "parties_llm": {
                            "plaintiff": ["<unknown>"],
                            "defendant": ["<unknown>"],
                            "original_plaintiff": ["<unknown>"],
                            "original_defendant": ["<unknown>"]
                        }
                    })
                    return record

    # 使用线程池并行处理非parent记录
    processed_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_single_record, record): record
            for record in records
        }

        for future in as_completed(future_to_record):
            try:
                processed_record = future.result()
                processed_records.append(processed_record)
            except Exception as e:
                original_record = future_to_record[future]
                print(
                    f"Failed to process record {original_record.get('chunk_id', 'unknown')}: {e}"
                )
                # 使用原记录加默认值
                original_record.update({
                    "dates": ["<unknown>"],
                    "locations": ["<unknown>"],
                    "people": {
                        "names": ["<unknown>"],
                        "roles": ["<unknown>"],
                        "occupations": ["<unknown>"]
                    },
                    "numbers": ["<unknown>"],
                    "parties_llm": {
                        "plaintiff": ["<unknown>"],
                        "defendant": ["<unknown>"],
                        "original_plaintiff": ["<unknown>"],
                        "original_defendant": ["<unknown>"]
                    }
                })
                processed_records.append(original_record)

    # 合并parent记录和处理后的记录，保持原顺序
    all_records = []
    non_parent_iter = iter(processed_records)

    for record in records:
        if record.get("is_parent", False):
            all_records.append(record)
        else:
            all_records.append(next(non_parent_iter))

    return all_records


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
                parent_col.insert([
                    [r["parent_id"] for r in parent_buffer],  # parent_id
                    [r["case_id"] for r in parent_buffer],  # case_id
                    [r["chunk"] for r in parent_buffer],  # chunk
                    [[0.0, 0.0] for _ in parent_buffer]  # vector
                ])
                total_parent += len(parent_buffer)
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

                chunks = [r["chunk"] for r in buffer]
                embeddings = ef(chunks)
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
                    embeddings["sparse"],
                    embeddings["dense"],
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
                total += len(buffer)
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

        chunks = [r["chunk"] for r in buffer]
        embeddings = ef(chunks)
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
            embeddings["sparse"],
            embeddings["dense"],
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
        total += len(buffer)

    if parent_buffer:
        parent_col.insert([
            [r["parent_id"] for r in parent_buffer],  # parent_id
            [r["case_id"] for r in parent_buffer],  # case_id
            [r["chunk"] for r in parent_buffer],  # chunk
            [[0.0, 0.0] for _ in parent_buffer]  # vector
        ])
        total_parent += len(parent_buffer)

    print(f"Inserted {total} records and {total_parent} parent records.")


def main():
    parser = argparse.ArgumentParser(
        description='Insert legal case data into Milvus')
    parser.add_argument(
        'csv_path', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--use-gcu',
                        help='Use Enflame GCU for embedding',
                        action='store_true')
    parser.add_argument('--parent-child',
                        help='Use parent child',
                        action='store_true')
    parser.add_argument('--llm-extract',
                        help='Use LLM to extract metadata',
                        action='store_true')
    parser.add_argument(
        '--llm-workers',
        type=int,
        default=4,
        help='Number of threads for LLM processing (default: 4)')
    parser.add_argument(
        '--csv-chunksize',
        type=int,
        default=100000,
        help='CSV reading chunk size for memory efficiency (default: 100k)')
    args = parser.parse_args()

    path = args.csv_path

    global IS_PARENT_CHILD
    global IS_LLM_EXTRACT
    IS_PARENT_CHILD = args.parent_child
    IS_LLM_EXTRACT = args.llm_extract

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

    if args.use_gcu:
        import torch
        import torch_gcu
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
        print("Using Enflame GCU for embedding")
    else:
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        print("Using CPU for embedding")
    dense_dim = ef.dim["dense"]
    col, parent_col = setup_milvus_collection(dense_dim)

    # 处理每个文件
    for csv_path in csv_files:
        print(f"Processing file: {csv_path}")
        if IS_LLM_EXTRACT:
            print(f"Using {args.llm_workers} workers for LLM processing")
        insert_data_streaming(col, parent_col, record_generator(csv_path, args.csv_chunksize), ef,
                              llm_client, args.llm_workers)

    print("All files processed successfully")


if __name__ == "__main__":
    main()
