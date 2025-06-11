import argparse
import os
import json
import uuid
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

# 配置
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
PARENT_COLLECTION_NAME = "criminal_cases_parent"
PARENT_CHUNK_SIZE = 4096
PARENT_CHUNK_OVERLAP = 0
COLLECTION_NAME = "criminal_cases"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 1000
CHAT_BASE_URL = "http://app-vllm-enflame-xxxxxxxx.namespace.svc.cluster.local/v1"
CHAT_MODEL = "Qwen2.5-72B-Instruct"
SYSTEM_PROMPT = "你是一名精确的法律信息提取助手，擅长从刑事案件描述中提取关键元素。"


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
    """
    使用LLM从案件文本中提取时间、地点、人物和被告人信息
    """
    user_prompt = f"""
请从下面的案件描述中提取关键信息，并严格按照要求返回。

案件描述如下：

<案件描述>
{chunk}
</案件描述>
""" + """
提取项的详细说明如下：

1. date（日期）：
   - 提取案件中出现的所有明确日期。
   - 如有可能，精确到"日"级别，否则精确到"月"级别。
   - 舍弃比"日"更精确的时间粒度，如时、分、秒、下午、晚上、左右。
   - 格式示例：["2014年2月20日", "2015年12月29日"]。

2. location（地点）：
   - 提取案件中提及的所有地点。
   - 提取层级要求如下：
     - 如有可能，保留至街道名、镇名、村名，否则保留至县名、区名或市名；
     - 严格禁止提取更具体的地标，如：门牌号、楼栋号、单位/企业名称（如工厂、超市、学校）、楼层信息、房间号等；
     - 举例说明：
       - 错误示例："南通市通州区五接镇江苏某某船舶重工有限公司1号宿舍楼"
       - 正确示例："南通市通州区五接镇"
       - 错误示例："杭州市淳安县千岛湖镇新安东路1120号美美食品商行"
       - 正确示例："杭州市淳安县千岛湖镇新安东路"
       - 错误示例："广州市南沙区大岗镇人民路宇航网吧门口"
       - 正确示例："广州市南沙区大岗镇人民路"
   - 如果地点信息中含有企业名、门牌号、楼号、具体设施名称，请舍弃这些内容，仅保留上级行政地名或街道/村级地名。
   - 格式示例：["杭州市淳安县千岛湖镇新安东路", "达州市通川区", "博罗县下河村", "桥东岸路"]。

3. people（人物）：
   - 提取所有在案件中出现的人物信息。
   - 尽可能提取：
     - 姓名（如有）；
     - 职业（如有，如店主、店员、公交司机等）。
     - 在当次案件中扮演的角色（如有，如被告人、被害人、同伙、公安民警等），注意这里不是人物的职业；
   - 如果某项信息缺失，请用空字符串""占位。
   - 格式示例：
     [
       {"name": "唐某", "role": "被告人", "occupation": ""},
       {"name": "卢某某", "role": "被害人", "occupation": "店主"}
       {"name": "", "role": "", "occupation": "店员"}
     ]

4. defendant（被告人）：
   - 提取案件中的被告人姓名。
   - 仅填写姓名，不添加"被告""被告人"等前缀。
   - 示例： "唐某"

请严格按照以下 JSON 格式返回提取结果（不要有任何其他文字解释）：

```json
{
  "date": ["..."],
  "location": ["..."],
  "people": [
    {"name": "...", "role": "...", "occupation": "..."}
  ],
  "defendant": "..."
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
        "date": extracted_data.get("date", ["<unknown>"]),
        "location": extracted_data.get("location", ["<unknown>"]),
        "people": people,
        "defendant": extracted_data.get("defendant", "<unknown>")
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
        FieldSchema(name="accusation",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=100,
                    max_capacity=13),
        FieldSchema(name="punish_of_money", dtype=DataType.INT64),
        FieldSchema(name="criminals",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=10,
                    max_capacity=30),
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
            FieldSchema(name="dates",
                        dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=30,
                        max_capacity=20),
            FieldSchema(name="locations",
                        dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=100,
                        max_capacity=20),
            FieldSchema(name="people", dtype=DataType.JSON),
            FieldSchema(name="defendant",
                        dtype=DataType.VARCHAR,
                        max_length=10)
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
        fact = item["fact"].replace(",", "，").replace(";", "；")
        meta = item["meta"]
        fact_id = str(uuid.uuid4())

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
                    "chunk": parent_chunk
                }

                chunks = chunk_text(
                    parent_chunk, CHUNK_SIZE, CHUNK_OVERLAP) if len(
                        parent_chunk) > CHUNK_SIZE else [parent_chunk]
                for chunk in chunks:
                    if len(chunk) <= 3:
                        continue

                    metadata = {
                        "relevant_articles":
                        [int(i) for i in set(meta["relevant_articles"])],
                        "accusation":
                        meta["accusation"],
                        "punish_of_money":
                        meta["punish_of_money"],
                        "criminals":
                        meta["criminals"],
                        "imprisonment":
                        meta["term_of_imprisonment"]["imprisonment"],
                        "life_imprisonment":
                        meta["term_of_imprisonment"]["life_imprisonment"],
                        "death_penalty":
                        meta["term_of_imprisonment"]["death_penalty"]
                    }

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

                metadata = {
                    "relevant_articles":
                    [int(i) for i in set(meta["relevant_articles"])],
                    "accusation":
                    meta["accusation"],
                    "punish_of_money":
                    meta["punish_of_money"],
                    "criminals":
                    meta["criminals"],
                    "imprisonment":
                    meta["term_of_imprisonment"]["imprisonment"],
                    "life_imprisonment":
                    meta["term_of_imprisonment"]["life_imprisonment"],
                    "death_penalty":
                    meta["term_of_imprisonment"]["death_penalty"]
                }

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
        chunk_id = record.get('chunk_id', 'unknown')
        max_retries = 2  # 总共3次尝试（初始1次 + 重试2次）

        for attempt in range(max_retries + 1):
            try:
                extracted_metadata = extract_metadata_with_llm(
                    chunk, llm_client)
                record.update(extracted_metadata)
                return record
            except Exception as e:
                if attempt < max_retries:
                    # 不是最后一次尝试，等待后重试
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s
                    print(
                        f"Chunk {chunk_id}: Attempt {attempt + 1} failed, retrying in {wait_time}s..."
                    )
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败，打印错误并返回默认值
                    print(
                        f"Error processing chunk {chunk_id} after {max_retries + 1} attempts: {e}"
                    )
                    # 返回带有默认值的记录
                    record.update({
                        "dates": ["<unknown>"],
                        "locations": ["<unknown>"],
                        "people": {
                            "names": ["<unknown>"],
                            "roles": ["<unknown>"],
                            "occupations": ["<unknown>"]
                        },
                        "defendant": "<unknown>"
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
                    "defendant": "<unknown>"
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
                    [r["fact_id"] for r in parent_buffer],  # fact_id
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
                    [r["fact_id"] for r in buffer],
                    [r["chunk"] for r in buffer],
                    [r["relevant_articles"] for r in buffer],
                    [r["accusation"] for r in buffer],
                    [r["punish_of_money"] for r in buffer],
                    [r["criminals"] for r in buffer],
                    [r["imprisonment"] for r in buffer],
                    [r["life_imprisonment"] for r in buffer],
                    [r["death_penalty"] for r in buffer],
                    embeddings["sparse"],
                    embeddings["dense"],
                ]
                if parent_col:
                    to_insert.insert(1, [r["parent_id"] for r in buffer])
                if IS_LLM_EXTRACT:
                    to_insert.extend([[r["dates"] for r in buffer],
                                      [r["locations"] for r in buffer],
                                      [r["people"] for r in buffer],
                                      [r["defendant"] for r in buffer]])
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
            [r["fact_id"] for r in buffer],
            [r["chunk"] for r in buffer],
            [r["relevant_articles"] for r in buffer],
            [r["accusation"] for r in buffer],
            [r["punish_of_money"] for r in buffer],
            [r["criminals"] for r in buffer],
            [r["imprisonment"] for r in buffer],
            [r["life_imprisonment"] for r in buffer],
            [r["death_penalty"] for r in buffer],
            embeddings["sparse"],
            embeddings["dense"],
        ]
        if parent_col:
            to_insert.insert(1, [r["parent_id"] for r in buffer])
        if IS_LLM_EXTRACT:
            to_insert.extend([[r["dates"] for r in buffer],
                              [r["locations"] for r in buffer],
                              [r["people"] for r in buffer],
                              [r["defendant"] for r in buffer]])
        col.insert(to_insert)
        total += len(buffer)

    if parent_buffer:
        parent_col.insert([
            [r["parent_id"] for r in parent_buffer],  # parent_id
            [r["fact_id"] for r in parent_buffer],  # fact_id
            [r["chunk"] for r in parent_buffer],  # chunk
            [[0.0, 0.0] for _ in parent_buffer]  # vector
        ])
        total_parent += len(parent_buffer)

    print(f"Inserted {total} records and {total_parent} parent records.")


def main():
    parser = argparse.ArgumentParser(
        description='Insert legal case data into Milvus')
    parser.add_argument(
        'jsonl_path',
        help='Path to JSONL file or directory containing JSON files')
    parser.add_argument('--use-gcu',
                        help='Use Enflame GCU for embedding',
                        action='store_true')
    parser.add_argument('--parent_child',
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
    args = parser.parse_args()

    path = args.jsonl_path

    global IS_PARENT_CHILD
    global IS_LLM_EXTRACT
    IS_PARENT_CHILD = args.parent_child
    IS_LLM_EXTRACT = args.llm_extract

    jsonl_files = []

    if os.path.isdir(path):
        # 如果是目录，递归处理目录及子目录下所有的json文件
        for root, _, files in os.walk(path):
            for file in files:
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
    for jsonl_path in jsonl_files:
        print(f"Processing file: {jsonl_path}")
        insert_data_streaming(col, parent_col, record_generator(jsonl_path),
                              ef, llm_client, args.llm_workers)

    print("All files processed successfully")


if __name__ == "__main__":
    main()
