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

# 配置
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
PARENT_COLLECTION_NAME = "law_hybrid_demo_parent"
PARENT_CHUNK_SIZE = 4096
PARENT_CHUNK_OVERLAP = 0
CHILD_COLLECTION_NAME = "law_hybrid_demo_child"
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 32
BATCH_SIZE = 50


# 读取 jsonl 文件，每行一个 dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_text(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\r\n", "\n", "。", "；", "，", "、"],
        keep_separator="end")
    return text_splitter.split_text(text)


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token="root:Milvus")

    # 创建文档集合
    parent_fields = [
        FieldSchema(name="parent_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=100),
        FieldSchema(name="fact_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=65535),
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

    # 创建分段集合
    fields = [
        FieldSchema(name="child_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="fact_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="relevant_articles",
                    dtype=DataType.ARRAY,
                    element_type=DataType.INT64,
                    max_capacity=10),
        FieldSchema(name="accusation",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=100,
                    max_capacity=10),
        FieldSchema(name="punish_of_money", dtype=DataType.INT64),
        FieldSchema(name="criminal", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="imprisonment", dtype=DataType.INT64),
        FieldSchema(name="life_imprisonment", dtype=DataType.BOOL),
        FieldSchema(name="death_penalty", dtype=DataType.BOOL),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    if utility.has_collection(CHILD_COLLECTION_NAME):
        Collection(CHILD_COLLECTION_NAME).drop()
    col = Collection(CHILD_COLLECTION_NAME, schema, consistency_level="Strong")
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
    parent_col.load()
    return col, parent_col


def record_generator(jsonl_path):
    parent_records = []  # 用于存储父文档记录
    child_records = []  # 用于存储子文档记录

    for item in read_jsonl(jsonl_path):
        fact = item["fact"].replace(",", "，").replace(";", "；")
        meta = item["meta"]
        fact_id = str(uuid.uuid4())

        # 第一级切分：将fact切分为多个parent
        parent_chunks = chunk_text(
            fact, PARENT_CHUNK_SIZE,
            PARENT_CHUNK_OVERLAP) if len(fact) > PARENT_CHUNK_SIZE else [fact]

        for parent_chunk in parent_chunks:
            if len(parent_chunk) <= 3:
                continue

            parent_id = str(uuid.uuid4())

            # 存储父文档
            parent_records.append({
                "parent_id": parent_id,
                "fact_id": fact_id,
                "chunk": parent_chunk
            })

            # 第二级切分：将parent切分为多个child
            child_chunks = chunk_text(
                parent_chunk, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP) if len(
                    parent_chunk) > CHILD_CHUNK_SIZE else [parent_chunk]

            for child_chunk in child_chunks:
                if len(child_chunk) <= 3:
                    continue

                child_id = str(uuid.uuid4())

                child_records.append({
                    "child_id":
                    child_id,
                    "parent_id":
                    parent_id,
                    "fact_id":
                    fact_id,
                    "chunk":
                    child_chunk,
                    "relevant_articles":
                    meta["relevant_articles"],
                    "accusation":
                    meta["accusation"],
                    "punish_of_money":
                    meta["punish_of_money"],
                    "criminal":
                    meta["criminals"][0],
                    "imprisonment":
                    meta["term_of_imprisonment"]["imprisonment"],
                    "life_imprisonment":
                    meta["term_of_imprisonment"]["life_imprisonment"],
                    "death_penalty":
                    meta["term_of_imprisonment"]["death_penalty"],
                })

                # 当子文档记录达到批量大小时，一起处理
                if len(child_records) >= BATCH_SIZE:
                    for record in child_records:
                        yield record
                    child_records = []

        # 当父文档记录达到批量大小时，一起处理
        if len(parent_records) >= BATCH_SIZE:
            for parent_data in parent_records:
                yield {"is_parent": True, **parent_data}
            parent_records = []

    # 处理剩余的子文档记录
    for record in child_records:
        yield record

    # 处理剩余的父文档记录
    for parent_data in parent_records:
        yield {"is_parent": True, **parent_data}


def insert_data_streaming(col, doc_col, record_iter, ef):
    chunk_buffer = []
    doc_buffer = []
    total_chunks = 0
    total_docs = 0

    for record in tqdm(record_iter, desc="Processing records"):
        if record.get("is_parent", False):
            # 这是一个文档记录
            doc_buffer.append(record)
            if len(doc_buffer) >= BATCH_SIZE:
                # 插入文档数据
                doc_col.insert([
                    [r["parent_id"] for r in doc_buffer],  # parent_id
                    [r["fact_id"] for r in doc_buffer],  # fact_id
                    [r["chunk"] for r in doc_buffer],  # chunk
                    [[0.0, 0.0] for _ in doc_buffer]  # vector
                ])
                total_docs += len(doc_buffer)
                doc_buffer = []
        else:
            # 这是一个分段记录
            chunk_buffer.append(record)
            if len(chunk_buffer) >= BATCH_SIZE:
                # 准备嵌入向量
                chunks = [r["chunk"] for r in chunk_buffer]
                embeddings = ef(chunks)

                # 插入分段数据
                col.insert([
                    [r["child_id"] for r in chunk_buffer],
                    [r["parent_id"] for r in chunk_buffer],
                    [r["fact_id"] for r in chunk_buffer],
                    [r["chunk"] for r in chunk_buffer],
                    [r["relevant_articles"] for r in chunk_buffer],
                    [r["accusation"] for r in chunk_buffer],
                    [r["punish_of_money"] for r in chunk_buffer],
                    [r["criminal"] for r in chunk_buffer],
                    [r["imprisonment"] for r in chunk_buffer],
                    [r["life_imprisonment"] for r in chunk_buffer],
                    [r["death_penalty"] for r in chunk_buffer],
                    embeddings["sparse"],
                    embeddings["dense"],
                ])
                total_chunks += len(chunk_buffer)
                chunk_buffer = []

    # 处理剩余的分段数据
    if chunk_buffer:
        chunks = [r["chunk"] for r in chunk_buffer]
        embeddings = ef(chunks)

        col.insert([
            [r["parent_id"] for r in chunk_buffer],
            [r["fact_id"] for r in chunk_buffer],
            [r["relevant_articles"] for r in chunk_buffer],
            [r["accusation"] for r in chunk_buffer],
            [r["punish_of_money"] for r in chunk_buffer],
            [r["criminal"] for r in chunk_buffer],
            [r["imprisonment"] for r in chunk_buffer],
            [r["life_imprisonment"] for r in chunk_buffer],
            [r["death_penalty"] for r in chunk_buffer],
            embeddings["sparse"],
            embeddings["dense"],
        ])
        total_chunks += len(chunk_buffer)

    # 处理剩余的文档数据
    if doc_buffer:
        doc_col.insert([
            [r["parent_id"] for r in doc_buffer],  # parent_id
            [[0.0, 0.0] for _ in doc_buffer],  # vector
            [r["chunk"] for r in doc_buffer],  # chunk
            [r["fact_id"] for r in doc_buffer]  # fact_id
        ])
        total_docs += len(doc_buffer)

    print(
        f"Inserted {total_chunks} child chunks and {total_docs} parent chunks."
    )


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python insert_data.py <jsonl_file>")
        return
    jsonl_path = sys.argv[1]
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]
    col, parent_col = setup_milvus_collection(dense_dim)
    insert_data_streaming(col, parent_col, record_generator(jsonl_path), ef)


if __name__ == "__main__":
    main()
