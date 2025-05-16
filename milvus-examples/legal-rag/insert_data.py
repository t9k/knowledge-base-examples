import sys
import os
import glob
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
COLLECTION_NAME = "law_hybrid_demo"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 50


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


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token="root:Milvus")
    fields = [
        FieldSchema(name="chunk_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100),
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


def record_generator(jsonl_path):
    for item in read_jsonl(jsonl_path):
        fact = item["fact"].replace(",", "，").replace(";", "；")
        meta = item["meta"]
        chunks = chunk_text(fact) if len(fact) > CHUNK_SIZE else [fact]

        for chunk in chunks:
            if len(chunk) <= 3:
                continue

            yield {
                "chunk_id":
                str(uuid.uuid4()),
                "chunk":
                chunk,
                "relevant_articles":
                [int(i) for i in meta["relevant_articles"]],
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
            }


def insert_data_streaming(col, record_iter, ef, batch_size=BATCH_SIZE):
    buffer = []
    total = 0
    for record in tqdm(record_iter, desc="Inserting records"):
        buffer.append(record)
        if len(buffer) >= batch_size:
            chunks = [r["chunk"] for r in buffer]
            embeddings = ef(chunks)
            to_insert = [
                [r["chunk_id"] for r in buffer],
                [r["chunk"] for r in buffer],
                [r["relevant_articles"] for r in buffer],
                [r["accusation"] for r in buffer],
                [r["punish_of_money"] for r in buffer],
                [r["criminal"] for r in buffer],
                [r["imprisonment"] for r in buffer],
                [r["life_imprisonment"] for r in buffer],
                [r["death_penalty"] for r in buffer],
                embeddings["sparse"],
                embeddings["dense"],
            ]
            col.insert(to_insert)
            total += len(buffer)
            buffer = []
    # 插入剩余部分
    if buffer:
        chunks = [r["chunk"] for r in buffer]
        embeddings = ef(chunks)
        to_insert = [
            [r["chunk_id"] for r in buffer],
            [r["chunk"] for r in buffer],
            [r["relevant_articles"] for r in buffer],
            [r["accusation"] for r in buffer],
            [r["punish_of_money"] for r in buffer],
            [r["criminal"] for r in buffer],
            [r["imprisonment"] for r in buffer],
            [r["life_imprisonment"] for r in buffer],
            [r["death_penalty"] for r in buffer],
            embeddings["sparse"],
            embeddings["dense"],
        ]
        col.insert(to_insert)
        total += len(buffer)
    print(f"Inserted {total} records.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python insert_data.py <jsonl_file>")
        return

    path = sys.argv[1]
    jsonl_files = []
    
    if os.path.isdir(path):
        # 如果是目录，递归处理目录及子目录下所有的json文件
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    jsonl_files.append(os.path.join(root, file))
        
        if not jsonl_files:
            print(f"No JSON files found in directory or subdirectories: {path}")
            return
        print(f"Found {len(jsonl_files)} JSON files to process")
    elif os.path.isfile(path):
        # 如果是文件，直接处理
        jsonl_files = [path]
    else:
        print(f"Path not found: {path}")
        return

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]
    col = setup_milvus_collection(dense_dim)

        
    # 处理每个文件
    for jsonl_path in jsonl_files:
        print(f"Processing file: {jsonl_path}")
        insert_data_streaming(col, record_generator(jsonl_path), ef)
    
    print("All files processed successfully")


if __name__ == "__main__":
    main()
