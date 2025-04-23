# 基本 RAG

本示例改编自官方教程 [Build RAG with Milvus](https://milvus.io/docs/build-rag-with-milvus.html)。

在 JupyterLab / CodeServer 的终端，安装依赖并准备数据：

```bash
pip install --upgrade pymilvus openai requests tqdm
wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

修改 Python 脚本中的 Milvus URI 和 LLM base URL 等全局变量，然后执行：

```bash
python basic-rag.py
```

## 预期输出

```bash
Creating embeddings: 100%|███████████████| 72/72 [00:03<00:00, 19.77it/s]
Search Results:
[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###",
        0.8661684393882751
    ],
    [
        "What data types does Milvus support on the primary key field?\n\nIn current release, Milvus supports both INT64 and string.\n\n###",
        0.8131600618362427
    ],
    [
        "Why is there no vector data in etcd?\n\netcd stores Milvus module metadata; MinIO stores entities.\n\n###",
        0.8087844848632812
    ]
]

Generated Answer:
In Milvus, data is stored in two main forms: inserted data and metadata. Inserted data, which includes vector data, scalar data, and collection-specific schema, is stored in persistent storage as incremental logs. These logs can be stored using various object storage backends such as MinIO, AWS S3, Google Cloud Storage, Azure Blob Storage, Alibaba Cloud OSS, and Tencent Cloud Object Storage (COS). On the other hand, metadata, generated within Milvus for each module, is stored in etcd.
```
