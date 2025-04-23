# 多模态检索

本示例改编自官方教程 [Multimodal RAG with Milvus](https://milvus.io/docs/multimodal_rag_with_milvus.md)。

## 使用方法

在 JupyterLab / CodeServer 的终端，安装依赖并准备数据：

```bash
pip install --upgrade pymilvus opencv-python qwen-vl-utils
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/research/visual_bge
pip install -e .
pip install torchvision timm einops ftfy
```

修改 Python 脚本中的 Milvus URI 等全局变量，然后执行：

```bash
python multimodal-search.py
```

## 预期输出

```bash
...
Loading checkpoint shards: 100%|██████████████████████████████| 5/5 [02:50<00:00, 34.06s/it]
Collection 'amazon_reviews_2023' created
Generating image embeddings: 100%|████████████████████████| 900/900 [00:36<00:00, 24.55it/s]
Inserted 900 embeddings into Milvus
Searching for images similar to ./images_folder/leopard.jpg with text: 'phone case with this image theme'
Reranking results with LLVM...
Reranking result: Ranked list: [7, 1, 2]

Reasons: 
1. The phone case in position 7 features a blue leopard print design, which directly matches the theme of the query image.
2. The phone case in position 1 is black and appears to have a fur-like texture, which could be interpreted as a more abstract representation of a leopard's coat.
3. The phone case in position 2 has a black and white owl design, which does not match the leopard theme at all.

The most suitable item is the phone case in position 7 because it directly incorporates the leopard print theme from the query image.
Original results saved to: ./images_folder/combined_image.jpg
Reranked results saved to: ./images_folder/reranked_combined_image.jpg
```

## TODO

* 将 rerank 改为在线推理
* 将 embedding 改为在线推理
