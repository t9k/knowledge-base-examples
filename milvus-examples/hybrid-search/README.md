# 混合检索

本示例改编自官方教程 [Hybrid Search with Milvus](https://milvus.io/docs/hybrid_search_with_milvus.md)。

在 JupyterLab / CodeServer 的终端，安装依赖并准备数据：

```bash
pip install --upgrade pymilvus "pymilvus[model]"
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
```

修改 Python 脚本中的 Milvus URI 等全局变量，然后执行：

```bash
python hybrid-search.py
```

## 预期输出

```bash
Loaded 502 documents
Example document: What is the best travel website?
Generating embeddings using BGE-M3 model...
Fetching 30 files: 100%|█████████████████████████████| 30/30 [00:00<00:00, 57456.22it/s]
pre tokenize: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 1133.29it/s]
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Inference Embeddings: 100%|█████████████████████████████| 32/32 [01:35<00:00,  2.97s/it]
Setting up Milvus collection...
Number of entities inserted: 0
Generating embeddings for the query...
Running searches...
Dense Search Results:
- What's the best way to start learning robotics?
- How do I learn a computer language like java?
- How can I get started to learn information security?
- What is Java programming? How To Learn Java Programming Language ?
- How can I learn computer security?
- What is the best way to start robotics? Which is the best development board that I can start working on it?
- How can I learn to speak English fluently?
- What are the best ways to learn French?
- How can you make physics easy to learn?
- How do we prepare for UPSC?

Sparse Search Results:
- What is Java* programming? How* To Learn Java Programming Language ?
- What's the best way* to start learning* robotics*?*
- What is the alternative* to* machine* learning?*
- *How* do I create a new Terminal and new shell in Linux using C* programming?*
- *How* do I create a new shell in a new terminal using C* programming* (Linux terminal)*?*
- Which business is better* to start* in Hyderabad*?*
- Which business is good* start* up in Hyderabad*?*
- What is the best way* to start* robotics*?* Which is the best development board that I can* start* working on it*?*
- What math does a complete newbie need* to* understand algorithms for computer* programming?* What books on algorithms are suitable for a complete beginner*?*
- *How* do you make life suit you and stop life from abusi*ng* you mentally and emotionally*?*

Hybrid Search Results:
- What is Java* programming? How* To Learn Java Programming Language ?
- What's the best way* to start learning* robotics*?*
- What is the best way* to start* robotics*?* Which is the best development board that I can* start* working on it*?*
- *How* do I learn a computer language like java*?*
- *How* can I get started* to* learn information security*?*
- *How* can I learn computer security*?*
- *How* can I learn* to* speak English fluently*?*
- What are the best ways* to* learn French*?*
- *How* can you make physics easy* to* learn*?*
- *How* do we prepare for UPSC*?*
```

## TODO

* 将 embedding 改为在线推理
