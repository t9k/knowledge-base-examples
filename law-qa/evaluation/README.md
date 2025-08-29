# 评估

## 构建评估问题集

1. 下载[法律知识问答](https://aistudio.baidu.com/datasetdetail/89457)数据集的数据文件 `laws_qa_data.zip`，移动到当前目录下
2. 解压数据文件：

```bash
unzip laws_qa_data.zip
unzip laws_data/9d3f0f41-c7df-4211-a32a-9e1fc74f8a68.zip
```

3. 抽取涉及刑法或民法典的高质量的问题：

```bash
# 基于规则和 LLM 抽取
# 需要指定 OPENAI_BASE_URL, MODEL_NAME, API_KEY
# 抽取结果与 LLM 有关，能力越强的 LLM，抽取的质量越高
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 MODEL_NAME=Qwen3-32B API_KEY=dummy python extract.py ./train
```

4. 得到 `questions.txt` 文件，每一行为一个问题，用于评估、演示或测试。
