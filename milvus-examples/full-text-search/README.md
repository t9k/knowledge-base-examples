# 全文检索

本示例改编自官方教程 [Full Text Search with Milvus](https://milvus.io/docs/full_text_search_with_milvus.md)。

在 JupyterLab / CodeServer 的终端，安装依赖并准备数据：

```bash
pip install --upgrade pymilvus openai requests tqdm
wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

修改 Python 脚本中的 Milvus URI 和 LLM base URL 等全局变量，然后执行：

```bash
python full-text-search.py
```

## 预期输出

```bash
Step 1: Inserting data into Milvus...
Data inserted into collection: milvus_hybrid

Step 2: Testing sparse search...
Query: create a logger?
Results (sparse search):
1. doc_10 (9.1150)
   use {
    crate::args::LogArgs,
    anyhow::{anyhow, Result},
    simplelog::{Config, LevelFilter, W...
2. doc_87 (7.0005)
                LoggerPtr INF = Logger::getLogger(LOG4CXX_TEST_STR("INF"));
                INF->setLevel(Level::getInfo());

                ...
3. doc_89 (6.7437)
   using namespace log4cxx;
using namespace log4cxx::helpers;

LOGUNIT_CLASS(FMTTestCase)
{
        LOGUNIT_TE...

Step 3: Running evaluation with hybrid search...
Pass@5: 0.8318
```
