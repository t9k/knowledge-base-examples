# 数据

## 数据概况

### 刑事案例

下载 [CAIL2018_ALL_DATA.zip](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) 文件，解压得到多个 json 文件（应为 jsonl 文件），这些文件的每一行是一个 json 对象，对应一个刑事案例，例如：

```json
{"fact": "公诉机关指控，2016年3月29日6时许，被告人严某某在其家中吸食毒品时被公安民警抓获，民警当场从其上衣口袋内搜缴甲基苯丙胺（冰毒）1包及甲基苯丙胺片剂（麻古）2包，共计12.44克。", "meta": {"relevant_articles": [248, 357], "accusation": ["非法持有毒品"], "punish_of_money": 2000, "criminals": ["严某某"], "term_of_imprisonment": {"death_penalty": false, "imprisonment": 6, "life_imprisonment": false}}}
```

其中各个字段的含义为：

- `fact`：事实描述（当作一个小文章来处理）
- `meta`：元数据，原本是作为预测任务的标签
  - `relevant_articles`：相关法条
  - `accusation`：罪名
  - `punish_of_money`：罚金
  - `criminals`：被告
  - `term_of_imprisonment`：
    - `death_penalty`：是否死刑
    - `imprisonment`：刑期（单位：月）
    - `life_imprisonment`：是否无期徒刑

解压后的数据文件放置在 <https://modelscope.cn/datasets/qazwsxplkj/CAIL2018> / <https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/CAIL2018>。

**数据规模**

全部 7 个 json 文件（`data_test.json`、`data_train.json`、`data_valid.json`、`test.json`、`train.json`、`rest_data.json`、`final_test.json`）：

- 共有 2,916,228 行/个案例
- 文件总大小为 3.38GB
- fact 文本总长度（以 python 的 `len()` 函数测量的字符数）为 1,060,206,212，平均长度为 363.6

### 民事案例

裁判文书数据购自淘宝，从 1985 年到 2021 年共 37 个压缩文件，总大小 ~94.3GB，总计 ~8506 万条数据。这里仅下载和处理 2021 年的数据。

使用网盘 App 下载 2021年裁判文书数据.zip 文件，解压得到多个 csv 文件，这些文件的每一行是一条数据，对应一个裁判文书，例如：

```csv
原始链接,案号,案件名称,法院,所属地区,案件类型,案件类型编码,来源,审理程序,裁判日期,公开日期,当事人,案由,法律依据,全文
https://wenshu.court.gov.cn/website/wenshu/181107ANFZ0BXSK4/index.html?docId=18e4dbdb13524e3b93afadb600a15151,（2021）黑7530执472号,赵文君、李建国等买卖合同纠纷首次执行执行通知书,黑龙江省诺敏河人民法院,黑龙江省,执行案件,2,www.macrodatas.cn,执行实施,2021-10-02,2021-10-03,赵文君；李建国；安亚君,买卖合同纠纷,,文书内容黑龙江省诺敏河人民法院结 案 通 知 书（2021）黑7530执472号李**、安亚君：关于你二人申请执行程希江、赵文君买卖合同纠纷一案，本案已执行完毕。根据最高人民法院《关于执行案件立案、结案问题的意见》第十五条之规定，本案做结案处理。特此通知。二〇二一年十月二日 微信公众号“马克 数据网”
```

**数据预处理**：

对于解压得到的 csv 文件，在同一目录下执行预处理脚本 [`preprocess.py`](./preprocess_judgment_docs.py)：

```bash
python preprocess_judgment_docs.py
```

具体细节请查看代码，这里介绍主要步骤：

1. 删除"来源"列
2. 保留"案件类型编码"为 1（民事）的行
3. 保留"案件名称"中包含"判决书"的行
4. 删除"全文"中包含"撤诉"的行
5. 移除"全文"末尾的广告信息（马克数据网相关）
6. 将HTML实体编码替换为对应字符
7. 移除"全文"中的换行符

预处理后的数据文件放置在 <https://modelscope.cn/datasets/qazwsxplkj/cn-judgment-docs> / <https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/cn-judgment-docs>。

**数据规模**

处理后的 2021 年数据共 10 个 csv 文件（`processed_2021_01.csv` 到 `processed_2021_10.csv`）：

- 共有 1,710,428 行/个案例
- 文件总大小为 3.97 GiB
- 文本总长度（以 python 的 `len()` 函数测量的字符数）为 5,336,145,891，平均长度为 3,120

### 法律法规

数据文件放置在 <https://modelscope.cn/datasets/qazwsxplkj/cn-laws> / <https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/cn-laws>。请前往查看包含哪些法律法规，以及数据预处理的过程。

**数据规模**

法律法规共 1,433 个 md 文件：

- 共有 144,632 行（空行不计入）
- 文件总大小为 22MB
- 文本总长度（以 python 的 `len()` 函数测量的字符数）为 7,869,601

## 数据处理与入库

数据通过 Python 脚本进行处理，并最终存入 Milvus 向量数据库。

### 刑事案件

脚本 [db_insert.py](../argo-workflows/bg-deploy-criminal-case/db_insert.py) 负责处理和入库刑事案件数据。

**主要步骤**：

1. 递归处理指定目录下的所有 json 文件
2. 逐行读取 json 文件，解析为 dict 实例，其中包含已有的元数据 relevant_articles、accusation、punish_of_money 等
3. 若启用父子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 parent chunk，继续使用 RecursiveCharacterTextSplitter 对 parent chunk 进行 chunking 得到 (child) chunk
    若不启用分子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 chunk
4. 对于每个 fact、parent chunk 和 chunk，生成一个 uuid 作为唯一标识符
5. 对于每个 chunk，qwen3-embedding-0.6b 模型生成密集向量，bge-m3 模型生成稀疏向量，另外可选地调用 qwen3-32b 模型提取额外的元数据
6. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息如下：

**chunking**

- child：langchain_text_splitters.RecursiveCharacterTextSplitter
  - seperator：["\r\n", "\n", "。", "；", "，", "、"]
  - keep_separator：end
  - chunk size：256
  - overlap：32
- parent (optional)：langchain_text_splitters.RecursiveCharacterTextSplitter
  - seperator：["\r\n", "\n", "。", "；", "，", "、"]
  - keep_separator：end
  - chunk size：4096
  - overlap：0

**embedding**

- sparse： bge-m3
  - pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理
- dense：qwen3-embedding-0.6b
  - vLLM 在线推理

**Milvus Collection**

- sparse indexing：SPARSE_INVERTED_INDEX，IP
- dense indexing：HNSW (M=24, efConstruction=400)，COSINE

**indexing**

- sparse indexing：SPARSE_INVERTED_INDEX，IP
- dense indexing：HNSW (M=24, efConstruction=400)，COSINE

**Collection fields**

- chunk_id (VarChar)：chunk 的 uuid
- chunk (VarChar)：chunk 的内容
- relevant_articles (Array[Int64])：相关法条
- accusation (VarChar)：罪名
- punish_of_money (Int64)：罚金
- criminals (VarChar)：被告人/罪犯
- imprisonment (Int64)：刑期（单位：月）
- life_imprisonment (Bool)：是否无期徒刑
- death_penalty (Bool)：是否死刑
- sparse_vector (SparseFloatVector)：稀疏嵌入
- dense_vector	(FloatVector)：密集嵌入

如启用 LLM 提取元数据，还有：

- dates (VarChar)：出现的日期
- locations (VarChar)：出现的地点
- people (VarChar)：出现的人物，包括其姓名、审判中的角色、职业
- numbers (VarChar)：出现的数值
- criminals_llm (VarChar)：LLM 提取的被告人

### 民事案件

脚本 [db_insert.py](../argo-workflows/bg-deploy-civil-case/db_insert.py) 负责处理和入库民事案件数据。

**主要步骤**：

1. 处理指定目录下的所有 csv 文件
2. 流式读取 csv 文件，解析为 dict 实例，其中包含已有的元数据 case_number、case_name、court 等
3. 若启用父子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 parent chunk，继续使用 RecursiveCharacterTextSplitter 对 parent chunk 进行 chunking 得到 (child) chunk
    若不启用分子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 chunk
4. 对于每个 fact、parent chunk 和 chunk，生成一个 uuid 作为唯一标识符
5. 对于每个 chunk，qwen3-embedding-0.6b 模型生成密集向量，bge-m3 模型生成稀疏向量，另外可选地调用 qwen3-32b 模型提取额外的元数据
6. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息如下：

**chunking**

- child：langchain_text_splitters.RecursiveCharacterTextSplitter
  - seperator：["\r\n", "\n", "。", "；", "，", "、", "：", "（", "）"]
  - keep_separator：end
  - chunk size：256
  - overlap：32
- parent (optional)：langchain_text_splitters.RecursiveCharacterTextSplitter
  - seperator：["\r\n", "\n", "。", "；", "，", "、", "：", "（", "）"]
  - keep_separator：end
  - chunk size：4096
  - overlap：0

**embedding**

- sparse： bge-m3
  - pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理
- dense：qwen3-embedding-0.6b
  - vLLM 在线推理

**Milvus Collection**

- sparse indexing：SPARSE_INVERTED_INDEX，IP
- dense indexing：HNSW (M=24, efConstruction=400)，COSINE

**indexing**

- sparse indexing：SPARSE_INVERTED_INDEX，IP
- dense indexing：HNSW (M=24, efConstruction=400)，COSINE

**Collection fields**

- chunk_id (VarChar)：chunk 的 uuid
- case_id (VarChar)：案例的 uuid
- chunk (VarChar)：chunk 的内容
- case_number (VarChar)：案例的案号
- case_name (VarChar)：案例的名称
- court (VarChar)：执行审判的法院
- region (VarChar)：地区
- judgment_date (VarChar)：审判日期
- parties (VarChar)：当事人
- case_cause (VarChar)：案由
- legal_basis (JSON)：审判的法律依据
- sparse_vector (SparseFloatVector)：稀疏嵌入
- dense_vector	(FloatVector)：密集嵌入

如启用 LLM 提取元数据，还有：

- dates (VarChar)：出现的日期
- locations (VarChar)：出现的地点
- people (VarChar)：出现的人物，包括其姓名、审判中的角色、职业
- numbers (VarChar)：出现的数值
- parties_llm (VarChar)：LLM 提取的当事人

### 法律数据

脚本 [db_insert.py](../argo-workflows/bg-deploy-law/db_insert.py) 负责处理和入库法律数据。

**主要步骤**：

1. 递归处理指定目录下的所有 md 文件
2. 读取整个 md 文件，进行两层 chunking：
   - 使用 MarkdownHeaderTextSplitter 基于 md 标题进行 chunking，标题不会被保留在 chunk 中，但会被抽取为元数据：一、二、三、四级标题分别作为元数据 law、part、chapter 和 section，或者一、二、三级标题分别作为元数据 law、chapter 和 section。没有标题则相应的元数据的值为空字符串。
   - 手动实现 chunking，使用模式 `r'(第[零一二三四五六七八九十百千]+条 |[一二三四五六七八九十百千]+、|<!-- INFO END -->|<!-- FORCE BREAK -->)'`，分别匹配法条和特殊分隔符。匹配法条时，保留匹配对象，并把匹配对象中的中文数字转换为数字，作为元数据 article 的值；匹配分隔符时，不保留匹配对象。
   - 最终每个 chunk 是一个法条（或法律通过信息），具有元数据 law、part、chapter、section 和 article。
3. 对于每个 chunk，生成一个 uuid 作为唯一标识符，bge-m3 嵌入模型生成密集向量和稀疏向量
4. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息如下：

**chunking**

- langchain_text_splitters.MarkdownHeaderTextSplitter
  - headers_to_split_on：
    - [("#", "Law"), ("##", "Part"), ("###", "Chapter"), ("####", "Section")]
    - [("#", "Law"), ("##", "Chapter"), ("###", "Section")]
  - strip_headers：True
- 手动实现 chunking 到一个一个法条：
  - separator：r'(第[零一二三四五六七八九十百千]+条 |[一二三四五六七八九十百千]+、|<!-- INFO END -->|<!-- FORCE BREAK -->)'
  - keep_separator：start

**embedding**

- sparse： bge-m3
  - pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理
- dense：qwen3-embedding-0.6b
  - vLLM 在线推理

**indexing**

- sparse indexing：SPARSE_INVERTED_INDEX，IP
- dense indexing：HNSW (M=24, efConstruction=400)，COSINE

**Collection fields**

- chunk_id (VarChar)：chunk 的 uuid
- chunk (VarChar)：chunk 的内容
- law (VarChar)：所属法律
- part (VarChar)：所属编
- chapter (VarChar)：所属章
- section (VarChar)：所属节
- article (Int64)：法条序号（0 表示没有序号）
- article_amended (Int64)：修正的刑法法条序号（0 表示没有修正刑法法条）
- sparse_vector (SparseFloatVector)：稀疏嵌入
- dense_vector	(FloatVector)：密集嵌入
