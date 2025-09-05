# 法律领域 RAG 应用

相关文档：[RAG 应用使用 MCP](https://docs.google.com/document/d/1syGEImzQfxdGJCAL9qBz0ql0ZrHvdB_HerH0uHWGJNY/edit?usp=sharing)

# 目的

1. 构建法律领域的生产级别的 RAG 应用  
2. 搞清楚各种细节，作为技术积累

# 数据

## 案例数据

### 刑法案例数据

原项目：[https://github.com/china-ai-law-challenge/CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018)  
数据源头：[中国裁判文书网](http://wenshu.court.gov.cn/)

下载 [https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018\_ALL\_DATA.zip](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) 文件，解压得到多个 json 文件（应为 jsonl 文件），这些文件的每一行是一个 json 对象，对应一个刑事案例，例如：

| {"fact": "公诉机关指控，2016年3月29日6时许，被告人严某某在其家中吸食毒品时被公安民警抓获，民警当场从其上衣口袋内搜缴甲基苯丙胺（冰毒）1包及甲基苯丙胺片剂（麻古）2包，共计12.44克。", "meta": {"relevant\_articles": \[248, 357\], "accusation": \["非法持有毒品"\], "punish\_of\_money": 2000, "criminals": \["严某某"\], "term\_of\_imprisonment": {"death\_penalty": false, "imprisonment": 6, "life\_imprisonment": false}}} |
| :---- |

其中各个字段的含义为：

* fact：事实描述（当作一个小文章来处理）  
* meta：元数据，原本是作为预测任务的标签  
  * relevant\_articles：相关法条  
  * accusation：罪名  
  * punish\_of\_money：罚金  
  * criminals：被告  
  * term\_of\_imprisonment：  
    * death\_penalty：是否死刑  
    * imprisonment：刑期（单位：月）  
    * life\_imprisonment：是否无期徒刑

解压后的数据文件放置在 [https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/CAIL2018](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/CAIL2018)。

#### 数据处理

检查发现 fact 文本存在以下问题：

| \# | 问题 | 示例 | 处理方法 |
| :---- | :---- | :---- | :---- |
| 1 | 连续的全角句号/分号/逗号/顿号 | 请依法判处。。 从而骗取张某、余某等七名受害人现金77000元，， 证人刘某、吴某、、杨海某、等人 | 丢弃长度小于等于 3 的 chunk |
| 2 | 半角符号 | 厮打过程中,被告人李某将侯某5胸部打伤 被害人王某甲、王某乙等人的陈述;被告人邵某的供述和辩解 证人邓某、帕某.艾散的证人证言等证据 重庆市南川区大观镇秋收.森林湖尚楼盘内部认购 朱某某被公安机关抓获.到案后如实供述了上述犯罪事实 | 替换为全角符号： "," \-\> "，"（前后都为数字的除外） ";" \-\> "；"  |
|  |  |  |  |

在[处理和插入数据](#处理和插入数据)的脚本中进行处理。

#### 数据特征

* 全部 7 个 json 文件（data\_test.json、data\_train.json、data\_valid.json、test.json、train.json、rest\_data.json、final\_test.json）：  
  * 共有 2,916,228 行/个案例  
  * 文件总大小为 3.38GB  
  * fact 文本总长度（以 python 的 len() 函数测量的字符数）为 1,060,206,212，平均长度为 363.6

* fact 文本的长度分布统计如下：

![][image1]

* 分位数：  
  * 最小值: 6  
  * 1% 分位数: 90.0  
  * 10% 分位数: 148.0  
  * 50% 分位数: 290.0  
  * 90% 分位数: 601.0  
  * 99% 分位数: 1581.0  
  * 最大值: 56,694  
* 短文本/长文本计数  
  * 长度小于 32：2,030 (0.07%)  
  * 长度大于 4096：4,374 (0.15%)  
  * 长度大于 8192：1,062 (0.04%)

* relevant\_articles 列表最多有 12 个元素  
* accusation 列表最多有 13 个元素，元素最长有 36 个字符

* 每一行是一条单独的数据，不同行之间没有关联

### 民法案例数据 {#民法案例数据}

原项目：自购自淘宝[中国裁判文书网现在全量裁判文书数据](https://item.taobao.com/item.htm?id=931320010030)  
数据源头：[中国裁判文书网](http://wenshu.court.gov.cn/)

裁判文书数据从 1985 年到 2021 年共 37 个压缩文件，总大小 \~94.3GB，总计 \~8506 万条数据。这里仅下载和处理 2021 年的数据。

使用网盘 App 下载 2021年裁判文书数据.zip 文件，解压得到多个 csv 文件，这些文件的每一行是一条数据，对应一个裁判文书，例如：

| 原始链接,案号,案件名称,法院,所属地区,案件类型,案件类型编码,来源,审理程序,裁判日期,公开日期,当事人,案由,法律依据,全文 https://wenshu.court.gov.cn/website/wenshu/181107ANFZ0BXSK4/index.html?docId=18e4dbdb13524e3b93afadb600a15151,（2021）黑7530执472号,赵文君、李建国等买卖合同纠纷首次执行执行通知书,黑龙江省诺敏河人民法院,黑龙江省,执行案件,2,www.macrodatas.cn,执行实施,2021-10-02,2021-10-03,赵文君；李建国；安亚君,买卖合同纠纷,,文书内容黑龙江省诺敏河人民法院结 案 通 知 书（2021）黑7530执472号李\*\*、安亚君：关于你二人申请执行程希江、赵文君买卖合同纠纷一案，本案已执行完毕。根据最高人民法院《关于执行案件立案、结案问题的意见》第十五条之规定，本案做结案处理。特此通知。二〇二一年十月二日 微信公众号“马克 数据网” |
| :---- |

原始数据文件放置在 [https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents/files/main/raw](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents/files/main/raw)。

#### 数据预处理

对于解压得到的 csv 文件，在同一目录下创建预处理脚本 [preprocess.py](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents/blob/main/processed/process.py) 并执行：

| python preprocess.py |
| :---- |

具体细节请查看代码，这里介绍主要步骤：

1. 删除"来源"列  
2. 保留"案件类型编码"为 1（民事）的行  
3. 保留"案件名称"中包含"判决书"的行  
4. 删除"全文"中包含"撤诉"的行  
5. 移除"全文"末尾的广告信息（马克数据网相关）  
6. 将HTML实体编码替换为对应字符  
7. 移除"全文"中的换行符

预处理后的数据文件放置在 [https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents/files/main/preprocessed](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents/files/main/preprocessed)。

#### 数据处理

检查发现全文文本存在以下问题：

| \# | 问题 | 示例 | 处理方法 |
| :---- | :---- | :---- | :---- |
| 1 | 连续的全角句号/分号/逗号/顿号 | 住址：新疆托克逊县。。 女，汉族，，住西宁市城东区。 | 丢弃长度小于等于 3 的 chunk |
| 2 | 半角符号 | 产籍号为42-111-164-2012,建筑面积116.5平方米,  | 替换为全角符号： "," \-\> "，"（前后都为数字的除外） ";" \-\> "；"  |
|  |  |  |  |

在[处理和插入数据](#处理和插入数据)的脚本中进行处理。

#### 数据特征

* 全部 10 个 csv 文件（processed\_2021\_01.csv 到 processed\_2021\_10.csv）：  
  * 共有 1,710,428 行/个案例  
  * 文件总大小为 3.97 GiB  
  * 文本总长度（以 python 的 len() 函数测量的字符数）为 5,336,145,891，平均长度为 3,120

* 裁判文书全文的长度分布统计如下：

![][image2]

* 分位数：  
  * 最小值: 8  
  * 1% 分位数: 594.0  
  * 10% 分位数: 1,187.0  
  * 50% 分位数: 2,540.0  
  * 90% 分位数: 5,557.0  
  * 99% 分位数: 11,846  
  * 最大值: 13,5658  
* 短文本/长文本计  
  * 长度大于 8192：5,5702 (3.26%)

* 每一行是一条单独的数据，不同行之间没有关联

## 法律数据

### 刑法数据

原项目：[https://github.com/Oreomeow/Law-Book](https://github.com/Oreomeow/Law-Book)  
数据源头：[国家法律法规数据库](https://flk.npc.gov.cn/)

拉取 [https://github.com/Oreomeow/Law-Book](https://github.com/Oreomeow/Law-Book) 项目，复制其中的刑法文档：

| git clone https://github.com/Oreomeow/Law-Book.gitcp Law-Book/7-刑法/刑法\* . rm \-rf Law-Book |
| :---- |

#### 数据预处理

做以下手工修改：

1. 补充刑法修改案十二：创建文件**刑法修正案12.md**，内容复制自 [http://www.npc.gov.cn/npc/c2/c30834/202312/t20231229\_433988.html](http://www.npc.gov.cn/npc/c2/c30834/202312/t20231229_433988.html)，格式参照**刑法修正案11.md**

预处理后的刑法文档放置在 [https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/cn-laws/files/main/criminal-law](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/cn-laws/files/main/criminal-law)。

#### 数据特征

* 刑法和 12 个刑法修正案共 13 个 md 文件：  
  * 共有 1,831 行（空行不计入）  
  * 文件总大小为 360KB  
  * 文本总长度（以 python 的 len() 函数测量的字符数）为 116,115

### 民法典数据

原项目：[https://github.com/Oreomeow/Law-Book](https://github.com/Oreomeow/Law-Book)  
数据源头：[国家法律法规数据库](https://flk.npc.gov.cn/)

拉取 [https://github.com/Oreomeow/Law-Book](https://github.com/Oreomeow/Law-Book) 项目，复制其中的民法典文档：

| git clone https://github.com/Oreomeow/Law-Book.gitcp Law-Book/3-民法典/\* . rm 0-README.md rm \-rf Law-Book |
| :---- |

#### 数据预处理

做以下手工修改和脚本处理：

1. 对于 **1-总则.md**，将下列行移动到“\# 中华人民共和国民法典”之后，“\# 总则“之前；对于其他文档，移除下列行：

| 2020年5月28日 第十三届全国人民代表大会第三次会议通过2021年1月1日 施行\<\!-- INFO END \--\> |
| :---- |

2. 对于 **8-附则.md**，移除下列行：

| \#\# |
| :---- |

3. 在 md 文件的同一目录下创建预处理脚本 [formatter.py](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/cn-laws/blob/main/civil-code/formatter.py) 并执行：

| python formatter.py . |
| :---- |

4. 修改文件名：**1-总则.md** \=\> **民法典1总则.md**，**2-物权编.md** \=\> **民法典2物权编.md**，以此类推。

预处理后的刑法文档放置在 [https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/cn-laws/files/main/civil-code](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/cn-laws/files/main/civil-code)。

#### 数据特征

* 民法典的 8 编共 8 个 md 文件：  
  * 共有 2,272 行（空行不计入）  
  * 文件总大小为 336KB  
  * 文本总长度（以 python 的 len() 函数测量的字符数）为 112,480

## 备选数据 {#备选数据}

CAIL 数据：

1. 2021  
   1. 阅读理解：刑事和民事案例  
   2. 论辩理解：辩诉双方论点  
   3. 法律问答  
2. 2022  
   1. 涉法舆情摘要：法律相关新闻报道  
   2. 论辩理解：辩诉双方论点

其他数据：

1. [CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant)：法律咨询问题分类；刑法罪名解说  
2. [LaWGPT](https://github.com/pengxiao-song/LaWGPT)：法律领域词表  
3. [LEVEN](https://github.com/thunlp/LEVEN?tab=readme-ov-file)：刑事判决书  
4. [LeCaRD](https://github.com/myx666/LeCaRD)：刑事案例类案检索  
5. [JEC-QA](https://github.com/thunlp/jec-qa?tab=readme-ov-file)：司法考试选择题  
6. [法律知识问答数据集](https://aistudio.baidu.com/aistudio/datasetdetail/89457)

# RAG 实现

## 处理和插入数据 {#处理和插入数据}

### 刑法案例

运行脚本 [insert\_data\_cail2018.py](https://github.com/t9k/knowledge-base-examples/blob/main/milvus-examples/legal-rag/insert_data_cail2018.py)：

| HF\_ENDPOINT=https://hf-mirror.com python insert\_data\_CAIL\_2018.py \\   \--use-gcu \\                  *\# 使用 gcu 进行 bge-m3 本地推理*   \--parent-child \\             *\# 启用父子分段*   \--llm-extract \\              *\# 启用 LLM 提取额外的元数据*   \--llm-workers 32 \\           *\# 向 LLM 发送请求的工作器数量*   ./final\_all\_data/exercise\_contest/data\_valid.json  *\# json 案例数据* |
| :---- |

主要步骤：

1. 递归处理指定目录下的所有 json 文件  
2. 逐行读取 json 文件，解析为 dict 实例，其中包含已有的元数据 relevant\_articles、accusation、punish\_of\_money 等  
3. 若启用父子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 parent chunk，继续使用 RecursiveCharacterTextSplitter 对 parent chunk 进行 chunking 得到 (child) chunk  
   若不启用分子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 chunk  
4. 对于每个 fact、parent chunk 和 chunk，生成一个 uuid 作为唯一标识符  
5. 对于每个 chunk，qwen3-embedding-0.6b 模型生成密集向量，bge-m3 模型生成稀疏向量，另外调用 qwen3-32b 模型[提取额外的元数据](##5-llm-提取标签测试-c)  
6. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息：

**chunking**

* child：langchain\_text\_splitters.RecursiveCharacterTextSplitter  
  * seperator：\["\\r\\n", "\\n", "。", "；", "，", "、"\]  
  * keep\_separator：end  
  * chunk size：256  
  * overlap：32  
* parent (optional)：langchain\_text\_splitters.RecursiveCharacterTextSplitter  
  * seperator：\["\\r\\n", "\\n", "。", "；", "，", "、"\]  
  * keep\_separator：end  
  * chunk size：4096  
  * overlap：0

**embedding**

* sparse： bge-m3  
  * pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理  
* dense：qwen3-embedding-0.6b  
  * vLLM 在线推理

**indexing**

* [sparse indexing](https://milvus.io/docs/index-vector-fields.md?tab=sparse)：SPARSE\_INVERTED\_INDEX，IP  
* dense indexing：HNSW ([M=24, efConstruction=400](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall))，COSINE

**Collection fields**

* chunk\_id (VarChar)：chunk 的 uuid  
* chunk (VarChar)：chunk 的内容  
* relevant\_articles (Array\[Int64\])：相关法条  
* accusation (VarChar)：罪名  
* punish\_of\_money (Int64)：罚金  
* criminals (VarChar)：被告人/罪犯  
* imprisonment (Int64)：刑期（单位：月）  
* life\_imprisonment (Bool)：是否无期徒刑  
* death\_penalty (Bool)：是否死刑  
* sparse\_vector (SparseFloatVector)：稀疏嵌入  
* dense\_vector	(FloatVector)：密集嵌入

如启用 LLM 提取元数据，还有：

* dates (VarChar)：出现的日期  
* locations (VarChar)：出现的地点  
* people (VarChar)：出现的人物，包括其姓名、审判中的角色、职业  
* numbers (VarChar)：出现的数值  
* criminals\_llm (VarChar)：LLM 提取的被告人

| chunking 操作如下： chunking 算法 \[[代码](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py#L58)\] 使用 "\\r\\n" 作为分隔符分割 fact 文本，分隔符本身会被合并到它的前一个 split 的末尾位置 遍历 splits： 若 split 的长度小于 chunk\_size，将其 append 到 \_good\_splits  List(str)  中 若 split 的长度大于等于 chunk\_size： 对于 \_good\_splits 调用合并算法进行合并，extend 到 final\_chunks List(str)  中，清空 \_good\_splits 如还有后续其他分隔符，递归调用 chunking 算法，继续分割这个超长的 split；否则，将这个超长的 split append 到 final\_chunks 中； 遍历完成后，对于 \_good\_splits 调用合并算法进行合并，extend 到 final\_chunks 中； 返回 final\_chunks 合并算法 \[[代码](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py#L107)\] 遍历 splits： 将 split append 到 current\_doc List(str) 中，直到总长度刚好不超过 chunk\_size（即总长度加上下一个 split 的长度大于 chunk\_size） 将 current\_doc 中的 splits 合并为一个 split，append 到 docs List(str) 中，然后进入循环： 条件：当 current\_doc 的总长度大于 overlap，或总长度加上下一个 split 的长度大于 chunk\_size 执行：从 current\_doc 踢出第一个 split 回到 a. 遍历完成后，将 current\_doc 中的 splits 合并为一个 split，append 到 docs 中； 返回 docs |
| :---- |

| 对于此 chunking 设置，刑法案例的 chunk 的长度分布统计如下：  ![][image3] chunk 总量：6,214,662 chunk / fact：2.13 分位数 最小值: 4 1% 分位数: 20.0 10% 分位数: 81.0 50% 分位数: 182.0 90% 分位数: 243.0 99% 分位数: 255.0 最大值: 933 |
| :---- |

| 父子分段的实现参考了 langchain.retrievers.parent\_document\_retriever 的[源代码](https://python.langchain.com/api_reference/_modules/langchain/retrievers/parent_document_retriever.html)，但没有使用它，而是手动实现。 |
| :---- |

### 民法案例

运行脚本 [insert\_data\_judgment\_docs.py](https://github.com/t9k/knowledge-base-examples/blob/main/milvus-examples/legal-rag/insert_data_judgment_docs.py)：

| HF\_ENDPOINT=https://hf-mirror.com python insert\_data\_judgment\_docs.py \\   \--use-gcu \\                  *\# 使用 gcu 进行 bge-m3 本地推理*   \--parent-child \\             *\# 启用父子分段*   \--llm-extract \\              *\# 启用 LLM 提取额外的元数据*   \--llm-workers 32 \\           *\# 向 LLM 发送请求的工作器数量*   ./judgment\_docs/10000.csv    *\# csv 案例数据* |
| :---- |

主要步骤：

1. 处理指定目录下的所有 csv 文件  
2. 流式读取 csv 文件，解析为 dict 实例，其中包含已有的元数据 case\_number、case\_name、court 等  
3. 若启用父子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 parent chunk，继续使用 RecursiveCharacterTextSplitter 对 parent chunk 进行 chunking 得到 (child) chunk  
   1. 若不启用分子分段：使用 RecursiveCharacterTextSplitter 基于 seperator 和 chunk size 对 fact 进行 chunking 得到 chunk  
4. 对于每个 fact、parent chunk 和 chunk，生成一个 uuid 作为唯一标识符  
5. 对于每个 chunk，qwen3-embedding-0.6b 模型生成密集向量，bge-m3 模型生成稀疏向量，另外调用 qwen3-32b 模型[提取额外的元数据](##5-llm-提取标签测试-c)  
6. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息：

**chunking**

* child：langchain\_text\_splitters.RecursiveCharacterTextSplitter  
  * seperator：\["\\r\\n", "\\n", "。", "；", "，", "、"\]  
  * keep\_separator：end  
  * chunk size：256  
  * overlap：32  
* parent (optional)：langchain\_text\_splitters.RecursiveCharacterTextSplitter  
  * seperator：\["\\r\\n", "\\n", "。", "；", "，", "、"\]  
  * keep\_separator：end  
  * chunk size：4096  
  * overlap：0

**embedding**

* sparse： bge-m3  
  * pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理  
* dense：qwen3-embedding-0.6b  
  * vLLM 在线推理

**indexing**

* [sparse indexing](https://milvus.io/docs/index-vector-fields.md?tab=sparse)：SPARSE\_INVERTED\_INDEX，IP  
* dense indexing：HNSW ([M=24, efConstruction=400](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall))，COSINE

**Collection fields**

* chunk\_id (VarChar)：chunk 的 uuid  
* case\_id (VarChar)：案例的 uuid  
* chunk (VarChar)：chunk 的内容  
* case\_number (VarChar)：案例的案号  
* case\_name (VarChar)：案例的名称  
* court (VarChar)：执行审判的法院  
* region (VarChar)：地区  
* judgment\_date (VarChar)：审判日期  
* parties (VarChar)：当事人  
* case\_cause (VarChar)：案由  
* legal\_basis (JSON)：审判的法律依据  
* sparse\_vector (SparseFloatVector)：稀疏嵌入  
* dense\_vector	(FloatVector)：密集嵌入

如启用 LLM 提取元数据，还有：

* dates (VarChar)：出现的日期  
* locations (VarChar)：出现的地点  
* people (VarChar)：出现的人物，包括其姓名、审判中的角色、职业  
* numbers (VarChar)：出现的数值  
* parties\_llm (VarChar)：LLM 提取的当事人

### 法律数据

运行脚本 [insert\_data\_laws.py](https://github.com/t9k/knowledge-base-examples/blob/main/milvus-examples/legal-rag/insert_data_laws.py)：

| HF\_ENDPOINT=https://hf-mirror.com python insert\_data\_laws.py \\  \--use-gcu \\                  *\# 使用 gcu 进行 bge-m3 本地推理*  ./cn-laws/criminal-law/      *\# 包含 markdown 法律数据的目录* |
| :---- |

主要步骤：

1. 递归处理指定目录下的所有 md 文件  
2. 读取整个 md 文件（最大的单一文件为**刑法.md**，大小 212KB），进行两层 chunking：  
   1. 使用 MarkdownHeaderTextSplitter 基于 md 标题进行 chunking，标题不会被保留在 chunk 中，但会被抽取为元数据：一、二、三、四级标题分别作为元数据 law、part、chapter 和 section。没有标题则相应的元数据的值为空字符串。  
   2. 手动实现 chunking，使用模式 r'(第\[零一二三四五六七八九十百千\]+条 |\[一二三四五六七八九十百千\]+、|\<\!-- INFO END \--\>)'，分别匹配法条，刑法修正案的法条，法律通过信息与正文的分隔符。匹配前两者（法条）时，保留匹配对象，并把匹配对象中的中文数字转换为数字，作为元数据 article 的值；匹配后者（分隔符）时，不保留匹配对象。  
   3. 最终每个 chunk 是一个法条（或法律通过信息），具有元数据 law、part、chapter、section 和 article。  
3. 对于每个 chunk，生成一个 uuid 作为唯一标识符，bge-m3 嵌入模型生成密集向量和稀疏向量  
4. 将 chunk 批次插入到 Milvus Collection 中

chunking、embedding 及 Milvus Collection 配置信息：

**chunking**

* langchain\_text\_splitters.MarkdownHeaderTextSplitter  
  * headers\_to\_split\_on：\[("\#", "Law"), ("\#\#", "Part"), ("\#\#\#", "Chapter"), ("\#\#\#\#", "Section")\]  
  * strip\_headers：True  
* 手动实现 chunking 到一个一个法条：  
  * separator：r'(第\[零一二三四五六七八九十百\]+条 |\[一二三四五六七八九十\]+、)'  
  * keep\_separator：start

**embedding**

* sparse： bge-m3  
  * pymilvus.model.hybrid.BGEM3EmbeddingFunction 离线批推理  
* dense：qwen3-embedding-0.6b  
  * vLLM 在线推理

**indexing**

* [sparse indexing](https://milvus.io/docs/index-vector-fields.md?tab=sparse)：SPARSE\_INVERTED\_INDEX，IP  
* dense indexing：HNSW ([M=24, efConstruction=400](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall))，COSINE

**Collection fields**

* chunk\_id (VarChar)：chunk 的 uuid  
* chunk (VarChar)：chunk 的内容  
* law (VarChar)：所属法律  
* part (VarChar)：所属编  
* chapter (VarChar)：所属章  
* section (VarChar)：所属节  
* article (Int64)：法条序号（0 表示没有序号）  
* article\_amended (Int64)：修正的刑法法条序号（0 表示没有修正刑法法条）  
* sparse\_vector (SparseFloatVector)：稀疏嵌入  
* dense\_vector	(FloatVector)：密集嵌入

## RAG

TODO
