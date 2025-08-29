import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Optional, Tuple

import requests


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))

ALLOWED_CATEGORIES = {
    "婚姻家庭",
    "债权债务",
    "交通事故",
    "房产纠纷",
}

DENY_PATTERNS = [
    # 投诉/举报/咨询 电话、热线
    r"(投诉|举报|维权|咨询).*(电话|热线)",
    r"(电话|热线).*(多少|几号)",
    # 律师费用相关
    r"(请)?律师.*(多少钱|收费|费用|要花多少|要多少钱)",
    r"律师费|律师费用",
    # 起诉/诉讼耗时、进度
    r"(起诉|打官司|立案).*(多久|多长时间|多少天|多久下来|多长时间下来)",
    r"(诉讼|开庭|审理|立案).*(周期|时长|多久|多长时间)",
    # 文书写法/模板/格式/版本：起诉书、离婚协议、借条等
    r"(起诉书|诉状|申诉书|起诉状|离婚协议书|协议书|协议|借条|欠条|合同|文书|起诉材料|告知书|声明|证明).*(怎么写|如何写|写法|模板|范本|格式|版本|样板|范文|样本|参考|范例)",
    r"(怎么写|如何写|写法|模板|范本|格式|版本|样板|范文|样本|参考|范例).*(起诉书|诉状|离婚协议|协议书|借条|欠条|合同|文书)",
    # 流程/地点/资格类泛问：外地能否离婚之类（缺少具体案情）
    r"外地.*离婚.*(可以|能|行|吗)",
    r"(可以|能|行)不(可以|能|行).*(离婚|结婚|起诉)",
    # 过度泛化的“怎么处理/怎么办”且含“纠纷”但缺少事实细节（启发式）
    r"纠纷.*(怎么处理|如何处理|怎么办)$",
    # 看守所/拘留所/监狱 会见/探视/通信/写信
    r"(看.{0,2}所|拘留所|监狱).*(会见|探视|写信|通信|通话|打电话)",
]


def should_force_reject(question: str) -> bool:
    text = question.strip()
    if not text:
        return True
    lowered = text.lower()
    for pattern in DENY_PATTERNS:
        if re.search(pattern, lowered):
            return True
    if len(text) < 6:
        return True
    return False


def normalize_category(name: Optional[str]) -> str:
    """标准化类别名：去除首尾空白与中间所有空白符。"""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", "", name.strip())


def serialize_single_line(text: str) -> str:
    """将任意问题规范为单行表示：转义 \\、\n、\r、\t，不加外围引号。"""
    try:
        escaped = (
            text.replace("\\", "\\\\")
            .replace("\r", r"\r")
            .replace("\n", r"\n")
            .replace("\t", r"\t")
        )
        return escaped
    except Exception:
        # 兜底：尽量保持无外围引号
        dumped = json.dumps(text, ensure_ascii=False)
        if len(dumped) >= 2 and dumped[0] == '"' and dumped[-1] == '"':
            return dumped[1:-1]
        return dumped


def build_messages(question: str) -> list:
    system_prompt = """你是一名中国法律助手，判断用户问题是否与以下法律直接相关：

* 《中华人民共和国刑法》（含历次修正案）
* 《中华人民共和国民法典》。

**参考信息：**

刑法的目录如下：

第一编 总则  
  第一章 刑法的任务、基本原则和适用范围  
  第二章 犯罪  
    第一节 犯罪和刑事责任  
    第二节 犯罪的预备、未遂和中止  
    第三节 共同犯罪  
    第四节 单位犯罪  
  第三章 刑罚  
    第一节 刑罚的种类  
    第二节 管制  
    第三节 拘役  
    第四节 有期徒刑、无期徒刑  
    第五节 死刑  
    第六节 罚金  
    第七节 剥夺政治权利  
    第八节 没收财产  
  第四章 刑罚的具体运用  
    第一节 量刑  
    第二节 累犯  
    第三节 自首和立功  
    第四节 数罪并罚  
    第五节 缓刑  
    第六节 减刑  
    第七节 假释  
    第八节 时效  
  第五章 其他规定  

第二编 分则  
  第一章 危害国家安全罪  
  第二章 危害公共安全罪  
  第三章 破坏社会主义市场经济秩序罪  
    第一节 生产、销售伪劣商品罪  
    第二节 走私罪  
    第三节 妨害对公司、企业的管理秩序罪  
    第四节 破坏金融管理秩序罪  
    第五节 金融诈骗罪  
    第六节 危害税收征管罪  
    第七节 侵犯知识产权罪  
    第八节 扰乱市场秩序罪  
  第四章 侵犯公民人身权利、民主权利罪  
  第第五章 侵犯财产罪  
  第六章 妨害社会管理秩序罪  
    第一节 扰乱公共秩序罪  
    第二节 妨害司法罪  
    第三节 妨害国（边）境管理罪  
    第四节 妨害文物管理罪  
    第五节 危害公共卫生罪  
    第六节 破坏环境资源保护罪  
    第七节 走私、贩卖、运输、制造毒品罪  
    第八节 组织、强迫、引诱、容留、介绍卖淫罪  
    第九节 制作、贩卖、传播淫秽物品罪  
  第七章 危害国防利益罪  
  第八章 贪污贿赂罪  
  第九章 渎职罪  
  第十章 军人违反职责罪附则

附则

民法典的目录如下：

第一编 总则
  第一章 基本规定
  第二章 自然人
    第一节 民事权利能力和民事行为能力
    第二节 监护
    第三节 宣告失踪和宣告死亡
    第四节 个体工商户和农村承包经营户
  第三章 法人
    第一节 一般规定
    第二节 营利法人
    第三节 非营利法人
    第四节 特别法人
  第四章 非法人组织
  第五章 民事权利
  第六章 民事法律行为
    第一节 一般规定
    第二节 意思表示
    第三节 民事法律行为的效力
    第四节 民事法律行为的附条件和附期限
  第七章 代理
    第一节 一般规定
    第二节 委托代理
    第三节 代理终止
  第八章 民事责任
  第九章 诉讼时效
  第十章 期间计算

第二编 物权编
  第一分编 通则
    第一章 一般规定
    第二章 物权的设立、变更、转让和消灭
      第一节 不动产登记
      第二节 动产交付
      第三节 其他规定
    第三章 物权的保护
  第二分编 所有权
    第四章 一般规定
    第五章 国家所有权和集体所有权、私人所有权
    第六章 业主的建筑物区分所有权
    第七章 相邻关系
    第八章 共有
    第九章 所有权取得的特别规定
  第三分编 用益物权
    第十章 一般规定
    第十一章 土地承包经营权
    第十二章 建设用地使用权
    第十三章 宅基地使用权
    第十四章 居住权
    第十五章 地役权
  第四分编 担保物权
    第十六章 一般规定
    第十七章 抵押权
      第一节 一般抵押权
      第二节 最高额抵押权
    第十八章 质权
      第一节 动产质权
      第二节 权利质权
    第十九章 留置权
  第五分编 占有
    第二十章 占有

第三编 合同编
  第一分编 通则
    第一章 一般规定
    第二章 合同的订立
    第三章 合同的效力
    第四章 合同的履行
    第五章 合同的保全
    第六章 合同的变更和转让
    第七章 合同的权利义务终止
    第八章 违约责任
  第二分编 典型合同
    第九章 买卖合同
    第十章 供用电、水、气、热力合同
    第十一章 赠与合同
    第十二章 借款合同
    第十三章 保证合同
      第一节 一般规定
      第二节 保证责任
    第十四章 租赁合同
    第十五章 融资租赁合同
    第十六章 保理合同
    第十七章 承揽合同
    第十八章 建设工程合同
    第十九章 运输合同
      第一节 一般规定
      第二节 客运合同
      第三节 货运合同
      第四节 多式联运合同
    第二十章 技术合同
      第一节 一般规定
      第二节 技术开发合同
      第三节 技术转让合同和技术许可合同
      第四节 技术咨询合同和技术服务合同
    第二十一章 保管合同
    第二十二章 仓储合同
    第二十三章 委托合同
    第二十四章 物业服务合同
    第二十五章 行纪合同
    第二十六章 中介合同
    第二十七章 合伙合同
  第三分编 准合同
    第二十八章 无因管理
    第二十九章 不当得利

第四编 人格权编
  第一章 一般规定
  第二章 生命权、身体权和健康权
  第三章 姓名权和名称权
  第四章 肖像权
  第五章 名誉权和荣誉权
  第六章 隐私权和个人信息保护

第五编 婚姻家庭编
  第一章 一般规定
  第二章 结婚
  第三章 家庭关系
    第一节 夫妻关系
    第二节 父母子女关系和其他近亲属关系
  第四章 离婚
  第五章 收养
    第一节 收养关系的成立
    第二节 收养的效力
    第三节 收养关系的解除

第六编 继承编
  第一章 一般规定
  第二章 法定继承
  第三章 遗嘱继承和遗赠
  第四章 遗产的处理

第七编 侵权责任编
  第一章 一般规定
  第二章 损害赔偿
  第三章 责任主体的特殊规定
  第四章 产品责任
  第五章 机动车交通事故责任
  第六章 医疗损害责任
  第七章 环境污染和生态破坏责任
  第八章 高度危险责任
  第九章 饲养动物损害责任
  第十章 建筑物和物件损害责任

附则

**判定规则：**

1. 如果回答问题需要引用或依据上述法条（罪名、量刑、刑责要素；或民事权利义务、合同、人格权、婚姻家庭、继承、侵权等），判定为 `true`。
2. 如果问题与典型刑事/民事案件在事实、案件类型或裁判要点上明显相似，且此类案件通常直接援引刑法或民法典条文，判定为 `true`。
3. 若问题表述不清楚、不完整，或存在严重语病导致无法判定具体法律关系或事实要点，判定为 `false`。
4. 若问题太笼统，缺少细节，判定为 `false`。
5. 以下类型一律判定为 `false`：
   - 投诉/举报/维权/咨询电话或热线类问题（如“投诉电话多少”）。
   - 律师费用/请律师多少钱/收费标准类问题。
   - 起诉/打官司/立案的耗时、进度、时长类问题（如“多久下来”）。
   - 看守所/拘留所/监狱的会见、探视、通信、写信、打电话等程序性问题。
   - 文书写法/模板/格式/版本类问题（如“起诉书怎么写”“离婚协议书模板”“借条范本”等）。
   - 仅询问能否/是否办理、地点资格、流程类的泛化问题，且缺少具体案情事实与要点（如“外地可以离婚吗”）。
6. 只需输出 `related` 与 `confidence` 两个字段。类别的校验与筛选由程序在加载数据后处理，不需要你在输出中给出。

**输出要求（严格）：**
仅输出 JSON，且字段、拼写、大小写严格一致；
confidence 的定义：你对 `related` 判断的确定性，取值范围为 [0,1]，1 表示极其确定，0 表示完全不确定；
当信息不足或边界模糊时，请给出 ≤ 0.5 的分值；

```json
{"related": true, "confidence": 0.90}
```

或

```json
{"related": false, "confidence": 0.20}
```
"""

    user_prompt = (
        "请只输出严格 JSON（不要额外文本、注释或 markdown 代码块）。\n"
        "请返回字段：related（布尔）与 confidence（0~1 小数）。\n"
        "confidence 定义：你对 related 判断的确定性，范围 [0,1]；信息不足或边界模糊时给 ≤0.5。\n"
        f"问题：{question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(messages: list, retries: int = 3, timeout: int = 300) -> str:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        # 尽量约束为 JSON 输出（若服务端不支持会被忽略）
        "response_format": {"type": "json_object"},
        # 启用 thinking/推理（不同服务端可能忽略未支持的字段）
        "extra_body": {
            "reasoning": {"effort": "medium"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # 兼容不同响应结构
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"] or ""
                if "text" in choice:  # 退化为非 chat 模式
                    return choice.get("text", "")
            # 无法解析时，直接返回原始文本
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
            # 指数退避
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def parse_related_flag(content: str) -> Optional[bool]:
    # 首选 JSON 解析
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "related" in parsed:
            val = parsed["related"]
            confidence = parsed.get("confidence")
            # 当 related 为 true 时，仅基于 confidence 阈值进行验收
            if isinstance(val, bool):
                try:
                    conf = float(confidence)
                except Exception:
                    conf = 0.0
                if val is True and conf < CONFIDENCE_THRESHOLD:
                    return False
                return val
            if isinstance(val, str):
                lowered = val.strip().lower()
                bool_val = True if lowered in {"true", "yes", "y"} else False if lowered in {"false", "no", "n"} else None
                if bool_val is None:
                    raise ValueError("无法解析 related 字段为布尔值")
                try:
                    conf = float(confidence)
                except Exception:
                    conf = 0.0
                if bool_val and conf < CONFIDENCE_THRESHOLD:
                    return False
                return bool_val
    except Exception:  # noqa: BLE001
        pass

    # 退化：从自由文本中粗略提取
    lowered = content.strip().lower()
    # 优先匹配显式 JSON 片段
    m = re.search(r"\{\s*\"related\"\s*:\s*(true|false)\s*\}", lowered)
    if m:
        return m.group(1) == "true"
    # 次优：YES/NO
    if re.search(r"\byes\b|\btrue\b", lowered):
        return True
    if re.search(r"\bno\b|\bfalse\b", lowered):
        return False
    return None


def process_file(
    input_path: Path,
    output_path: Path,
    sleep_seconds: float = 0.0,
    progress_every: int = 1000,
) -> Tuple[int, int, int]:
    processed = 0
    kept = 0
    skipped = 0

    # 读取已存在的问题，避免重复写入
    existing: set = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f_out:
            for line in f_out:
                line = line.rstrip("\n")
                if line:
                    existing.add(line)

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("a", encoding="utf-8") as f_append:
        for raw_line in f_in:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue

            # 在加载 obj 之后基于已有标签先行过滤
            raw_cause = obj.get("cause")
            norm_cause = normalize_category(raw_cause)
            if not norm_cause or norm_cause not in ALLOWED_CATEGORIES:
                # 不在四大类内，直接丢弃，避免在 prompt 里判断
                processed += 1
                continue

            # 本地预过滤：直接拒绝不想要的问题，减少无效 LLM 调用
            if should_force_reject(question):
                processed += 1
                continue

            messages = build_messages(question.strip())
            content = call_llm(messages)
            flag = parse_related_flag(content)
            if flag is None:
                # 无法确定，保守丢弃
                processed += 1
                continue

            if flag:
                serialized = serialize_single_line(question)
                if question not in existing and serialized not in existing:
                    f_append.write(serialized + "\n")
                    f_append.flush()
                    existing.add(question)
                    existing.add(serialized)
                kept += 1

            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"进度: 已处理 {processed} 行，已提取 {kept} 问题", flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return processed, kept, skipped


def process_directory(
    input_dir: Path,
    output_path: Path,
    sleep_seconds: float = 0.0,
    progress_every: int = 1000,
) -> Tuple[int, int, int]:
    processed = 0
    kept = 0
    skipped = 0

    # 读取已存在的问题，避免重复写入
    existing: set = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f_out:
            for line in f_out:
                line = line.rstrip("\n")
                if line:
                    existing.add(line)

    with output_path.open("a", encoding="utf-8") as f_append:
        for json_file in sorted(input_dir.rglob("*.json")):
            try:
                content = json_file.read_text(encoding="utf-8").strip()
            except Exception:
                skipped += 1
                continue

            if not content:
                skipped += 1
                continue

            try:
                obj = json.loads(content)
            except Exception:
                skipped += 1
                continue

            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue

            # 在加载 obj 之后基于已有标签先行过滤
            raw_cause = obj.get("cause")
            norm_cause = normalize_category(raw_cause)
            if not norm_cause or norm_cause not in ALLOWED_CATEGORIES:
                # 不在四大类内，直接丢弃，避免在 prompt 里判断
                processed += 1
                continue

            # 本地预过滤：直接拒绝不想要的问题，减少无效 LLM 调用
            if should_force_reject(question):
                processed += 1
                continue

            messages = build_messages(question.strip())
            content = call_llm(messages)
            flag = parse_related_flag(content)
            if flag is None:
                # 无法确定，保守丢弃
                processed += 1
                continue

            if flag:
                serialized = serialize_single_line(question)
                if question not in existing and serialized not in existing:
                    f_append.write(serialized + "\n")
                    f_append.flush()
                    existing.add(question)
                    existing.add(serialized)
                kept += 1

            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"进度: 已处理 {processed} 行，已提取 {kept} 问题", flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return processed, kept, skipped


def main() -> None:
    here = Path(__file__).resolve().parent
    output_path = here / "questions.txt"

    if len(sys.argv) < 2:
        raise SystemExit("用法: python extract.py <数据目录或JSONL文件路径>")

    input_arg = Path(sys.argv[1]).resolve()
    progress_every = int(os.environ.get("PROGRESS_EVERY", "1000"))

    if input_arg.is_dir():
        processed, kept, skipped = process_directory(
            input_arg,
            output_path,
            sleep_seconds=0.0,
            progress_every=progress_every,
        )
    else:
        if not input_arg.exists():
            raise FileNotFoundError(f"找不到输入路径: {input_arg}")
        processed, kept, skipped = process_file(
            input_arg,
            output_path,
            sleep_seconds=0.0,
            progress_every=progress_every,
        )
    print(
        json.dumps(
            {
                "processed": processed,
                "kept": kept,
                "skipped": skipped,
                "model": MODEL_NAME,
                "base_url": BASE_URL,
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
