from typing import List, Optional, Dict, Any
import time
import math
import json
import re
import logging
import os
import threading
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from core.tokenizer import get_tokenizer

# 模块级日志器（同时写入文件 ../logs/openai_api_YYYYMMDD_PID.log 与标准输出）
logger = logging.getLogger(__name__)
try:
    # 清理已有 handler，避免重复输出
    if logger.handlers:
        logger.handlers.clear()
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir,
        f"openai_api_{time.strftime('%Y%m%d')}_{os.getpid()}.log",
    )
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _stream_handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _file_handler.setFormatter(_formatter)
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("Logging to %s", log_path)
except Exception:
    # 兜底：若文件 handler 初始化失败，退回到标准输出
    if not logger.handlers:
        _fallback = logging.StreamHandler()
        _fallback.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(_fallback)
    logger.setLevel(logging.INFO)

# =========================
# 统计聚合（小时/天）
# =========================
_metrics_lock = threading.Lock()
_metrics = {
    'hour': {
        'stream': {
            'ttft_ms': [],
            'total_ms': [],
            'out_tokens': [],
            'tps': [],
        },
        'non_stream': {
            'total_ms': [],
            'out_tokens': [],
            'tps': [],
        },
        'count_stream': 0,
        'count_non_stream': 0,
        'period_start': datetime.now().replace(minute=0, second=0, microsecond=0),
    },
    'day': {
        'stream': {
            'ttft_ms': [],
            'total_ms': [],
            'out_tokens': [],
            'tps': [],
        },
        'non_stream': {
            'total_ms': [],
            'out_tokens': [],
            'tps': [],
        },
        'count_stream': 0,
        'count_non_stream': 0,
        'period_start': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    }
}


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    try:
        n = len(sorted_vals)
        if n == 0:
            return None
        if n == 1:
            return float(sorted_vals[0])
        pos = q * (n - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(sorted_vals[lower])
        weight = pos - lower
        return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)
    except Exception:
        return None


def _calc_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {'min': None, 'p50': None, 'p90': None, 'p99': None, 'max': None}
    vals = sorted(values)
    return {
        'min': float(vals[0]),
        'p50': _percentile(vals, 0.50),
        'p90': _percentile(vals, 0.90),
        'p99': _percentile(vals, 0.99),
        'max': float(vals[-1]),
    }


def _reset_scope(scope: str) -> None:
    now = datetime.now()
    if scope == 'hour':
        _metrics['hour'] = {
            'stream': {'ttft_ms': [], 'total_ms': [], 'out_tokens': [], 'tps': []},
            'non_stream': {'total_ms': [], 'out_tokens': [], 'tps': []},
            'count_stream': 0,
            'count_non_stream': 0,
            'period_start': now.replace(minute=0, second=0, microsecond=0),
        }
    elif scope == 'day':
        _metrics['day'] = {
            'stream': {'ttft_ms': [], 'total_ms': [], 'out_tokens': [], 'tps': []},
            'non_stream': {'total_ms': [], 'out_tokens': [], 'tps': []},
            'count_stream': 0,
            'count_non_stream': 0,
            'period_start': now.replace(hour=0, minute=0, second=0, microsecond=0),
        }


def _flush_and_log(scope: str) -> None:
    with _metrics_lock:
        data = _metrics.get(scope, {})
        if not data:
            return
        period_start = data.get('period_start')
        period_end = datetime.now()
        # 计算统计
        s = data['stream']
        ns = data['non_stream']
        count_stream = data.get('count_stream', 0)
        count_non_stream = data.get('count_non_stream', 0)
        stats = {
            'stream': {
                'ttft_ms': _calc_stats([v for v in s['ttft_ms'] if v is not None]),
                'total_ms': _calc_stats(s['total_ms']),
                'out_tokens': _calc_stats(s['out_tokens']),
                'tps': _calc_stats(s['tps']),
            },
            'non_stream': {
                'total_ms': _calc_stats(ns['total_ms']),
                'out_tokens': _calc_stats(ns['out_tokens']),
                'tps': _calc_stats(ns['tps']),
            }
        }
        logger.info(
            "[AGG-%s] period=%s~%s, count_stream=%d, count_non_stream=%d",
            scope.upper(), period_start, period_end, count_stream, count_non_stream,
        )
        def _fmt(d: Dict[str, Optional[float]]) -> str:
            def f(x):
                return '-' if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
            return f"min={f(d['min'])}, p50={f(d['p50'])}, p90={f(d['p90'])}, p99={f(d['p99'])}, max={f(d['max'])}"
        # 分别输出流式与非流式的各项指标
        logger.info(
            "[AGG-%s][STREAM] ttft_ms{%s}; total_ms{%s}; out_tokens{%s}; tps{%s}",
            scope.upper(), _fmt(stats['stream']['ttft_ms']), _fmt(stats['stream']['total_ms']),
            _fmt(stats['stream']['out_tokens']), _fmt(stats['stream']['tps'])
        )
        logger.info(
            "[AGG-%s][NON_STREAM] total_ms{%s}; out_tokens{%s}; tps{%s}",
            scope.upper(), _fmt(stats['non_stream']['total_ms']),
            _fmt(stats['non_stream']['out_tokens']), _fmt(stats['non_stream']['tps'])
        )
        _reset_scope(scope)


def _record_stream_metrics(ttft_ms: Optional[int], total_ms: int, out_tokens: int, tps: float) -> None:
    with _metrics_lock:
        try:
            _metrics['hour']['count_stream'] += 1
            _metrics['day']['count_stream'] += 1
            if ttft_ms is not None:
                _metrics['hour']['stream']['ttft_ms'].append(float(ttft_ms))
                _metrics['day']['stream']['ttft_ms'].append(float(ttft_ms))
            _metrics['hour']['stream']['total_ms'].append(float(total_ms))
            _metrics['day']['stream']['total_ms'].append(float(total_ms))
            _metrics['hour']['stream']['out_tokens'].append(float(out_tokens))
            _metrics['day']['stream']['out_tokens'].append(float(out_tokens))
            _metrics['hour']['stream']['tps'].append(float(tps))
            _metrics['day']['stream']['tps'].append(float(tps))
        except Exception:
            pass


def _record_nonstream_metrics(total_ms: int, out_tokens: int, tps: float) -> None:
    with _metrics_lock:
        try:
            _metrics['hour']['count_non_stream'] += 1
            _metrics['day']['count_non_stream'] += 1
            _metrics['hour']['non_stream']['total_ms'].append(float(total_ms))
            _metrics['day']['non_stream']['total_ms'].append(float(total_ms))
            _metrics['hour']['non_stream']['out_tokens'].append(float(out_tokens))
            _metrics['day']['non_stream']['out_tokens'].append(float(out_tokens))
            _metrics['hour']['non_stream']['tps'].append(float(tps))
            _metrics['day']['non_stream']['tps'].append(float(tps))
        except Exception:
            pass


def _scheduler_loop():
    while True:
        try:
            now = datetime.now()
            next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            next_midnight = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            sleep_seconds = max(0.5, (min(next_hour, next_midnight) - now).total_seconds())
            time.sleep(sleep_seconds)
            # 醒来后再次判断两个边界（可能同时触发）
            now2 = datetime.now()
            if now2 >= next_hour:
                _flush_and_log('hour')
            if now2 >= next_midnight:
                _flush_and_log('day')
        except Exception:
            # 避免线程退出
            time.sleep(5.0)


# 启动后台聚合线程
_t = threading.Thread(target=_scheduler_loop, name="metrics-aggregator", daemon=True)
_t.start()


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


def create_app(bot, model_name: str, api_key: Optional[str] = None, allow_cors: bool = False, tokenizer_path: Optional[str] = None) -> FastAPI:
    app = FastAPI()
    # 懒加载 tokenizer（可选）
    tokenizer = None
    if tokenizer_path:
        try:
            tokenizer = get_tokenizer(tokenizer_path)
            logger.info("Loaded tokenizer '%s'", tokenizer_path)
        except Exception:
            tokenizer = None
            logger.warning("Failed to load tokenizer '%s'. Will fall back to char-level throughput.", tokenizer_path)

    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        try:
            if tokenizer is not None:
                return len(tokenizer.encode(text))
        except Exception:
            pass
        # 回退：按字符近似
        return len(text)


    if allow_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _auth(request: Request):
        if not api_key:
            return
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = auth_header.split(" ", 1)[1].strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _normalize_arguments(args: Any) -> str:
        """将工具调用 arguments 规范为非空字符串。空/None/空对象均返回 "{}"。"""
        try:
            if isinstance(args, str):
                s = args.strip()
                return s if s else "{}"
            if args in (None, "", {}):
                return "{}"
            return json.dumps(args, ensure_ascii=False)
        except Exception:
            return "{}"

    def _extract_tool_response_text(data: List[Dict[str, Any]]) -> Optional[str]:
        """从事件列表中提取工具检索结果文本。
        - 直接从 content 中提取所有 <source id="...">...</source> 区块；丢弃其它文本。
        """
        parts: List[str] = []
        try:
            for item in data:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                if not isinstance(content, str) or not content:
                    continue

                # 抽取所有 <source id="n"> ... </source>
                blocks = re.findall(r"<source\s+id=\"\d+\">[\s\S]*?</source>", content)
                if blocks:
                    parts.append("\n\n".join([b.strip() for b in blocks]))
        except Exception:
            pass
        text = "\n".join([p for p in parts if p]).strip()
        return text if text else None

    @app.get("/v1/models")
    async def list_models(request: Request):
        _auth(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "owner",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        _auth(request)

        # 记录原始请求消息（包含 system）
        try:
            raw_messages = [
                (m.model_dump() if hasattr(m, "model_dump") else {"role": getattr(m, "role", None), "content": getattr(m, "content", None)})
                for m in (req.messages or [])
            ]
            logger.info("Incoming request messages: %s", json.dumps(raw_messages, ensure_ascii=False))
        except Exception as _:
            logger.exception("Failed to log raw request messages")

        # Filter out system messages; bot already has system prompt configured
        messages: List[Dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                continue
            messages.append({"role": m.role, "content": m.content or ""})

        if req.stream:
            created_ts = int(time.time())
            logger.info("Streaming response started: model=%s, temperature=%s, top_p=%s, max_tokens=%s", req.model or model_name, req.temperature, req.top_p, req.max_tokens)

            def _accumulate_oai_message(data: List[Dict[str, Any]]) -> Dict[str, Any]:
                # 将 Qwen Agent 累计消息合并为一个 OpenAI assistant message 的累计视图
                message: Dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
                seen_call_ids = set()
                for item in data:
                    # reasoning 增量聚合
                    if isinstance(item, dict) and item.get("reasoning_content"):
                        message["reasoning_content"] += str(item.get("reasoning_content") or "")
                    # content 增量聚合
                    if isinstance(item, dict) and item.get("role") == "assistant" and item.get("content"):
                        message["content"] += str(item.get("content") or "")
                    # 工具调用聚合（如存在 function_call）
                    if isinstance(item, dict) and item.get("function_call"):
                        fc = item.get("function_call") or {}
                        func_id = (item.get("extra", {}) or {}).get("function_id")
                        if not func_id:
                            func_id = f"call_{len(message['tool_calls'])}"
                        # 仅在 arguments 为完整 JSON 时才采纳
                        args_str = _normalize_arguments(fc.get("arguments"))
                        try:
                            parsed_args = json.loads(args_str)
                            # 仅采纳非空对象参数
                            if isinstance(parsed_args, dict) and parsed_args and func_id not in seen_call_ids:
                                message["tool_calls"].append({
                                    "id": func_id,
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": args_str,
                                    }
                                })
                                seen_call_ids.add(func_id)
                        except Exception:
                            # 非完整 JSON（增量中间态）不输出
                            pass
                return message

            def event_generator():
                # 维护累计状态，做增量 diff
                first_chunk_sent = False
                first_token_emitted = False
                ttft_s = None
                last_content = ""
                last_reasoning = ""
                last_tool_calls_len = 0
                last_tool_response = ""
                collected_tool_calls: List[Dict[str, Any]] = []
                gen_t0 = time.perf_counter()
                try:
                    for event in bot.run(messages=messages):
                        if isinstance(event, list):
                            if not first_token_emitted and event[0]['content']:
                                first_token_emitted = True
                                ttft_s = max(0.0, time.perf_counter() - gen_t0)

                            acc = _accumulate_oai_message(event)

                            # 发送首个空内容块，含 role 字段（与期望示例一致）
                            if not first_chunk_sent:
                                init_chunk = {
                                    "id": f"chatcmpl-{created_ts}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.model or model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                first_chunk_sent = True
                                yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

                            # content 增量前，优先尝试输出 tool_response（如有）
                            try:
                                tool_resp_text = _extract_tool_response_text(event)
                                if isinstance(tool_resp_text, str):
                                    new_tool_resp = tool_resp_text[len(last_tool_response):]
                                    if new_tool_resp:
                                        tr_chunk = {
                                            "id": f"chatcmpl-{created_ts}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": req.model or model_name,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"tool_response": new_tool_resp},
                                                }
                                            ],
                                        }
                                        yield f"data: {json.dumps(tr_chunk, ensure_ascii=False)}\n\n"
                                        last_tool_response = tool_resp_text
                            except Exception:
                                pass

                            # content 增量
                            if isinstance(acc.get("content"), str):
                                new_content = acc["content"][len(last_content):]
                                if new_content:
                                    chunk = {
                                        "id": f"chatcmpl-{created_ts}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": req.model or model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": new_content},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    last_content = acc["content"]

                            # reasoning_content 增量
                            if isinstance(acc.get("reasoning_content"), str):
                                new_reasoning = acc["reasoning_content"][len(last_reasoning):]
                                if new_reasoning:
                                    chunk = {
                                        "id": f"chatcmpl-{created_ts}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": req.model or model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"reasoning_content": new_reasoning, "tool_calls": []},
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    last_reasoning = acc["reasoning_content"]

                            # tool_calls 增量（如有新增，则逐条下发）
                            tool_calls = acc.get("tool_calls") or []
                            if isinstance(tool_calls, list) and len(tool_calls) > last_tool_calls_len:
                                for i in range(last_tool_calls_len, len(tool_calls)):
                                    # 仅当该 tool_call 的 arguments 为完整 JSON 时才发送
                                    try:
                                        tc = tool_calls[i]
                                        args_raw = ((tc or {}).get("function") or {}).get("arguments", "{}")
                                        parsed_args = json.loads(args_raw)
                                        # 仅当为非空对象时才下发
                                        if not (isinstance(parsed_args, dict) and parsed_args):
                                            continue
                                    except Exception:
                                        # 跳过不完整或非JSON的增量 tool_call
                                        continue
                                    tc_chunk = {
                                        "id": f"chatcmpl-{created_ts}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": req.model or model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"tool_calls": [tool_calls[i]]},
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                                    try:
                                        collected_tool_calls.append(tool_calls[i])
                                    except Exception:
                                        pass
                                last_tool_calls_len = len(tool_calls)

                        elif isinstance(event, dict):
                            # 在 dict 事件中，若未发送首块则先发 role 初始化块
                            if not first_chunk_sent:
                                init_chunk = {
                                    "id": f"chatcmpl-{created_ts}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.model or model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                first_chunk_sent = True
                                yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

                            # 先尝试从该 dict 事件中提取 tool_response
                            tool_resp_text = _extract_tool_response_text([event])
                            if isinstance(tool_resp_text, str):
                                new_tool_resp = tool_resp_text[len(last_tool_response):]
                                if new_tool_resp:
                                    tr_chunk = {
                                        "id": f"chatcmpl-{created_ts}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": req.model or model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"tool_response": new_tool_resp},
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(tr_chunk, ensure_ascii=False)}\n\n"
                                    last_tool_response = tool_resp_text

                            # 然后处理可能的 assistant 内容片段
                            if event.get("role") == "assistant" and event.get("content"):
                                content_piece = event.get("content")
                                chunk = {
                                    "id": f"chatcmpl-{created_ts}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.model or model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": content_piece},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                last_content += content_piece
                        elif isinstance(event, str):
                            # 兼容性：纯字符串片段视为 content 片段
                            if not first_chunk_sent:
                                init_chunk = {
                                    "id": f"chatcmpl-{created_ts}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.model or model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                first_chunk_sent = True
                                yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

                            # 先尝试从字符串片段中提取 tool_response
                            try:
                                tool_resp_text = _extract_tool_response_text([{ "content": event }])
                                if isinstance(tool_resp_text, str):
                                    new_tool_resp = tool_resp_text[len(last_tool_response):]
                                    if new_tool_resp:
                                        tr_chunk = {
                                            "id": f"chatcmpl-{created_ts}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": req.model or model_name,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"tool_response": new_tool_resp},
                                                }
                                            ],
                                        }
                                        yield f"data: {json.dumps(tr_chunk, ensure_ascii=False)}\n\n"
                                        last_tool_response = tool_resp_text
                            except Exception:
                                pass

                            # 然后发送内容
                            chunk = {
                                "id": f"chatcmpl-{created_ts}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": req.model or model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": event},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            last_content += event
                except Exception as e:
                    logger.exception("Exception occurred during streaming generation: %s", e)
                    err = {"error": {"message": str(e), "type": "internal_error", "code": 500}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                # final empty delta with finish_reason
                try:
                    total_s = max(1e-6, time.perf_counter() - gen_t0)
                    out_tok = _count_tokens(last_content)
                    tps = out_tok / total_s if total_s > 0 else math.nan
                    # 记录聚合指标（流式）
                    try:
                        _record_stream_metrics(
                            ttft_ms=(int(ttft_s * 1000) if ("ttft_s" in locals() and ttft_s is not None) else None),
                            total_ms=int(total_s * 1000),
                            out_tokens=out_tok,
                            tps=float(tps) if not math.isnan(tps) else 0.0,
                        )
                    except Exception:
                        pass
                    logger.info(
                        "[METRICS] ttft_ms=%s, total_time_ms=%d, output_tokens=%d, throughput_toks_per_s=%.2f",
                        ("%d" % int(ttft_s * 1000) if ttft_s is not None else "-"), int(total_s * 1000), out_tok, tps, 
                    )
                    # 打印全部请求消息（原始与过滤后）
                    logger.info("All request messages: %s", json.dumps(raw_messages, ensure_ascii=False))
                    # 构造并打印最终全部 messages（包含 assistant 最终内容、tool_calls、可选 tool_response）
                    try:
                        final_messages_log: List[Dict[str, Any]] = list(messages)
                        assistant_snapshot: Dict[str, Any] = {"role": "assistant", "content": last_content}
                        if last_reasoning:
                            assistant_snapshot["reasoning_content"] = last_reasoning
                        if collected_tool_calls:
                            assistant_snapshot["tool_calls"] = collected_tool_calls
                        if last_tool_response:
                            assistant_snapshot["tool_response"] = last_tool_response
                        final_messages_log.append(assistant_snapshot)
                        logger.info("All messages: %s", json.dumps(final_messages_log, ensure_ascii=False))
                    except Exception:
                        logger.exception("Failed to log final all messages (streaming)")
                except Exception:
                    pass
                final_chunk = {
                    "id": f"chatcmpl-{created_ts}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model or model_name,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming path: prefer final aggregated messages if present
        logger.info("Non-streaming response started: model=%s, temperature=%s, top_p=%s, max_tokens=%s", req.model or model_name, req.temperature, req.top_p, req.max_tokens)
        assistant_text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        final_messages: Optional[List[Dict[str, Any]]] = None
        gen_t0 = time.perf_counter()
        try:
            for event in bot.run(messages=messages):
                if isinstance(event, dict):
                    # Reasoning parts (best-effort)
                    if (event.get("type") in ("reasoning", "thinking")) and event.get("content"):
                        reasoning_parts.append(event["content"])
                    # Assistant fragments
                    if event.get("role") == "assistant" and event.get("content"):
                        assistant_text_parts.append(event["content"])
                    # Tool calls in streaming events (best-effort)
                    if event.get("function_call"):
                        fc = event.get("function_call") or {}
                        args_str = _normalize_arguments(fc.get("arguments"))
                        try:
                            parsed_args = json.loads(args_str)
                            if isinstance(parsed_args, dict) and parsed_args:
                                tool_calls.append({
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": args_str,
                                    }
                                })
                        except Exception:
                            # 非完整 JSON（增量中间态）不输出
                            pass
                elif isinstance(event, str):
                    assistant_text_parts.append(event)
                elif isinstance(event, list):
                    final_messages = event
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

        assistant_text: str = ""
        reasoning_content: str = ""
        if final_messages:
            # Extract the last assistant message and aggregate tool calls/reasoning if present
            for item in final_messages:
                if isinstance(item, dict) and item.get("function_call"):
                    fc = item.get("function_call") or {}
                    args_str = _normalize_arguments(fc.get("arguments"))
                    try:
                        parsed_args = json.loads(args_str)
                        if isinstance(parsed_args, dict) and parsed_args:
                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": fc.get("name", ""),
                                    "arguments": args_str,
                                }
                            })
                    except Exception:
                        pass
                # Reasoning keys that might be used
                for key in ("reasoning_content", "thinking_content", "thoughts", "reasoning"):
                    if isinstance(item, dict) and item.get(key):
                        reasoning_parts.append(str(item.get(key)))
            for item in reversed(final_messages):
                if isinstance(item, dict) and item.get("role") == "assistant" and item.get("content"):
                    assistant_text = item.get("content", "").strip()
                    break
        else:
            assistant_text = "".join(assistant_text_parts).strip()

        reasoning_content = "\n".join([p for p in reasoning_parts if p]).strip()

        created_ts = int(time.time())
        resp = {
            "id": f"chatcmpl-{created_ts}",
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model or model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_text,
                        "reasoning_content": reasoning_content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        # 记录非流式的最终回复
        try:
            total_s = time.perf_counter() - gen_t0
            out_tok = _count_tokens(assistant_text)
            tps = out_tok / total_s if total_s > 0 else math.nan
            # 记录聚合指标（非流式）
            try:
                _record_nonstream_metrics(
                    total_ms=int(total_s * 1000),
                    out_tokens=out_tok,
                    tps=float(tps) if not math.isnan(tps) else 0.0,
                )
            except Exception:
                pass
            logger.info(
                "[METRICS] total_time_ms=%d, output_tokens=%d, throughput_toks_per_s=%.2f",
                int(total_s * 1000), out_tok, tps,
            )
            # 同时打印全部请求消息（原始与过滤后）
            logger.info("All request messages: %s", json.dumps(raw_messages, ensure_ascii=False))
            # 打印最终全部 messages（包含 assistant 最终内容、tool_calls）
            try:
                final_messages_log: List[Dict[str, Any]] = list(messages)
                assistant_snapshot: Dict[str, Any] = {"role": "assistant", "content": assistant_text}
                if reasoning_content:
                    assistant_snapshot["reasoning_content"] = reasoning_content
                if tool_calls:
                    assistant_snapshot["tool_calls"] = tool_calls
                final_messages_log.append(assistant_snapshot)
                logger.info("All messages: %s", json.dumps(final_messages_log, ensure_ascii=False))
            except Exception:
                logger.exception("Failed to log final all messages (non-streaming)")
            # 可选：如需完整输出可取消注释
            # logger.info("Full non-streaming response: %s", json.dumps(resp, ensure_ascii=False))
        except Exception:
            pass
        return JSONResponse(resp)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app


def run_api(bot, model_name: str, host: str = "0.0.0.0", port: int = 8001, api_key: Optional[str] = None, allow_cors: bool = False, tokenizer_path: Optional[str] = None):
    import uvicorn

    app = create_app(bot, model_name=model_name, api_key=api_key, allow_cors=allow_cors, tokenizer_path=tokenizer_path)
    uvicorn.run(app, host=host, port=port)
