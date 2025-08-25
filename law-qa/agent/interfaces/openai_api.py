from typing import List, Optional, Dict, Any
import time
import json
import re
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

# 模块级日志器（避免重复添加 handler）
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


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


def create_app(bot, model_name: str, api_key: Optional[str] = None, allow_cors: bool = False) -> FastAPI:
    app = FastAPI()

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
                last_content = ""
                last_reasoning = ""
                last_tool_calls_len = 0
                last_tool_response = ""
                collected_tool_calls: List[Dict[str, Any]] = []
                try:
                    for event in bot.run(messages=messages):
                        if isinstance(event, list):
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
                    logger.info(
                        "Streaming finished: content_len=%d, reasoning_len=%d, tool_calls=%d, tool_response_len=%d",
                        len(last_content), len(last_reasoning), last_tool_calls_len, len(last_tool_response),
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
                        logger.info("All messages (final): %s", json.dumps(final_messages_log, ensure_ascii=False))
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
        assistant_text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        final_messages: Optional[List[Dict[str, Any]]] = None
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
            logger.info(
                "Non-streaming finished: content_len=%d, reasoning_len=%d, tool_calls=%d",
                len(assistant_text), len(reasoning_content), len(tool_calls),
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
                logger.info("All messages (final): %s", json.dumps(final_messages_log, ensure_ascii=False))
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


def run_api(bot, model_name: str, host: str = "0.0.0.0", port: int = 8001, api_key: Optional[str] = None, allow_cors: bool = False):
    import uvicorn

    app = create_app(bot, model_name=model_name, api_key=api_key, allow_cors=allow_cors)
    uvicorn.run(app, host=host, port=port)
