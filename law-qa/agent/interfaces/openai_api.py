from typing import List, Optional, Dict, Any
import time
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse


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

        # Filter out system messages; bot already has system prompt configured
        messages: List[Dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                continue
            messages.append({"role": m.role, "content": m.content or ""})

        if req.stream:
            created_ts = int(time.time())

            def _accumulate_oai_message(data: List[Dict[str, Any]]) -> Dict[str, Any]:
                # 将 Qwen Agent 累计消息合并为一个 OpenAI assistant message 的累计视图
                message: Dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
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
                        message["tool_calls"].append({
                            "id": (item.get("extra", {}) or {}).get("function_id", f"call_{len(message['tool_calls'])}"),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("arguments") if isinstance(fc.get("arguments"), str) else json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                            }
                        })
                return message

            def event_generator():
                # 维护累计状态，做增量 diff
                first_chunk_sent = False
                last_content = ""
                last_reasoning = ""
                last_tool_calls_len = 0
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
                                last_tool_calls_len = len(tool_calls)

                        elif isinstance(event, dict):
                            # 兼容性：个别实现可能产出 dict 片段（content）
                            if event.get("role") == "assistant" and event.get("content"):
                                content_piece = event.get("content")
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
                    err = {"error": {"message": str(e), "type": "internal_error", "code": 500}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                # final empty delta with finish_reason
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
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("arguments") if isinstance(fc.get("arguments"), str) else json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                            }
                        })
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
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": fc.get("arguments") if isinstance(fc.get("arguments"), str) else json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                        }
                    })
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
        return JSONResponse(resp)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app


def run_api(bot, model_name: str, host: str = "0.0.0.0", port: int = 8001, api_key: Optional[str] = None, allow_cors: bool = False):
    import uvicorn

    app = create_app(bot, model_name=model_name, api_key=api_key, allow_cors=allow_cors)
    uvicorn.run(app, host=host, port=port)
