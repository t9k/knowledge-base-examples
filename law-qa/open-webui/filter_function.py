import ast
import json
from pydantic import BaseModel


class Filter:
    # Valves: Configuration options for the filter
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Initialize valves (optional configuration for the Filter)
        self.valves = self.Valves()
        self.user_prompt_prefix = (
            "提示：法小助必须精准识别用户的真实意图；需要调用检索工具进行查询，除非答案已经在上下文中；"
            "如果用户没有提及法条编号，不得使用法条编号来查询或检索；"
            "如果检索结果为空，应移除过滤表达式，重新调用检索工具。\n\n"
        )
        self.round = 0
        self.tool_calls = {}

    def inlet(self, body: dict) -> dict:
        n = len(body["messages"])
        self.round = (n + 1) // 2

        for m in body["messages"]:
            if m["content"].startswith("\n🔍 正在检索"):
                m["content"] = m["content"].split("个来源\n\n")[-1]

        if n > 1:
            m = body["messages"][-1]
            assert m["role"] == "user"
            m["content"] = self.user_prompt_prefix + m["content"]

        for i in range(self.round - 1, 0, -1):
            tool_call = self.tool_calls.get(i)
            if tool_call:
                tool_call_msg = {
                    "role": "assistant",
                    "content": "<tool_call>\n" + repr(tool_call) + "\n</tool_call>",
                }
                body["messages"].insert(2 * i - 1, tool_call_msg)

        print(f"Round: {self.round}")
        print(f"inlet called: {body}")

        return body

    async def stream(self, event: dict, __event_emitter__=None) -> dict:
        # Modify streamed chunks of model output.
        print(f"stream event: {event}")

        try:
            if event.get("object") == "chat.completion.chunk":
                choices = event.get("choices", [])
                if not choices:
                    return event

                delta = choices[0].get("delta", {})

                tool_call = delta.get("tool_calls")
                tool_response = delta.get("tool_response")
                if tool_call:
                    raw_call = tool_call[0]
                    fn = raw_call["function"]["name"]
                    arguments = raw_call["function"].get("arguments")
                    try:
                        parsed_arguments = (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else (arguments or {})
                        )
                    except Exception:
                        parsed_arguments = arguments

                    self.tool_calls[self.round] = {
                        "name": fn,
                        "arguments": parsed_arguments,
                    }
                    if isinstance(fn, str):
                        if "criminal_law" in fn:
                            self.search_category = "刑法"
                        elif "civil_code" in fn:
                            self.search_category = "民法典"
                        elif "criminal_case" in fn:
                            self.search_category = "刑事案件"
                        elif "civil_case" in fn:
                            self.search_category = "民事案件"

                    # await __event_emitter__(
                    #     {
                    #         "type": "status",  # See the event types list below
                    #         "data": {
                    #             "description": f"🔍 正在检索{self.search_category}...",
                    #             "done": False,
                    #             "hidden": False,
                    #         },
                    #     }
                    # )

                    return {
                        "id": event["id"],
                        "object": event["object"],
                        "created": event["created"],
                        "model": event["model"],
                        "choices": [
                            {
                                "index": choice["index"],
                                "delta": {
                                    "content": f"🔍 正在检索{self.search_category}..."
                                },
                                "finish_reason": None,
                            }
                            for choice in event["choices"]
                        ],
                    }

                elif tool_response:
                    # 解析 <source id="..."> ... </source> 块
                    import re as _re
                    import json as _json

                    blocks = _re.findall(
                        r"<source\s+id=\"(\d+)\">\s*([\s\S]*?)</source>", tool_response
                    )
                    citations = []
                    for sid, body in blocks:
                        try:
                            data = _json.loads(body)
                            name = ((data.get("source") or {}).get("name")) or ""
                            documents = data.get("document") or []
                            distances = data.get("distances") or []
                            citation = {
                                "source": {
                                    "id": sid,
                                    "name": name,
                                    "description": (documents[0] if documents else ""),
                                    "meta": {},
                                },
                                "document": documents,
                                "distances": distances,
                            }
                            citations.append(citation)
                        except Exception as e:
                            print("解析 <source> 块出错:", e)

                    # 逐条发送引用来源给前端
                    for citation in citations:
                        await __event_emitter__(
                            {
                                "type": "source",  # See the event types list below
                                "data": citation,
                            }
                        )

                    # await __event_emitter__(
                    #     {
                    #         "type": "status",  # See the event types list below
                    #         "data": {
                    #             "description": f"📄 检索到 {len(citations)} 个来源",
                    #             "done": False,
                    #             "hidden": False,
                    #         },
                    #     }
                    # )

                    return {
                        "id": event["id"],
                        "object": event["object"],
                        "created": event["created"],
                        "model": event["model"],
                        "choices": [
                            {
                                "index": choice["index"],
                                "delta": {
                                    "content": f"\n📄 检索到 {len(citations)} 个来源"
                                },
                                "finish_reason": None,
                            }
                            for choice in event["choices"]
                        ],
                    }

        except Exception as e:
            print("Filter.stream 运行异常:", e)

        return event

    # def outlet(self, body: dict) -> None:
    #     # This is where you manipulate model outputs.
    #     print(f"outlet called: {body}")
