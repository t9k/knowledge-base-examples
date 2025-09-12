import asyncio
import argparse
from typing import Any, Dict, Tuple, List
from fastmcp import Client
from pprint import pprint

async def run_for_client(base_url: str) -> Tuple[str, Dict[str, Any]]:
    """与指定 MCP 服务器交互并返回结果。"""
    result: Dict[str, Any] = {"tools": None, "resources": None, "prompts": None}
    try:
        async with Client(base_url) as client:
            # Basic server interaction
            await client.ping()

            # List available operations
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            result["tools"] = tools
            result["resources"] = resources
            result["prompts"] = prompts
    except Exception as exc:  # noqa: BLE001 - 简单打印错误并继续其他客户端
        result["error"] = repr(exc)
    return base_url, result

async def main(clients: List[str]) -> None:
    """串行测试传入的 MCP 服务器 URL 列表。"""
    for url in clients:
        base_url, data = await run_for_client(url)
        print(f"\n===== {base_url} =====")
        if "error" in data:
            print(f"Error: {data['error']}")
            continue
        print("Tools:")
        pprint(data["tools"])
        print("Resources:")
        pprint(data["resources"])
        print("Prompts:")
        pprint(data["prompts"]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP servers by listing tools/resources/prompts.")
    parser.add_argument("clients", nargs="+", help="One or more MCP base URLs, e.g. https://host/mcp/foo/")
    args = parser.parse_args()
    asyncio.run(main(args.clients))
