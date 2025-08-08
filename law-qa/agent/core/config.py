import os
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    # Modes
    mode: Optional[str] = None  # 'cli' | 'webui' | 'api'

    # LLM
    model: str = 'Qwen3-32B'
    model_server: str = 'http://127.0.0.1:8000/v1'
    api_key: str = 'EMPTY'
    temperature: float = 0.0
    top_p: float = 0.95
    enable_thinking: bool = True
    thinking_budget: int = 8192

    # Tokenizer and history
    tokenizer_path: str = 'Qwen/Qwen3-32B'
    max_tokens: int = 10000

    # MCP servers
    enable_law_searcher: bool = False
    enable_case_searcher: bool = False
    enable_reranker: bool = False
    law_searcher_url: str = os.getenv('LAW_SEARCHER_URL', 'https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/')
    case_searcher_url: str = os.getenv('CASE_SEARCHER_URL', 'https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/')
    reranker_url: str = os.getenv('RERANKER_URL', 'https://home.qy.t9kcloud.cn/mcp/reranker/mcp/')

    # API server
    api_host: str = os.getenv('AGENT_API_HOST', '0.0.0.0')
    api_port: int = int(os.getenv('AGENT_API_PORT', '8001'))
    api_auth_key: Optional[str] = os.getenv('AGENT_API_KEY')
    allow_cors: bool = False

    # WebUI
    avatar_path: str = './chatbot.png'


def build_config_from_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description='Legal assistant agent')

    parser.add_argument('--mode', type=str, choices=['cli', 'webui', 'api'], default=None)

    # LLM
    parser.add_argument('--model', type=str, default=os.getenv('AGENT_MODEL', 'Qwen3-32B'))
    parser.add_argument('--model-server', type=str, default=os.getenv('AGENT_MODEL_SERVER', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--temperature', type=float, default=float(os.getenv('AGENT_TEMPERATURE', '0.0')))
    parser.add_argument('--top-p', type=float, default=float(os.getenv('AGENT_TOP_P', '0.95')))

    # thinking switch
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--enable-thinking', dest='enable_thinking', action='store_true', help='Enable thinking mode')
    group.add_argument('--disable-thinking', dest='enable_thinking', action='store_false', help='Disable thinking mode')
    parser.set_defaults(enable_thinking=True)
    
    parser.add_argument('--thinking-budget', type=int, default=int(os.getenv('AGENT_THINKING_BUDGET', '8192')))

    # Tokenizer & history
    parser.add_argument('--tokenizer-path', type=str, default=os.getenv('AGENT_TOKENIZER_PATH', 'Qwen/Qwen3-32B'))
    parser.add_argument('--max-tokens', type=int, default=int(os.getenv('AGENT_MAX_TOKENS', '10000')))

    # MCP
    parser.add_argument('--law-searcher', action='store_true')
    parser.add_argument('--case-searcher', action='store_true')
    parser.add_argument('--reranker', action='store_true')
    parser.add_argument('--law-searcher-url', type=str, default=os.getenv('LAW_SEARCHER_URL', 'https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/'))
    parser.add_argument('--case-searcher-url', type=str, default=os.getenv('CASE_SEARCHER_URL', 'https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/'))
    parser.add_argument('--reranker-url', type=str, default=os.getenv('RERANKER_URL', 'https://home.qy.t9kcloud.cn/mcp/reranker/mcp/'))

    # API
    parser.add_argument('--api-host', type=str, default=os.getenv('AGENT_API_HOST', '0.0.0.0'))
    parser.add_argument('--api-port', type=int, default=int(os.getenv('AGENT_API_PORT', '8001')))
    parser.add_argument('--api-key', type=str, default=os.getenv('AGENT_API_KEY'))
    parser.add_argument('--allow-cors', action='store_true')

    # WebUI
    parser.add_argument('--webui', action='store_true')
    parser.add_argument('--avatar', type=str, default=os.getenv('AGENT_AVATAR_PATH', './chatbot.png'))

    args = parser.parse_args()

    cfg = AgentConfig(
        mode=args.mode,
        model=args.model,
        model_server=args.model_server,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
        tokenizer_path=args.tokenizer_path,
        max_tokens=args.max_tokens,
        enable_law_searcher=args.law_searcher,
        enable_case_searcher=args.case_searcher,
        enable_reranker=args.reranker,
        law_searcher_url=args.law_searcher_url,
        case_searcher_url=args.case_searcher_url,
        reranker_url=args.reranker_url,
        api_host=args.api_host,
        api_port=args.api_port,
        api_auth_key=args.api_key,
        allow_cors=args.allow_cors,
        avatar_path=args.avatar,
    )

    return cfg 