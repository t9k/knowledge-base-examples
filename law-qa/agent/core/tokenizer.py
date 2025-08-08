from typing import List, Dict, Any
from transformers import AutoTokenizer

_tokenizer_cache = {}


def get_tokenizer(tokenizer_path: str):
    if tokenizer_path not in _tokenizer_cache:
        _tokenizer_cache[tokenizer_path] = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
    return _tokenizer_cache[tokenizer_path]


MESSAGE_WRAPPING_TOKENS = 4
FUNCTION_CALL_WRAPPING_TOKENS = 8


def count_tokens(messages: List[Dict[str, Any]], tokenizer_path: str) -> int:
    tokenizer = get_tokenizer(tokenizer_path)
    total_tokens = 0
    texts = []
    for msg in messages:
        content = msg.get('content', '')
        function_call = msg.get('function_call', '')
        if content:
            texts.append(content)
            total_tokens += MESSAGE_WRAPPING_TOKENS
        elif function_call:
            import json
            texts.append(json.dumps(function_call, ensure_ascii=False))
            total_tokens += FUNCTION_CALL_WRAPPING_TOKENS
    return len(tokenizer.encode(''.join(texts))) + total_tokens
