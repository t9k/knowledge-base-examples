from typing import Dict, List, Any, Optional
from .tokenizer import count_tokens


class ConversationStore:

    def get(self, session_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def set(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):

    def __init__(self, tokenizer_path: str, max_tokens: int) -> None:
        self._db: Dict[str, List[Dict[str, Any]]] = {}
        self._tokenizer_path = tokenizer_path
        self._max_tokens = max_tokens

    def get(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._db.get(session_id, []))

    def set(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._db[session_id] = truncate_messages(messages,
                                                 self._tokenizer_path,
                                                 self._max_tokens)

    def clear(self, session_id: str) -> None:
        self._db.pop(session_id, None)


def truncate_messages(messages: List[Dict[str, Any]], tokenizer_path: str,
                      max_tokens: int) -> List[Dict[str, Any]]:
    while count_tokens(messages, tokenizer_path) > max_tokens and messages:
        messages.pop(0)
    return messages


def append_turn(messages: List[Dict[str, Any]], user_msg: Dict[str, Any],
                events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages.append(user_msg)
    for e in events:
        if e.get('function_call'):
            messages.append(e)
        elif e.get('role') == 'assistant' and e.get('content'):
            messages.append(e)
    return messages
