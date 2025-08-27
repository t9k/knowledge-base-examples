import pprint
from typing import List, Dict, Any
import time
import math
from core.tokenizer import get_tokenizer

from qwen_agent.utils.output_beautify import typewriter_print

from core.conversation import InMemoryConversationStore, append_turn


def run_cli(bot, tokenizer_path: str, max_tokens: int):
    # 使用会话存储，基于 tokenizer+max_tokens 做历史截断
    store = InMemoryConversationStore(tokenizer_path=tokenizer_path,
                                      max_tokens=max_tokens)
    tokenizer = get_tokenizer(tokenizer_path)
    session_id = 'cli'

    user_prompt_prefix = ("提示：法小助必须精准识别用户的真实意图；需要调用检索工具进行查询，除非答案已经在上下文中；"
                          "如果用户没有提及法条编号，不得使用法条编号来查询或检索；"
                          "如果检索结果为空，应移除过滤表达式，重新调用检索工具。\n\n")

    messages: List[Dict[str, Any]] = store.get(session_id)
    while True:
        print()
        query = input('\nUser: ')
        if query.strip() == '':
            continue
        if query.strip() == '/history':
            print('\n=== Chat History ===')
            pprint.pprint(messages, width=120)
            continue
        if query.strip() == '/clear':
            store.clear(session_id)
            messages = []
            print('Chat history cleared!\n')
            continue
        if query.strip() in ('/exit', '/quit'):
            print('Bye!')
            break

        # 取最新历史（已按 token 截断）
        messages = store.get(session_id)

        # 组织当前 user 消息（首轮不加提示，后续轮次加前缀）
        content = (user_prompt_prefix + query) if messages else query
        user_msg: Dict[str, Any] = {'role': 'user', 'content': content}

        response_plain_text = ''
        print()
        print('Assistant: ')

        try:
            # 传入“历史 + 当前 user 消息”
            first_token_emitted = False
            gen_t0 = time.perf_counter()
            for response in bot.run(messages=messages + [user_msg]):
                # Streaming output.
                if not first_token_emitted:
                    first_token_emitted = True
                    ttft_s = max(0.0, time.perf_counter() - gen_t0)
                response_plain_text = typewriter_print(response,
                                                       response_plain_text)
        except KeyboardInterrupt:
            print("\nInterrupted, returning to input.")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue

        # Append the bot's complete response to the chat history.
        sanitized_events: List[Dict[str, Any]] = []
        for r in response:
            if r.get('function_call', ''):
                sanitized_events.append(r)
            elif r.get('role') == 'assistant' and r.get('content'):
                content = r['content']
                if not content.endswith('\n</think>\n\n'):
                    content = content.split('\n</think>\n\n')[-1]
                sanitized_events.append({'role': 'assistant', 'content': content})

        messages = append_turn(messages, user_msg, sanitized_events)
        # 统计输出 token 数与吞吐
        try:
            total_s = time.perf_counter() - gen_t0
            out_text = "".join([e.get('content', '') for e in sanitized_events if e.get('role') == 'assistant'])
            out_tok = len(tokenizer.encode(out_text))
            tps = out_tok / total_s if total_s > 0 else math.nan
            print(f"\n[METRICS] TTFT: {int(ttft_s*1000)} ms, Total: {int(total_s*1000)} ms, Output tokens: {out_tok}, TPS: {tps:.2f}")
        except Exception:
            pass
        store.set(session_id, messages)
