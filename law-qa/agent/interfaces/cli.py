import pprint
from typing import List, Dict, Any

from qwen_agent.utils.output_beautify import typewriter_print

from core.tokenizer import get_tokenizer


def run_cli(bot, tokenizer_path: str, max_tokens: int):
    try:
        tokenizer = get_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from '{tokenizer_path}': {e}")
        return

    user_prompt_prefix = ("提示：法小助必须精准识别用户的真实意图；需要调用检索工具进行查询，除非答案已经在上下文中；"
                          "如果用户没有提及法条编号，不得使用法条编号来查询或检索；"
                          "如果检索结果为空，应移除过滤表达式，重新调用检索工具。\n\n")

    messages: List[Dict[str, Any]] = []
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
            messages = []
            print('Chat history cleared!\n')
            continue
        if query.strip() in ('/exit', '/quit'):
            print('Bye!')
            break

        if messages:
            query = user_prompt_prefix + query
        messages.append({'role': 'user', 'content': query})

        response_plain_text = ''
        print()
        print('Assistant: ')

        try:
            for response in bot.run(messages=messages):
                # Streaming output.
                response_plain_text = typewriter_print(response,
                                                       response_plain_text)
        except KeyboardInterrupt:
            print("\nInterrupted, returning to input.")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue

        # Append the bot's complete response to the chat history.
        for r in response:
            if r.get('function_call', ''):
                messages.append(r)
            elif r['role'] == 'assistant' and r['content']:
                messages.append(r)
