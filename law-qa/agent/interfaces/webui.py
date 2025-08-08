import os
from qwen_agent.gui import WebUI


def run_webui(bot, avatar_path: str = './chatbot.png'):
    chatbot_config = {}
    if os.path.exists(avatar_path):
        chatbot_config['agent.avatar'] = avatar_path
    else:
        print(
            f"Warning: avatar not found at {avatar_path}, continuing without it."
        )
    WebUI(bot, chatbot_config=chatbot_config).run()
