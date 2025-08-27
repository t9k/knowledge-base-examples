from core.config import build_config_from_args
from core.bot import create_bot
from interfaces.cli import run_cli
from interfaces.webui import run_webui
from interfaces.openai_api import run_api


def main():
    cfg = build_config_from_args()
    bot = create_bot(cfg)

    if cfg.mode == 'api':
        print(f"Starting API server at http://{cfg.api_host}:{cfg.api_port}")
        run_api(bot,
                model_name='law-assistant',
                host=cfg.api_host,
                port=cfg.api_port,
                api_key=cfg.api_auth_key,
                allow_cors=cfg.allow_cors,
                tokenizer_path=cfg.tokenizer_path)
    elif cfg.mode == 'webui':
        run_webui(bot, avatar_path=cfg.avatar_path)
    else:
        run_cli(bot,
                tokenizer_path=cfg.tokenizer_path,
                max_tokens=cfg.max_tokens)


if __name__ == '__main__':
    main()
