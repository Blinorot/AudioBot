import hydra

from src.handlers import init_all_models
from src.streaming import init_stream


@hydra.main(version_base=None, config_path="src/configs", config_name="audio_bot")
def init_run(config):
    all_models, history = init_all_models(config)
    init_stream(all_models, history, config)


if __name__ == "__main__":
    init_run()
