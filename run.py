import hydra

from src.handlers import init_all_models
from src.streaming import init_stream


@hydra.main(version_base=None, config_path="src/configs", config_name="audio_bot")
def init_run(config):
    """
    Initialize all models and history.

    Start chatting with an assistant.

    Args:
        config: Hydra config to control the assistant workflow.
    """
    all_models, history = init_all_models(config)
    init_stream(all_models, history, config)


if __name__ == "__main__":
    init_run()
