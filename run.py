from src.handlers import init_all_models
from src.streaming import init_stream


def init_run(device="cpu"):
    all_models, history = init_all_models(device)
    init_stream(all_models, history, device)


if __name__ == "__main__":
    init_run()
