from src.handlers import init_all_models
from src.streaming import init_stream

def init_run(device="cpu"):
    all_models, metadata = init_all_models(device)
    init_stream(all_models, metadata, device)
        

if __name__ == "__main__":
    init_run()