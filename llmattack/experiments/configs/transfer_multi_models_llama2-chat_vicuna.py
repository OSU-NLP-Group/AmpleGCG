import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "lmsys/vicuna-7b-v1.5"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "lmsys/vicuna-7b-v1.5"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["llama-2", "vicuna"]
    config.devices = ["cuda:0", "cuda:1"]

    return config