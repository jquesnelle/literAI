import gc
import torch


def free_memory_after(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        torch.cuda.empty_cache()
        gc.collect()
    return wrapper