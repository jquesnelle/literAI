import transformers
import diffusers

def logger_error(func):
    def wrapper():
        transformers.logging.set_verbosity_error()
        diffusers.logging.set_verbosity_error()
        func()
    return wrapper
