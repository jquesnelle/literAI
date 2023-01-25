import transformers
import diffusers
import os
import unicodedata
import re

_BASE_OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_output_dir(title: str, *subdirs) -> str:
    dir =  os.path.join(_BASE_OUTPUT_DIR, slugify(title), *subdirs)
    os.makedirs(dir, exist_ok=True)
    return dir

def set_base_output_dir(path: str):
    global _BASE_OUTPUT_DIR
    _BASE_OUTPUT_DIR = path

def get_base_output_dir() -> str:
    return _BASE_OUTPUT_DIR

def logger_error(func):
    def wrapper(*args, **kwargs):
        transformers.logging.set_verbosity_error()
        diffusers.logging.set_verbosity_error()
        func(*args, **kwargs)
    return wrapper
