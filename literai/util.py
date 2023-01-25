import transformers
import diffusers
import os
import unicodedata
import re


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
    dir = os.path.join(os.getcwd(), 'output', slugify(title), *subdirs)
    os.makedirs(dir, exist_ok=True)
    return dir


def logger_error(func):
    def wrapper(*args, **kwargs):
        transformers.logging.set_verbosity_error()
        diffusers.logging.set_verbosity_error()
        func(*args, **kwargs)
    return wrapper
