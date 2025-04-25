import logging
import nltk
import numpy as np


def set_logging(log_file, level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        level=level,
        filemode='w'
    )


def truncate(text, max_length):
    if max_length <= 0:
        return text
    words = text.split(' ')
    if len(words) <= max_length:
        return text
    return ' '.join(words[:max_length])


