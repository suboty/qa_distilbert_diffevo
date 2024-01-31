import os
import sys
import time
import logging


def get_logger():
    logger_ = logging.getLogger('root')

    if os.environ.get('LOGGING_LEVEL') and os.environ.get('LOGGING_LEVEL').lower() == 'debug':
        logger_.setLevel(logging.DEBUG)
    else:
        logger_.setLevel(logging.INFO)

    if os.environ.get('LOGGER_HANDLER') and os.environ.get('LOGGER_HANDLER').lower() == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(f'{int(time.time())}.log')

    if os.environ.get('LOGGING_LEVEL') and os.environ.get('LOGGING_LEVEL').lower() == 'debug':
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger_.addHandler(handler)

    return logger_


if os.environ.get('LOGGER_DISABLE') != '1':
    logger = get_logger()
else:
    logger = None
