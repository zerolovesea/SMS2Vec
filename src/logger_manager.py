import sys
from loguru import logger as loguru_logger

class LoggerManager:
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = loguru_logger
            cls._logger.remove()
            cls._logger.add(sink=sys.stderr, format="{time:YYYY-MM-DD HH:mm} | {level} | {message}")
        return cls._logger
