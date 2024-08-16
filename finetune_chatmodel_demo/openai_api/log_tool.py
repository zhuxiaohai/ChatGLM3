import logging
from typing import Union

def get_formatter():
    try:
        import colorlog
        formatter = colorlog.ColoredFormatter(
            '[%(log_color)s%(levelname)s]\t%(name)s: %(message)s%(reset)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    except ModuleNotFoundError:
        formatter = logging.Formatter('[%(levelname)s]\t%(name)s: %(message)s')

    return formatter


class ConsoleLogger(logging.Logger):
    def __init__(self, name: str, log_level: Union[int, str]):
        super(ConsoleLogger, self).__init__(name, log_level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(get_formatter())
        self.addHandler(console_handler)
        self.console_handler = console_handler

    def debug(self, *args, **kwargs):
        super(ConsoleLogger, self).debug(*args, **kwargs)
        self.console_handler.flush()

    def info(self, *args, **kwargs):
        super(ConsoleLogger, self).info(*args, **kwargs)
        self.console_handler.flush()

    def warning(self, *args, **kwargs):
        super(ConsoleLogger, self).warning(*args, **kwargs)
        self.console_handler.flush()

    def error(self, *args, **kwargs):
        super(ConsoleLogger, self).error(*args, **kwargs)
        self.console_handler.flush()

    def critical(self, *args, **kwargs):
        super(ConsoleLogger, self).critical(*args, **kwargs)
        self.console_handler.flush()
