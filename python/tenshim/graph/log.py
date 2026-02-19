#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

import os
from typing import Callable

ERROR = 0
WARN = 1
INFO = 2
DEBUG = 3
TRACE = 4

log_levels: list[str] = ['error', 'warn', 'info', 'debug', 'trace']
log_prefixes: list[str] = ['error:', ' warn:', ' info:', 'debug:', 'trace:']
log_level_map: dict[str, int] = {name: i for i, name in enumerate(log_levels)}

log_level_name: str = os.environ.get('LOG_LEVEL', 'debug').lower()


log_level: int = log_level_map[log_level_name]


def nolog(*args, **kwargs) -> None:
    pass


# noinspection PyDecorator
@classmethod
def nolog_method(cls, *args, **kwargs) -> None:
    pass


def make_logger(level: int) -> Callable[..., None]:
    if level <= log_level:
        prefix = log_prefixes[level]
        def logger(msg: str, *args, **kwargs) -> None:
            print(prefix, msg, *args, **kwargs)
        logger.__qualname__ = log_levels[level]
        return logger
    return nolog


def make_log_method(level: int) -> classmethod:
    prefix = log_prefixes[level]
    def logger(cls, msg: str, *args, **kwargs) -> None:
        cls.log(msg, *args, prefix=prefix, level=level, **kwargs)
    logger.__qualname__ = log_levels[level]
    return classmethod(logger)


error = make_logger(ERROR)
warn = make_logger(WARN)
info = make_logger(INFO)
debug = make_logger(DEBUG)
trace = make_logger(TRACE)

loggers = error, warn, info, debug, trace

log_methods = tuple(make_log_method(level) for level in range(len(log_levels)))


def turn_logging_on(obj, from_level: int, to_level: int) -> None:
    for level in range(from_level, to_level):
        setattr(obj, log_levels[level], loggers[level])


def turn_logging_off(obj, from_level: int, to_level: int) -> None:
    for level in range(from_level, to_level):
        setattr(obj, log_levels[level], nolog)


def change_logging(obj, from_level: int, to_level: int) -> None:
    info(f'changing logging for {obj} from {from_level} to {to_level}')
    if from_level < to_level:
        turn_logging_on(obj, from_level+1, to_level+1)
    elif from_level > to_level:
        turn_logging_off(obj, to_level+1, from_level+1)


class Logging:

    __slots__ = ()

    _log_level: int = -1

    def __init_subclass__(cls, log: int | str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if log is not None:
            cls.set_log_level(log)

    @classmethod
    def log(cls, msg: str, *args, prefix: str = None, level: int = None, **kwargs) -> None:
        if prefix is None:
            prefix = '  LOG:' if level is None else log_prefixes[level]
        print(prefix, msg, *args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def warn(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def info(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def debug(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def trace(cls, *args, **kwargs) -> None:
        pass

    @classmethod
    def set_log_level(cls, level: int) -> None:
        old_level = cls._log_level
        if level != old_level:
            cls._log_level = level
            trace(f'changing logging for {cls.__qualname__} from {old_level} to {level}')
            for l in range(old_level+1, level+1, -1 if level < old_level else 1):
                logger = nolog_method if l > level else log_methods[l]
                trace(f'setting logger for {log_levels[l]} to {logger}')
                setattr(cls, log_levels[l], logger)


Logging.set_log_level(log_level)
