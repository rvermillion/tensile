#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

import time

from os import environ
from typing import Any, Callable, ClassVar, Optional, Self, TypeAlias
from .represent import Representable
from .util import class_qname
from . import deployment

TRACE = 0
DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4


LogLevel = int

log_levels: list[str] = ['trace', 'debug', 'info', 'warn', 'error']
log_levels_upper: list[str] = [s.upper() for s in log_levels]

log_level_map: dict[str, LogLevel] = {level: i for i, level in enumerate(log_levels)}

log_level_prefixes: list[str] = [
    'TRACE',
    'DEBUG',
    ' INFO',
    ' WARN',
    'ERROR',
]


def coerce_log_level(level: str|LogLevel) -> LogLevel:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level = level.lower().strip(' \t\n\r')
        if level in log_level_map:
            return log_level_map[level]
    raise ValueError(f'Invalid log level: {level}')


log_level: LogLevel = coerce_log_level(environ.get('LOG_LEVEL', 'info'))

# def write(msg: str, *args, level: LogLevel = None, **kwargs):
#     if level is None:
#         level = log_level
#     if level >= log_level:
#         msg = msg.format(*args, **kwargs)
#         pfx = log_level_prefixes[level]
#         global_log(pfx + msg)
#         # global_log(msg, *args, **kwargs)

def trace(*args, **kwargs):
    write(*args, level=TRACE, **kwargs)

def debug(*args, **kwargs):
    write(*args, level=DEBUG, **kwargs)

def info(*args, **kwargs):
    write(*args, level=INFO, **kwargs)

def warn(*args, **kwargs):
    write(*args, level=WARN, **kwargs)

def error(*args, **kwargs):
    write(*args, level=ERROR, **kwargs)



class LogEvent(Representable):

    __slots__ = ('message', 'log_level', 'origin', 'timestamp', 'extra')

    message: str
    log_level: LogLevel
    origin: str
    timestamp: float
    extra: Optional[dict]

    def __init__(self, message: str, level: LogLevel, origin: str, timestamp: float = None, extra: dict = None):
        self.message = message
        self.log_level = level
        self.origin = origin
        self.timestamp = time.time() if timestamp is None else timestamp
        self.extra = extra if extra else None

    @property
    def level_prefix(self) -> str:
        return log_level_prefixes[self.log_level]

    @property
    def level(self) -> str:
        return log_levels_upper[self.log_level]

    @property
    def time(self) -> str:
        return time.strftime('%Y%m%d:%H%M%S', time.localtime(self.timestamp))
        # return time.localtime(self.timestamp)

    def __format__(self, format_spec):
        if format_spec:
            f = format_spec[0]
            if f == 'T':
                return time.strftime(format_spec[1:], time.localtime(self.timestamp))
            if f == 'L':
                return self.level
        return repr(self)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        return self._repr_attrs('level', 'origin', 'time', 'message')


class LogObject(Representable):

    __slots__ = ()

    def configure(self, config: dict, /, **kwargs) -> Self:
        if kwargs:
            if config:
                kwargs.update(config)
            return self._configure(kwargs)
        elif config:
            return self._configure(config)

    def _configure(self, config: dict) -> Self:
        return self


class LogOutput(LogObject):

    __slots__ = ()

    def write(self, event: LogEvent):
        raise NotImplementedError()


LogFormat = Callable[[LogEvent], str]

default_format: LogFormat = '{0:T%Y%m%d:%H%M%S} {0.level:>5} {0.origin:40} {0.message}'.format


class TextLogOutput(LogOutput):

    __slots__ = ('format',)

    format: LogFormat

    # noinspection PyShadowingBuiltins
    def __init__(self, format: LogFormat = None):
        if format is None:
            format = default_format
        self.format = format

    def _configure(self, config: dict) -> Self:
        super()._configure(config)
        if fmt := config.get('format'):
            self.set_format(fmt)
        return self

    # noinspection PyShadowingBuiltins
    def set_format(self, format: str|LogFormat = default_format):
        if isinstance(format, str):
            format = format.format
        self.format = format

    def write(self, event: LogEvent):
        text = self.format(event)
        self.write_text(text)

    def write_text(self, text: str) -> None:
        raise NotImplementedError()


class ConsoleLogOutput(TextLogOutput):

    __slots__ = ()

    def write_text(self, text: str) -> None:
        print(text)


class CompositeLogOutput(LogOutput):

    __slots__ = ('outputs',)

    outputs: list[LogOutput]

    def __init__(self, outputs: list[LogOutput] = None):
        if outputs is None:
            outputs = []
        self.outputs = outputs

    def write(self, event: LogEvent):
        for output in self.outputs:
            output.write(event)


default_output = ConsoleLogOutput()


class Logger(LogOutput):

    __slots__ = ('owner', 'output', 'level')

    owner: Optional[type['Logging']]
    output: LogOutput
    level: LogLevel

    def __init__(self, level: LogLevel, owner: type['Logging'] = None, output: LogOutput = None):
        if output is None:
            output = default_output
        self.owner = owner
        self.level = level
        self.output = output

    @property
    def name(self) -> str:
        return class_qname(self.owner) if self.owner else 'root'

    def _configure(self, config: dict) -> Self:
        super()._configure(config)
        if level := config.get('level'):
            self.set_level(level)
        if output := config.get('output'):
            self.output = self.output.configure(output)
        return self

    def set_level(self, level: str | LogLevel):
        self.level = coerce_log_level(level)

    def log(self, msg: str, *args, level: LogLevel, source: type = None, timestamp: float = None, extra: dict = None):
        if level >= self.level:
            if extra:
                message = msg.format(*args, **extra)
            else:
                message = msg.format(*args)
            origin = class_qname(source) if source else self.name
            event = LogEvent(message, level, origin, timestamp, extra)
            self.output.write(event)

    def debug(self, msg: str, *args):
        if self.level <= DEBUG:
            self.log(msg, *args, level=DEBUG)

    def info(self, msg: str, *args):
        if self.level <= INFO:
            self.log(msg, *args, level=INFO)

    def warn(self, msg: str, *args):
        if self.level <= WARN:
            self.log(msg, *args, level=WARN)

    def error(self, msg: str, *args):
        if self.level <= ERROR:
            self.log(msg, *args, level=ERROR)

    def extend(self, owner: type['Logging'], level: str|LogLevel = None, output: LogOutput = None):
        if level is None: level = self.level
        else: level = coerce_log_level(level)
        if output is None: output = self.output
        return Logger(level, owner, output)

    def _repr_args(self, **options) -> str:
        return self.name


root_logger = Logger(log_level)
write = root_logger.log


class Logging:

    __slots__ = ()

    verbose: int = 1

    # noinspection PyMethodMayBeStatic
    def log(self, msg: str, *args, level: LogLevel = None):
        self.class_log(msg, *args, level=level)

    def debug(self, msg: str, *args):
        if self.logger.level <= DEBUG:
            self.log(msg, *args, level=DEBUG)

    def info(self, msg: str, *args):
        if self.logger.level <= INFO:
            self.log(msg, *args, level=INFO)

    def warn(self, msg: str, *args):
        if self.logger.level <= WARN:
            self.log(msg, *args, level=WARN)

    def error(self, msg: str, *args):
        if self.logger.level <= ERROR:
            self.log(msg, *args, level=ERROR)

    @classmethod
    def class_log(cls, msg: str, *args, level: LogLevel = None, **extra):
        cls.logger.log(msg, *args, level=level, source=cls, extra=extra)
        # write(*args, level=level, **kwargs)

    @classmethod
    def class_debug(cls, msg: str, *args):
        if cls.logger.level <= DEBUG:
            cls.class_log(msg, *args, level=DEBUG)

    @classmethod
    def class_info(cls, msg: str, *args):
        if cls.logger.level <= INFO:
            cls.class_log(msg, *args, level=INFO)

    @classmethod
    def class_warn(cls, msg: str, *args):
        if cls.logger.level <= WARN:
            cls.class_log(msg, *args, level=WARN)

    @classmethod
    def class_error(cls, msg: str, *args):
        if cls.logger.level <= ERROR:
            cls.class_log(msg, *args, level=ERROR)

    def __init_subclass__(cls, log: int | str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if log is not None:
            cls.logger = cls.logger.extend(cls, level=log)

    logger: ClassVar[Logger] = root_logger
