#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import re

from ..infra.types import (Any, ClassVar, Optional, Self, Sequence, TYPE_CHECKING)


if TYPE_CHECKING:
    import patchlm.config.config


range_pat = re.compile(r'^(-?\d*):(-?\d*)$')
regex_pat = re.compile(r'^/(.*)/$')


class KeyMatcher:

    precedence: int = 10
    template: str
    value: Any
    template_pattern: ClassVar[re.Pattern] = re.compile(r'^_$')
    submatchers: Sequence[type['KeyMatcher']]

    def __init__(self, template: str, value: Any, match: re.Match = None, submatchers: Sequence[type['KeyMatcher']] = ()):
        self.template = template
        self.value = value
        self.submatchers = submatchers
        if match is not None:
            self.init_match(match)

    def init_match(self, match: re.Match):
        self.init_groups(*match.groups())

    def init_groups(self, *groups: str):
        pass

    def matches(self, key: str, **kwarg) -> bool:
        return True

    @classmethod
    def build(cls, template: str, value: Any, **kwargs) -> Optional[Self]:
        match = cls.template_pattern.match(template)
        if match is not None:
            return cls(template, value, match=match, **kwargs)
        return None

    @classmethod
    def build_first(cls, template: str, value: Any, matchers: Sequence[type['KeyMatcher']] = None, **kwargs) -> Optional[Self]:
        if matchers is not None:
            kwargs.setdefault('submatchers', matchers)
            for matcher_cls in matchers:
                if matcher := matcher_cls.build(template, value, **kwargs):
                    return matcher
        return None

    def __lt__(self, other):
        return self.precedence < other.precedence

    def __repr__(self):
        return f'{self.__class__.__name__}({self.template})'


class BinaryOpKeyMatcher(KeyMatcher):

    left: KeyMatcher
    right: KeyMatcher

    def init_groups(self, left: str, right: str):
        for submatcher in self.submatchers:
            if matcher := submatcher.build(left, self.value, submatchers=self.submatchers):
                self.left = matcher
            if matcher := submatcher.build(right, self.value, submatchers=self.submatchers):
                self.right = matcher
        self.precedence = min(self.left.precedence, self.right.precedence)-1


class AndKeyMatcher(BinaryOpKeyMatcher):

    template_pattern = re.compile(r'^([^&]+)&(.+)$')

    def matches(self, key: str, **kwargs) -> bool:
        return self.left.matches(key, **kwargs) and self.right.matches(key, **kwargs)


class OrKeyMatcher(BinaryOpKeyMatcher):

    template_pattern = re.compile(r'^([^|]+)\|(.+)$')

    def matches(self, key: str, **kwargs) -> bool:
        return self.left.matches(key, **kwargs) or self.right.matches(key, **kwargs)


default_matchers = KeyMatcher, AndKeyMatcher, OrKeyMatcher


class RegexKeyMatcher(KeyMatcher):

    precedence = 5
    template_pattern = regex_pat
    pattern: re.Pattern

    def init_groups(self, pattern: str):
        self.pattern = re.compile(pattern)

    def matches(self, key: str, **kwargs) -> bool:
        return self.pattern.match(key) is not None


class IndexKeyMatcher(KeyMatcher):

    def matches(self, key: str, config: 'patchlm.config.config.ListConfig' = None, **kwargs) -> bool:
        idx = int(key)
        return self.matches_index(idx, config=config, **kwargs)

    def matches_index(self, idx: int, config: 'patchlm.config.config.ListConfig' = None, **kwargs) -> bool:
        return False


class RangeKeyMatcher(IndexKeyMatcher):

    precedence = 3
    template_pattern = range_pat
    start: Optional[int]
    end: Optional[int]

    def init_groups(self, start: str, end: str):
        self.start = int(start) if start else None
        self.end = int(end) if end else None

    def matches_index(self, idx: int, config: 'patchlm.config.config.ListConfig' = None, **kwargs) -> bool:
        count = config.count if config else 0
        if start := self.start:
            start = count + start if start < 0 else start
            if end := self.end:
                end = count + end if end < 0 else end
                return start <= idx < end
            else:
                return start <= idx
        else:
            if end := self.end:
                end = count + end if end < 0 else end
                return idx < end
            else:
                return True


class ModuloKeyMatcher(IndexKeyMatcher):

    precedence = 2
    template_pattern = re.compile(r'%(\d+)=(\d+)$')
    size: int
    mod: int

    def init_groups(self, size: str, mod: str = None):
        self.size = int(size)
        self.mod = int(mod) if mod else 0

    def matches_index(self, idx: int, config: 'patchlm.config.config.ListConfig' = None, **kwargs) -> bool:
        return idx % self.size == self.mod


list_matchers = *default_matchers, RangeKeyMatcher, RegexKeyMatcher, ModuloKeyMatcher

