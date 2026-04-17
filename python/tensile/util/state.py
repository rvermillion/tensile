#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import Any, Object


class StateAware(Object):

    __slots__ = ()

    def state_dict(self) -> dict[str, Any]:
        state = {}
        self._add_state(state)
        return state

    def _add_state(self, state: dict[str, Any]):
        pass

