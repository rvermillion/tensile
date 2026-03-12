#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import json

class Storable:

    __slots__ = ()

    def store(self, arrays: dict, metadata: dict, prefix: str = ''):
        self._store_arrays(arrays, prefix=prefix)
        self._store_metadata(metadata, prefix=prefix)

    def _store_arrays(self, arrays: dict, prefix: str = ''):
        pass

    def _store_metadata(self, metadata: dict, prefix: str = ''):
        md = self._metadata_to_store()
        if md:
            metadata[prefix + 'metadata'] = json.dumps(md)

    # noinspection PyMethodMayBeStatic
    def _metadata_to_store(self) -> dict|None:
        return None


class Loadable:

    __slots__ = ()

    def load(self, arrays: dict, metadata: dict, prefix: str = ''):
        raise NotImplementedError()


__all__ = [
    'Loadable',
    'Storable',
]
