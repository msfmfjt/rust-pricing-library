"""Bounded, process-local memoization; never a source of saved risk evidence."""

from collections import OrderedDict
from collections.abc import MutableMapping
from threading import RLock

from .domain import require


class BoundedCache(MutableMapping):
    """An entry-bounded LRU mapping with atomic lookup and replacement."""

    def __init__(self, capacity):
        self._capacity = capacity
        self._entries = OrderedDict()
        self._lock = RLock()

    def __getitem__(self, key):
        with self._lock:
            value = self._entries[key]
            self._entries.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def __delitem__(self, key):
        with self._lock:
            del self._entries[key]

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self):
        with self._lock:
            return len(self._entries)


class SnapshotCache:
    """Results and audits expire together; in-flight calls keep their own state."""

    def __init__(self, snapshot_hash, capacity):
        self.snapshot_hash = snapshot_hash
        self.results = BoundedCache(capacity)
        self.native = BoundedCache(capacity)
        self.shared = None
        self._lock = RLock()

    def check_surface(self, shared):
        with self._lock:
            require(self.shared is None or self.shared == shared, "surface_mismatch",
                    "instruments do not share the same market/reporting surface")

    def store(self, key, shared, risk, audit=None):
        with self._lock:
            # Another contract can finish while this one releases the GIL in
            # native pricing. Check again before publishing either result.
            self.check_surface(shared)
            self.shared = shared
            self.results[key] = (risk, audit)
