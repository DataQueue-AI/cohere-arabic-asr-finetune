"""Bounded local disk cache for audio streamed from GCS during training.

Downloads gs://bucket/key -> <cache_dir>/bucket/key on first access and returns
the local path, so soundfile can read it exactly like a local file. A background
janitor thread evicts the least-recently-accessed files once the cache exceeds
`max_bytes`, so a node never needs enough disk for the whole dataset -- only
enough for the working set that's actually being touched this epoch.

Safe to share across DDP ranks / DataLoader workers on the same node: downloads
are per-key locked and written via a temp-file + atomic rename.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_client = None
_client_lock = threading.Lock()
_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def is_gcs_uri(path) -> bool:
    return isinstance(path, str) and path.startswith("gs://")


def _get_client(key_file: str | None):
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google.cloud import storage
                _client = (storage.Client.from_service_account_json(key_file)
                           if key_file and os.path.isfile(key_file) else storage.Client())
    return _client


def _lock_for(key: str) -> threading.Lock:
    with _download_locks_guard:
        lock = _download_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _download_locks[key] = lock
        return lock


class GCSCache:
    def __init__(self, cache_dir: str, max_bytes: int, key_file: str | None = None,
                 trim_interval_s: float = 60.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.key_file = key_file
        self._stop = threading.Event()
        self._janitor = threading.Thread(target=self._janitor_loop, args=(trim_interval_s,), daemon=True)
        self._janitor.start()

    def local_path(self, gs_uri: str) -> str:
        bucket, _, key = gs_uri[len("gs://"):].partition("/")
        local = self.cache_dir / bucket / key
        if local.is_file() and local.stat().st_size > 0:
            os.utime(local, None)  # bump atime so the janitor treats it as fresh
            return str(local)
        with _lock_for(str(local)):
            if local.is_file() and local.stat().st_size > 0:
                os.utime(local, None)
                return str(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_name(local.name + f".tmp{os.getpid()}")
            blob = _get_client(self.key_file).bucket(bucket).blob(key)
            blob.download_to_filename(str(tmp))
            os.replace(tmp, local)
        return str(local)

    def _janitor_loop(self, interval_s: float):
        while not self._stop.wait(interval_s):
            try:
                self._trim()
            except Exception:
                pass

    def _trim(self):
        entries = []
        total = 0
        for root, _, files in os.walk(self.cache_dir):
            for name in files:
                if ".tmp" in name:
                    continue
                p = os.path.join(root, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                total += st.st_size
                entries.append((st.st_atime, st.st_size, p))
        if total <= self.max_bytes:
            return
        entries.sort()  # oldest-accessed first (LRU eviction)
        for _atime, size, p in entries:
            if total <= self.max_bytes:
                break
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass


_cache_singleton: GCSCache | None = None
_cache_singleton_lock = threading.Lock()


def get_cache(cache_dir: str, max_gb: float, key_file: str | None = None) -> GCSCache:
    global _cache_singleton
    if _cache_singleton is None:
        with _cache_singleton_lock:
            if _cache_singleton is None:
                _cache_singleton = GCSCache(cache_dir, int(max_gb * (1024 ** 3)), key_file)
    return _cache_singleton
