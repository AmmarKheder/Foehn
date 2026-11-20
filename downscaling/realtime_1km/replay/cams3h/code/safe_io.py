#!/usr/bin/env python3
"""Safe writes for the cams3h work (the file system dropped a write on 2026-11-19): every file is written to a temporary
name in the same directory, flushed, fsynced, read back and compared, then renamed with os.replace. Numpy only."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


def _fsync_dir(d):
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_bytes(path, data):
    """bytes -> path (temporary file, flush, fsync, read back, os.replace); returns the SHA-256 of the content"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        back = f.read()
    if back != data:
        tmp.unlink(missing_ok=True)
        raise OSError(f"read-back differs from what was written: {path} ({len(back)} of {len(data)} bytes)")
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return hashlib.sha256(data).hexdigest()


def write_text(path, text):
    return write_bytes(path, text.encode("utf-8"))


def write_json(path, obj, **kw):
    kw.setdefault("indent", 1)
    kw.setdefault("default", float)
    return write_text(path, json.dumps(obj, **kw))


def write_npz(path, compressed=False, **arrays):
    """np.savez through a memory buffer, then write_bytes; the file is reopened and every array compared"""
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    h = write_bytes(path, buf.getvalue())
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(arrays), (sorted(z.files), sorted(arrays))
        for k, v in arrays.items():
            a, b = np.asarray(v), z[k]
            assert a.shape == b.shape and a.dtype == b.dtype, (k, a.shape, b.shape, a.dtype, b.dtype)
            if a.dtype.kind == "f":
                assert np.array_equal(a, b, equal_nan=True), k
            else:
                assert np.array_equal(a, b), k
    return h


def finalize_file(tmp, path):
    """a file already written under a temporary name (a download): fsync, check that it can be read, rename"""
    tmp, path = Path(tmp), Path(path)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
        h = hashlib.sha256()
        n = 0
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
            n += len(chunk)
    assert n == tmp.stat().st_size and n > 0, (n, tmp.stat().st_size)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return h.hexdigest(), n
