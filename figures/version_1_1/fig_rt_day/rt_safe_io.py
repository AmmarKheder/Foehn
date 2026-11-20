"""Safe writes for the files of this figure folder (the file system dropped writes on 2026-11-19): every file is written
under a temporary name in the same directory, flushed, fsynced, read back and compared, then renamed with os.replace.
After replay/cams3h/code/safe_io.py; named rt_safe_io so that it never shadows that module."""
import hashlib
import io
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
    assert len(data) > 0, f"nothing to write to {path}"
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
    """text -> path as UTF-8, no newline translation (a csv written to io.StringIO(newline="") keeps its line ends)"""
    return write_bytes(path, text.encode("utf-8"))


def write_npz(path, compressed=True, **arrays):
    """np.savez[_compressed] through a memory buffer, then write_bytes; the file is reopened and every array compared"""
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    h = write_bytes(path, buf.getvalue())
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(arrays), (sorted(z.files), sorted(arrays))
        for k, v in arrays.items():
            a, b = np.asarray(v), z[k]
            assert a.shape == b.shape and a.dtype == b.dtype, (k, a.shape, b.shape, a.dtype, b.dtype)
            assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b), k
    return h


def savefig(fig, path, **kw):
    """fig.savefig through a memory buffer (format from the extension of `path`), then write_bytes"""
    path = Path(path)
    buf = io.BytesIO()
    fig.savefig(buf, format=path.suffix[1:], **kw)
    return write_bytes(path, buf.getvalue())
