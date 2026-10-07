"""Transport-neutral access to the store's component meshes.

The meshes are the ``mesh`` index defined by :mod:`cadgen.store.meshes`. This
module owns the bounded probe/body protocol and TESB framing used by both the
snapshot host and the Viewer, and the snapshot host's produce request. Clients
only read: cadgen writes every mesh. URI decoding and HTTP status codes stay in
the hosts, so store and snapshot code never import ``cadgen.viewer``.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable

__all__ = [
    "TESS_CACHE_BATCH_MAGIC", "TESS_CACHE_BATCH_MAX_BYTES",
    "TESS_CACHE_BATCH_MAX_NAMES", "TESS_CACHE_BATCH_VERSION",
    "TESS_CACHE_METADATA_MAX_BYTES", "encode_tessellation_cache_batch",
    "is_tessellation_cache_key", "parse_tess_cache_admission",
    "produce_tess_cache", "read_tess_cache_batch", "read_tess_cache_probe",
    "read_tessellation_cache", "tessellation_cache_dir",
]

TESS_CACHE_BATCH_MAGIC = 0x42534554  # "TESB" little-endian
TESS_CACHE_BATCH_VERSION = 1
TESS_CACHE_BATCH_MAX_NAMES = 256
TESS_CACHE_BATCH_MAX_BYTES = 32 * 1024 * 1024
TESS_CACHE_METADATA_MAX_BYTES = 256 * 1024


def tessellation_cache_dir() -> str:
    from cadgen.store.paths import index_dir
    return str(index_dir("mesh"))


def is_tessellation_cache_key(value: object) -> bool:
    from cadgen.store.meshes import valid_key
    return valid_key(value)


def _validated_key(key: str) -> str:
    if not is_tessellation_cache_key(key):
        raise ValueError(f"invalid tessellation cache key: {key!r}")
    return key


def parse_tess_cache_admission(digest, raw_limit) -> tuple[str, int]:
    """Validate an exact object digest and the caller's admitted byte bound."""
    from cadgen.store.meshes import MAX_SAFE_INTEGER
    if (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(raw_limit) is not str or re.fullmatch(r"[0-9]{1,16}", raw_limit) is None):
        raise ValueError("cache read requires an exact object and maxBytes")
    limit = int(raw_limit)
    if not 0 < limit <= MAX_SAFE_INTEGER:
        raise ValueError("maxBytes must be a positive safe integer")
    return digest, limit


def read_tessellation_cache(
    key: str, *, expected_object: str | None = None, max_bytes: int | None = None,
) -> bytes | None:
    """Read one verified body, optionally bound to a probe's exact facts."""
    key = _validated_key(key)
    try:
        from cadgen.store.meshes import read
        return read(key, expected_object=expected_object, max_bytes=max_bytes)
    except (OSError, ValueError):
        return None


def encode_tessellation_cache_batch(entries: Iterable[bytes | None]) -> bytes:
    """Frame opaque entries as the browser codec's aligned TESB v1 container."""
    materialized = list(entries)
    out = bytearray(struct.pack(
        "<III", TESS_CACHE_BATCH_MAGIC, TESS_CACHE_BATCH_VERSION, len(materialized),
    ))
    for entry in materialized:
        length = len(entry) if entry else 0
        out += struct.pack("<I", length)
        if length:
            out += entry
            out += b"\0" * (-length % 4)
    return bytes(out)


def _request_items(body: bytes | None, field: str) -> list | None:
    if len(body or b"") > TESS_CACHE_METADATA_MAX_BYTES:
        return None
    try:
        parsed = json.loads(bytes(body or b"").decode("utf-8", errors="replace"))
    except (ValueError, RecursionError):
        return None
    items = parsed.get(field) if type(parsed) is dict and set(parsed) == {field} else None
    return items if type(items) is list and len(items) <= TESS_CACHE_BATCH_MAX_NAMES else None


def read_tess_cache_probe(body: bytes | None) -> dict | None:
    """Return bounded index facts without loading a TESS or SURF body."""
    from cadgen.store.meshes import probe
    inputs = _request_items(body, "tessellationInputs")
    if inputs is None:
        return None
    entries = {}
    for key in inputs:
        row = probe(key) if type(key) is str else None
        if row is not None:
            entries[key] = row
    return {"entries": entries}


def produce_tess_cache(body: bytes | None) -> dict | None:
    """Mesh what a probe found missing, and answer as the probe would.

    In-process kernel work: only a host that may mesh on its own thread -- the
    snapshot host -- serves this. A key whose surface the store does not hold
    stays missing.
    """
    from cadgen.store.surfaces import produce_meshes
    inputs = _request_items(body, "tessellationInputs")
    if inputs is None or any(type(key) is not str for key in inputs):
        return None
    return {"entries": {key: row for key, row in produce_meshes(inputs).items() if row is not None}}


def read_tess_cache_batch(body: bytes | None) -> bytes | None:
    """Read admitted exact objects into a response capped at 32 MiB."""
    requests = _request_items(body, "entries")
    if requests is None:
        return None
    entries: list[bytes | None] = []
    remaining = TESS_CACHE_BATCH_MAX_BYTES - 12 - 4 * len(requests)
    for request in requests:
        data = None
        if type(request) is dict and set(request) == {
            "tessellationInput", "object", "maxBytes",
        }:
            key, digest, limit = (
                request["tessellationInput"], request["object"], request["maxBytes"],
            )
            if (is_tessellation_cache_key(key) and type(digest) is str
                    and type(limit) is int and limit > 0 and remaining >= 0):
                data = read_tessellation_cache(
                    key, expected_object=digest, max_bytes=min(limit, remaining),
                )
                if data and len(data) + (-len(data) % 4) > remaining:
                    data = None
        entries.append(data)
        if data:
            remaining -= len(data) + (-len(data) % 4)
    return encode_tessellation_cache_batch(entries)
