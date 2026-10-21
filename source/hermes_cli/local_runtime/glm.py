"""GLM-5.3 compatibility for the pinned mainline engine; original GGUFs stay intact."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import struct
import threading

GLM53_ARCHITECTURES = frozenset({"glm5next", "glm5-next"})
# The per-card model/output mapping was checked against native llama-model.cpp at
# b11706 (f74e2fa6a31cc33ab4ee4ae516641cfe62df3924). A newer tag is not evidence
# that its layer partition and tensor residency semantics remain equivalent.
_GLM_CUDA_PLACEMENT_ENGINES = frozenset({"b11706"})
_LOCK = threading.RLock()
_MAX_HEADER = 64 << 20
_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def engine_refusal(header, engine_tag: str | None, *, cuda_device_count: int = 0) -> str | None:
    if getattr(header, "architecture", "") not in GLM53_ARCHITECTURES:
        return None
    if engine_tag is None:
        from hermes_cli.local_runtime.bootstrap import serving_engine

        engine = serving_engine()
        engine_tag = engine.tag if engine is not None else None
    if cuda_device_count > 1 and engine_tag not in _GLM_CUDA_PLACEMENT_ENGINES:
        return ("GLM per-card CUDA placement is verified only for llama.cpp b11706. "
                "This engine's layer mapping has not been verified; no model launch preset was "
                "created. Use the verified engine or single-device planning until compatibility "
                "has been checked.")
    try:
        build = int(str(engine_tag).removeprefix("b"))
    except ValueError:
        build = 0
    if build < 11368:
        return "GLM-5.3-Flash requires llama.cpp b11368 or newer (this patch is tested with b11706). Update the local runtime before loading it."
    return None


def compatible_template(source: str) -> str:
    """Equivalent bracket indexing works in both Jinja2 and llama.cpp's minja."""
    for name in ("m.content", "tr.output", "entry.output"):
        source = source.replace(name + ".0.", name + "[0].")
    # llama.cpp's capability probe supplies tools=[null]. Skipping null preserves real tools.
    return source.replace("for tool in tools %", "for tool in tools if tool %").replace(
        "for tool in tools -%", "for tool in tools if tool -%")


def _read(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated GLM metadata shard")
    return value


def _string(stream) -> str:
    size = struct.unpack("<Q", _read(stream, 8))[0]
    if size > _MAX_HEADER:
        raise ValueError("oversized GLM metadata string")
    return _read(stream, size).decode("utf-8")


def _skip_value(stream, kind: int, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("nested GLM metadata exceeds supported depth")
    if kind in _SCALAR_BYTES:
        _read(stream, _SCALAR_BYTES[kind])
    elif kind == 8:
        _string(stream)
    elif kind == 9:
        element, count = struct.unpack("<IQ", _read(stream, 12))
        if count > _MAX_HEADER:
            raise ValueError("oversized GLM metadata array")
        if element in _SCALAR_BYTES:
            _read(stream, count * _SCALAR_BYTES[element])
        else:
            for _ in range(count):
                _skip_value(stream, element, depth + 1)
    else:
        raise ValueError("unsupported GLM metadata type")


def _pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def normalized_metadata(raw: bytes) -> bytes:
    """Rename only legacy architecture metadata in a tensor-free first split shard."""
    if len(raw) > _MAX_HEADER:
        raise ValueError("GLM first shard is not a bounded metadata-only shard")
    stream = io.BytesIO(raw)
    prefix = _read(stream, 24)
    magic, version, tensors, count = struct.unpack("<4sIQQ", prefix)
    if magic != b"GGUF" or version not in (2, 3) or tensors or count > len(raw) // 12:
        raise ValueError("GLM legacy repair requires a GGUF v2/v3 shard with zero tensors")
    output = [prefix]
    for _ in range(count):
        key = _string(stream)
        kind_raw = _read(stream, 4)
        kind = struct.unpack("<I", kind_raw)[0]
        start = stream.tell()
        _skip_value(stream, kind)
        value = raw[start:stream.tell()]
        if key.startswith("glm5next."):
            key = "glm5-next." + key[len("glm5next."):]
        if key == "general.architecture":
            if kind != 8 or value != _pack_string("glm5next"):
                raise ValueError("GLM legacy architecture does not match")
            value = _pack_string("glm5-next")
        output.extend((_pack_string(key), kind_raw, value))
    # There are no tensors or tensor offsets in this shard. Padding to 32 bytes is sufficient.
    result = b"".join(output)
    return result + b"\0" * (-len(result) % 32)


def _legacy_model(path: Path) -> Path:
    from hermes_cli.local_runtime.gguf import SPLIT_PART_RE, _read_part, split_parts
    from utils import atomic_write_bytes

    first = _read_part(path)
    match = SPLIT_PART_RE.search(path.name)
    if (not match or int(match[1]) != 1 or first.n_tensors
            or first.metadata.get("split.no") != 0 or path.stat().st_size > _MAX_HEADER):
        raise ValueError("Legacy glm5next requires a tensor-free first split shard; use the publisher's Shard_Rewrite files for this model")
    parts = split_parts(path)
    if parts is None or len(parts) != int(match[2]):
        raise ValueError("GLM split download is incomplete")
    sources = [part.resolve(strict=True) for part in parts]
    # Locate beside the weights so hardlinks never copy hundreds of GB across volumes.
    signature = repr([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources])
    raw = path.read_bytes()
    key = hashlib.sha256(raw + signature.encode("utf-8")).hexdigest()[:24]
    directory = sources[0].parent / ".hermes-glm-compat" / key
    directory.mkdir(parents=True, exist_ok=True)
    for original, source in zip(parts[1:], sources[1:]):
        target = directory / original.name
        try:
            os.link(source, target)
        except FileExistsError:
            if not os.path.samefile(source, target):
                raise ValueError("GLM compatibility shard differs from the original") from None
    target = directory / path.name
    normalized = normalized_metadata(raw)
    if not target.exists() or target.read_bytes() != normalized:
        atomic_write_bytes(target, normalized)
    return target


def preset_keys(path: Path, header) -> dict[str, str]:
    """Prepare tiny sidecars after admission; never edit or copy original weight tensors."""
    if getattr(header, "architecture", "") not in GLM53_ARCHITECTURES:
        return {}
    from hermes_cli.local_runtime.binaries import runtimes_root
    from utils import atomic_write_text

    source = header.metadata.get("tokenizer.chat_template")
    if not isinstance(source, str) or not source or len(source) > 256 * 1024:
        raise ValueError("GLM chat template is missing or exceeds the supported size")
    template = compatible_template(source)
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    target = runtimes_root() / "templates" / (digest + ".jinja")
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text(encoding="utf-8-sig") != template:
            atomic_write_text(target, template)
        model = _legacy_model(path) if header.architecture == "glm5next" else path
    return {"model": str(model), "chat-template-file": str(target),
            "chat-template-kwargs": '{"reasoning_effort":"max"}',
            "flash-attn": "off", "cache-type-k": "f16", "cache-type-v": "f16"}


def remove_model_files(paths: list[Path]) -> None:
    """Release compatibility hardlinks with model deletion, so they cannot retain tensor data."""
    from hermes_cli.local_runtime.gguf import SPLIT_PART_RE, _read_part

    names = {path.name for path in paths}
    with _LOCK:
        directories = []
        for path in paths:
            match = SPLIT_PART_RE.search(path.name)
            if match is None or int(match[1]) != 1:
                continue
            cache = path.resolve().parent / ".hermes-glm-compat"
            if not cache.is_dir():
                continue
            for candidate in cache.glob("*/" + path.name):
                directory = candidate.parent
                if cache.is_symlink() or directory.is_symlink():
                    raise ValueError("Redirected GLM compatibility cache refused")
                children = list(directory.iterdir())
                if any(child.name not in names or not child.is_file() or child.is_symlink()
                       for child in children):
                    raise ValueError("Unexpected GLM compatibility cache contents; original model kept")
                header = _read_part(candidate)
                if header.architecture != "glm5-next" or header.n_tensors:
                    raise ValueError("Unrecognized GLM compatibility metadata; original model kept")
                directories.append((directory, children))
        for directory, children in directories:
            for child in children:
                child.unlink()
            directory.rmdir()
        for path in paths:
            path.unlink(missing_ok=True)
