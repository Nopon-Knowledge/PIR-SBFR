"""Shared, conservative filesystem helpers for dataset conversion.

The preparation commands deliberately never replace an existing path.  Re-running
the same command is allowed when the existing artifact is byte-for-byte identical;
an inconsistent destination is reported as an error instead of being overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


MATERIALIZE_MODES = ("symlink", "copy")


class DestinationConflictError(FileExistsError):
    """Raised when an existing output differs from the requested artifact."""


def resolve_input_path(path: Path, base: Path) -> Path:
    """Resolve ``path`` relative to ``base`` without requiring it to exist."""

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_file(path: Path, description: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {path}")
    return path


def safe_relative_path(value: str, description: str) -> Path:
    """Return an untrusted dataset path after rejecting traversal/absolute paths."""

    normalized = value.replace("\\", "/")
    path = Path(normalized)
    has_drive_prefix = bool(path.parts and path.parts[0].endswith(":"))
    if not value or path.is_absolute() or has_drive_prefix or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe {description}: {value!r}")
    return path


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def files_equal(first: Path, second: Path) -> bool:
    """Compare two regular files without loading large images into memory."""

    try:
        if first.samefile(second):
            return True
    except (FileNotFoundError, OSError):
        pass
    if not first.is_file() or not second.is_file():
        return False
    if first.stat().st_size != second.stat().st_size:
        return False
    return _sha256(first) == _sha256(second)


def safe_write_bytes(destination: Path, content: bytes) -> bool:
    """Create a file, or accept an identical existing file.

    Returns ``True`` when a new file was created and ``False`` on an idempotent
    re-run.  The exclusive create protects unrelated existing data.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(destination)):
        if destination.is_file() and destination.read_bytes() == content:
            return False
        raise DestinationConflictError(f"Refusing to overwrite existing path: {destination}")

    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.is_file() and destination.read_bytes() == content:
            return False
        raise DestinationConflictError(f"Refusing to overwrite existing path: {destination}")
    except BaseException:
        # Only remove the path created by this call; never follow or unlink a race winner.
        if destination.is_file() and not destination.is_symlink():
            destination.unlink()
        raise
    return True


def safe_write_text(destination: Path, content: str) -> bool:
    return safe_write_bytes(destination, content.encode("utf-8"))


def safe_write_json(destination: Path, payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return safe_write_text(destination, text)


def safe_write_yaml(destination: Path, payload: Mapping[str, Any]) -> bool:
    text = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False)
    return safe_write_text(destination, text)


def safe_materialize(source: Path, destination: Path, mode: str) -> bool:
    """Create an image link/copy without ever replacing an existing destination."""

    if mode not in MATERIALIZE_MODES:
        raise ValueError(f"mode must be one of {MATERIALIZE_MODES}, got {mode!r}")
    source = require_file(Path(source).resolve(), "Source file")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if os.path.lexists(str(destination)):
        if files_equal(source, destination):
            return False
        raise DestinationConflictError(f"Refusing to overwrite existing path: {destination}")

    try:
        if mode == "symlink":
            target = os.path.relpath(str(source), str(destination.parent.resolve()))
            destination.symlink_to(target)
        else:
            # Exclusive destination creation means shutil.copy2 cannot silently replace data.
            with source.open("rb") as input_handle, destination.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            shutil.copystat(source, destination, follow_symlinks=True)
    except FileExistsError:
        if files_equal(source, destination):
            return False
        raise DestinationConflictError(f"Refusing to overwrite existing path: {destination}")
    except BaseException:
        if destination.is_file() and not destination.is_symlink():
            destination.unlink()
        raise
    return True


def read_id_file(path: Path) -> list[str]:
    """Read one image id per line, tolerating standard VOC trailing columns."""

    ids: list[str] = []
    seen: set[str] = set()
    split_lines = require_file(path, "Split id file").read_text(encoding="utf-8-sig").splitlines()
    for line_number, line in enumerate(split_lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        image_id = stripped.split()[0]
        safe_relative_path(image_id, f"image id on line {line_number} of {path}")
        if "/" in image_id or "\\" in image_id:
            raise ValueError(f"Image ids must be basenames, got {image_id!r} in {path}:{line_number}")
        if image_id in seen:
            raise ValueError(f"Duplicate image id {image_id!r} in {path}:{line_number}")
        seen.add(image_id)
        ids.append(image_id)
    if not ids:
        raise ValueError(f"Split id file is empty: {path}")
    return ids


def yolo_line(category: int, box: Sequence[float]) -> str:
    """Format ``(cx, cy, width, height)`` normalized YOLO coordinates."""

    values = " ".join(f"{min(1.0, max(0.0, float(value))):.8f}" for value in box)
    return f"{int(category)} {values}"


def labels_text(lines: Iterable[str]) -> str:
    materialized = list(lines)
    return "\n".join(materialized) + ("\n" if materialized else "")
