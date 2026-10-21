"""Verify a source release against its SHA-256 inventory (standard library only)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        inventory = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
        if inventory.get("schemaVersion") != 1 or not isinstance(inventory.get("files"), dict):
            raise ValueError("Unsupported integrity inventory.")
        files = inventory["files"]
        seen: set[str] = set()
        for name, expected in files.items():
            relative = PurePosixPath(name)
            if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                    or ":" in name or "\\" in name or str(relative) != name
                    or name.casefold() in seen):
                raise ValueError("Unsafe or duplicate inventory path.")
            seen.add(name.casefold())
            path = root.joinpath(*relative.parts)
            if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
                raise ValueError(f"Symbolic link refused: {name}")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Missing or changed release file: {name}")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and ".git" not in p.relative_to(root).parts}
        if actual != set(files) | {"checksums.json"}:
            raise ValueError("Unexpected or missing files in release directory.")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {len(files)} files. Checksums detect corruption; use the official repository/release for provenance.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())