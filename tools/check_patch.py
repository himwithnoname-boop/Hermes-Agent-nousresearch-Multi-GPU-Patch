"""Read-only baseline and patch applicability check for an isolated Hermes checkout."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BASES = {"5b147ab036d6d19131bff8d20701b5001eab844a", "4d04303818b61bf3aa09278af48356315515965b"}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    supported = json.loads((package / "installer/compatibility.json").read_text(encoding="utf-8"))["baselines"]
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    def git(*command: str) -> str:
        result = subprocess.run(["git", "-C", str(args.checkout), *command],
                                capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Git validation failed.")
        return result.stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        if head not in supported:
            raise RuntimeError("Unsupported revision; check installer/compatibility.json for validated releases.")
        patch = package / "patches" / supported[head]["patch"]
        if git("status", "--porcelain", "--untracked-files=normal"):
            raise RuntimeError("Checkout is not clean. Use a separate clean checkout.")
        git("apply", "--check", "--binary", str(patch))
    except (OSError, RuntimeError) as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1
    print("Baseline and patch applicability verified. No files changed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())