"""Thin wrappers around the Go CLIs (JSON in, JSON out)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from common import BIN, KEYS


def _run(name: str, payload: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    out = path[:-5] + ".out.json"
    p = subprocess.run([str(BIN / name), "-in", path, "-out", out], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{p.stderr[-3000:]}")
    with open(out) as f:
        res = json.load(f)
    os.remove(path)
    os.remove(out)
    return res


def commit(out_dir: str, clients: list[dict]) -> dict:
    return _run("commit", {"out_dir": out_dir, "clients": clients})


def prove(jobs: list[dict]) -> dict:
    return _run("prove", {"keys": str(KEYS), "jobs": jobs})


def verify(checks: list[dict]) -> dict:
    return _run("verify", {"keys": str(KEYS), "checks": checks})
