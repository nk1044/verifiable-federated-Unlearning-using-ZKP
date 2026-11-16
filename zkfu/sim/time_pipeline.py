"""Time the pipeline steps that the simulations do not record individually.

    ../../.venv/bin/python time_pipeline.py   ->  ../results/pipeline_timing.json
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

import common
import gocli
from common import ROOT, WORK, accuracy, fedavg
from extractor import Extractor, add_trigger, load_mnist


def main():
    res = {}
    t = time.time()
    Xtr, ytr, Xte, yte = load_mnist()
    res["load_mnist_s"] = time.time() - t
    with tempfile.TemporaryDirectory() as tmp:          # fresh pre-training (not the cached extractor)
        common_work = common.WORK
        import extractor as ex
        ex.WORK = Path(tmp)
        t = time.time()
        e = Extractor("cnn+grid", Xtr[50000:], ytr[50000:])
        res["extractor_pretrain_s"] = time.time() - t
        ex.WORK = common_work
    t = time.time()
    F = e.features(Xtr[:50000])
    res["feature_extraction_50k_s"] = time.time() - t
    t = time.time()
    e.features(add_trigger(Xte))
    res["feature_extraction_10k_triggered_s"] = time.time() - t
    # commitment of one client (Go: salts, 2048 leaves, Merkle tree, write private file)
    with tempfile.TemporaryDirectory() as tmp:
        t = time.time()
        out = gocli.commit(tmp, [{"client_id": 0, "features": F[:1000].tolist(), "labels": ytr[:1000].tolist()}])
        res["commit_one_client_total_s"] = time.time() - t
        res["commit_one_client_go_ms"] = out["clients"][0]["ms"]
    com = json.loads(next(WORK.glob("commit_cnn_grid_*.json")).read_text())
    res["commit_50_clients_total_s"] = com.get("seconds")
    # aggregation + evaluation of a 50-client round
    rng = np.random.default_rng(0)
    heads = [(rng.integers(-2**20, 2**20, (10, 64)).tolist(), rng.integers(-2**20, 2**20, 10).tolist()) for _ in range(50)]
    t = time.time()
    W, b = fedavg(heads)
    res["fedavg_50_heads_ms"] = (time.time() - t) * 1000
    tf = e.features(Xte)
    t = time.time()
    accuracy(W, b, tf, yte.numpy())
    res["evaluate_10k_ms"] = (time.time() - t) * 1000
    (ROOT / "results" / "pipeline_timing.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
