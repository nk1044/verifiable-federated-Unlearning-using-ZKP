"""Integration points for *Verifiable FU with ZKPs*.

No concrete ZK proof system is chosen here (you haven't decided yet). This module fixes the
INTERFACES so a Groth16 / Halo2 / EZKL / zkLLVM backend can be dropped in later without touching
the FL/FU/attack code. The three ideas from your ZKP write-up map to:

  * DatasetCommitment  -> Com(A): Merkle root over the client's salted samples, published once
                          before training. "Binds the data" (Step 2).
  * forget mask b      -> one bit per sample of A (1 = use, 0 = forget). Handled in client.py;
                          mask_digest() binds it into a proof (Step 3, bitmask from zkUnlearner).
  * Verifier           -> the server-side check of each ClientMessage (Step 1, "prove the procedure").

Verifier implementations:
  * NullVerifier   accepts every update. This is exactly the trust assumption the FUsion paper
                   attacks, so run the attack under this to reproduce the paper.
  * ReplayVerifier a CLEARTEXT ORACLE: it re-runs the fixed honest procedure on the committed data
                   + declared mask and accepts only a bit-identical update (torch.allclose). This is
                   the *statement* a ZK proof certifies, minus the privacy. A FUsion-patched update
                   fails it, demonstrating why the ZKP defense works -- before any real prover exists.

To plug in a real ZK system later, implement `verify(msg)` by checking msg.proof against the public
inputs (global-model digest, update digest, Com(A), mask_digest). The honest procedure that the
circuit must encode is client.honest_round() -- keep it deterministic and fixed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------- digests / commitments
def tensor_digest(t: torch.Tensor) -> str:
    """Stable hash of a float tensor (binds a proof to an exact model / update)."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


class DatasetCommitment:
    """Com(A) = Merkle root over salted samples. Hides the data, locks it against later change."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, salt: bytes):
        self.salt = salt
        leaves = []
        for i in range(len(y)):
            leaves.append(_h(salt, X[i].cpu().numpy().tobytes(), int(y[i]).to_bytes(4, "big", signed=True)))
        self.leaves = leaves
        self.root = self._merkle(leaves) if leaves else _h(salt)

    @staticmethod
    def _merkle(leaves):
        level = list(leaves)
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [_h(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        return level[0]

    @property
    def root_hex(self) -> str:
        return self.root.hex()


def mask_digest(mask: torch.Tensor) -> str:
    """Digest of the forget bitmask b (binds which samples were used in a given round's proof)."""
    return hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- message + verifiers
@dataclass
class ClientMessage:
    """What a client sends the server each round. `proof` is a placeholder for a real ZK proof."""
    client_id: int
    phase: str                       # "fl" | "fu"
    base_digest: str                 # digest of the global model the round started from
    update: torch.Tensor             # the weight update g_i  (new_weights - base_weights)
    commit_root: Optional[str] = None
    mask_hash: Optional[str] = None
    proof: Optional[object] = None   # None here; a ZKVerifier would require this to be present/valid
    meta: dict = field(default_factory=dict)


class NullVerifier:
    """Accepts everything. Reproduces the unverified setting the FUsion backdoor attacks."""

    name = "null"

    def verify(self, msg: ClientMessage) -> bool:
        return True


class ReplayVerifier:
    """Cleartext oracle. Recomputes the honest procedure and accepts only a matching update.

    `honest_fn(client_id, base_vec)` must return the honest update tensor for that client/round.
    The runner wires this to Client.honest_round so malicious (patched) updates are rejected.
    """

    name = "replay"

    def __init__(self, honest_fn, atol: float = 1e-5, rtol: float = 1e-4):
        self.honest_fn = honest_fn
        self.atol, self.rtol = atol, rtol
        self.rejected: list[int] = []

    def verify(self, msg: ClientMessage) -> bool:
        expected = self.honest_fn(msg.client_id, msg)
        ok = expected is not None and torch.allclose(msg.update, expected, atol=self.atol, rtol=self.rtol)
        if not ok:
            self.rejected.append(msg.client_id)
        return ok


def make_verifier(cfg, honest_fn):
    if cfg.verifier == "none":
        return NullVerifier()
    if cfg.verifier == "replay":
        return ReplayVerifier(honest_fn)
    raise ValueError(cfg.verifier)
