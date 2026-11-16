"""FUsion backdoor (Malicious Forgetting, INFOCOM 2026, Sec. III) + baseline & sign-flip.

Pipeline:
  prepare()  -- run once on the converged global model (end of FL training):
     1. train a local backdoor model g_bd per malicious client (clean + poisoned data);
     2. identify a CONTIGUOUS backdoor-critical (BC) layer set by forward layer substitution
        + the contiguous expansion heuristic (Eqs. 4-8);
     3. select top-rho trigger-sensitive (TS) units in each BC layer via Grad-CAM;
     4. construct the backdoor subnetwork: train only the TS units on backdoor data with a clean
        activation-preserving regulariser (Eq. 10); store S_bd (nonzero only at TS units in BC layers).

  inject() -- each unlearning round, for a malicious client:
     honest unlearning  ->  fuse  w̃ = ŵ_i + beta * S_bd  (Eq. 11)
     ->  fine-tune on clean+backdoor (Eq. 12)  ->  submit g̃ = w̃* - ŵ_t.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models
from .data import add_trigger


@dataclass
class Prepared:
    S_bd: torch.Tensor          # full-dim fusion vector (Eq. 11), nonzero only at TS units in BC layers
    bc_layers: list             # contiguous BC-layer indices H_BC (0-based weight-layer indices)
    ts_units: dict              # {layer_idx: LongTensor of selected output-unit indices}
    delta_bar: list            # normalised per-layer contribution δ̄(h)
    gbd_ba: float = 0.0         # backdoor accuracy of the prepared local backdoor model g_bd


# ----------------------------------------------------------------- backdoor model g_bd
def _train_gbd(template, g_vec, Xc, yc, Xb, yb, cfg):
    """Local backdoor model: composite loss Eq.(2), EDic[L] + gamma*EDip[L]."""
    net = models.load(template, g_vec)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    gen = torch.Generator().manual_seed(cfg.seed + 11)
    net.train()
    for _ in range(cfg.prep_steps):
        bi = torch.randint(0, len(yc), (cfg.batch_size,), generator=gen)
        pi = torch.randint(0, len(yb), (cfg.batch_size,), generator=gen)
        opt.zero_grad()
        loss = F.cross_entropy(net(Xc[bi]), yc[bi]) + cfg.gamma * F.cross_entropy(net(Xb[pi]), yb[pi])
        loss.backward()
        opt.step()
    return net


# ----------------------------------------------------------------- BC-layer identification (Eqs. 4-8)
def _ba(net, Xb, yb):
    with torch.no_grad():
        return (net(Xb).argmax(1) == yb).float().mean().item()


def _bc_layers(template, g_vec, gbd, Xb, yb, cfg):
    """Forward layer substitution: BA drop when layer h of g_bd is replaced by benign w. Then
    normalise (Eq. 4), average (Eq. 5), anchor (Eq. 6), contiguous expansion (Eqs. 7-8)."""
    benign = models.load(template, g_vec)
    bl, gl = models.weight_layers(benign), models.weight_layers(gbd)
    H = len(gl)
    base_ba = _ba(gbd, Xb, yb)
    drops = []
    for h in range(H):
        probe = copy.deepcopy(gbd)
        pl = models.weight_layers(probe)
        with torch.no_grad():                      # replace layer h with its benign counterpart
            pl[h][1].weight.copy_(bl[h][1].weight)
            if pl[h][1].bias is not None:
                pl[h][1].bias.copy_(bl[h][1].bias)
        drops.append(max(0.0, base_ba - _ba(probe, Xb, yb)))
    drops = torch.tensor(drops)
    delta = drops / (drops.max() + 1e-12)           # δ_i(h), Eq.(4)
    return delta


def _contiguous(delta_bar, theta):
    """Anchor h* = argmax δ̄; expand while δ̄(h) >= theta*δ̄(h*) (Eqs. 6-8)."""
    h_star = int(torch.argmax(delta_bar))
    thr = theta * delta_bar[h_star].item()
    lo = hi = h_star
    while lo - 1 >= 0 and delta_bar[lo - 1].item() >= thr:
        lo -= 1
    while hi + 1 < len(delta_bar) and delta_bar[hi + 1].item() >= thr:
        hi += 1
    return list(range(lo, hi + 1))


# ----------------------------------------------------------------- TS-unit selection (Grad-CAM)
def _gradcam_units(gbd, bc_layers, Xb, yb, rho):
    """Top-rho output units (conv channels / linear neurons) per BC layer by Grad-CAM score."""
    wl = models.weight_layers(gbd)
    acts, grads, handles = {}, {}, []
    for li in bc_layers:
        mod = wl[li][1]

        def fwd(m, i, o, li=li):
            o.retain_grad()
            acts[li] = o

        handles.append(mod.register_forward_hook(fwd))
    gbd.zero_grad()
    out = gbd(Xb)
    score = out.gather(1, yb.view(-1, 1)).sum()     # target-logit response to triggers
    score.backward()
    for li in bc_layers:
        grads[li] = acts[li].grad
    for hdl in handles:
        hdl.remove()

    ts = {}
    for li in bc_layers:
        a, g = acts[li].detach(), grads[li].detach()
        if a.dim() == 4:                            # conv: Grad-CAM weight = mean grad over spatial
            w = g.mean(dim=(2, 3))                   # (B,C)
            cam = F.relu((w.unsqueeze(-1).unsqueeze(-1) * a).sum(1, keepdim=False))  # unused shape guard
            imp = F.relu((w[..., None, None] * a).sum(dim=(2, 3))).mean(0)  # (C,)
        else:                                        # linear: per-neuron relu(grad*act)
            imp = F.relu(g * a).mean(0)              # (units,)
        n_units = imp.numel()
        k = max(1, int(round(rho * n_units)))
        ts[li] = torch.topk(imp, k).indices.sort().values
    return ts


# ----------------------------------------------------------------- subnetwork construction (Eq. 10)
def _ts_param_mask(template, bc_layers, ts_units):
    """Boolean mask over the flat parameter vector: True only on weight rows of TS output units."""
    masks = []
    wl = models.weight_layers(template)
    mod2li = {id(m): li for li, (_, m) in enumerate(wl)}
    for p_mod, p in _named_params_with_mod(template):
        m = torch.zeros_like(p, dtype=torch.bool)
        li = mod2li.get(id(p_mod))
        if li in bc_layers and p.dim() >= 2:         # weight matrix/tensor; select TS output rows
            m[ts_units[li]] = True
        masks.append(m.flatten())
    return torch.cat(masks)


def _named_params_with_mod(model):
    """Yield (owning_module, param) in parameters() order, weights then biases per layer."""
    out = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            out.append((mod, mod.weight))
            if mod.bias is not None:
                out.append((mod, mod.bias))
    return out


def _train_subnetwork(template, g_vec, bc_layers, ts_units, Xb, yb, Xc, yc, cfg):
    """Train only TS units on backdoor data; regularise BC-layer activations on clean inputs (Eq.10)."""
    net = models.load(template, g_vec)
    ref = models.load(template, g_vec)              # frozen benign reference for the clean reg term
    ref.eval()
    ts_mask = _ts_param_mask(template, bc_layers, ts_units).view(-1)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    gen = torch.Generator().manual_seed(cfg.seed + 23)
    hmax = max(bc_layers)
    upto = min(hmax + 1, 3)
    net.train()
    for _ in range(cfg.sub_rounds * cfg.sub_steps):
        bi = torch.randint(0, len(yb), (cfg.batch_size,), generator=gen)
        ci = torch.randint(0, len(yc), (cfg.batch_size,), generator=gen)
        opt.zero_grad()
        l_bd = F.cross_entropy(net(Xb[bi]), yb[bi])
        with torch.no_grad():
            ref_feat = ref.features(Xc[ci], upto)
        l_clean = F.mse_loss(net.features(Xc[ci], upto), ref_feat)
        (cfg.lambda1 * l_bd + cfg.lambda2 * l_clean).backward()
        # keep only TS-unit gradients -> other weights stay at benign values
        with torch.no_grad():
            flat = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                              for p in net.parameters()])
            flat = flat * ts_mask
            i = 0
            for p in net.parameters():
                n = p.numel()
                if p.grad is not None:
                    p.grad.copy_(flat[i:i + n].view_as(p))
                i += n
        opt.step()
    S = (models.get_vec(net) - g_vec) * ts_mask     # nonzero only at TS units in BC layers (Eq.11)
    return S


# ----------------------------------------------------------------- public API
def prepare(template, g_vec, mal_clean, mal_bd, cfg):
    """Collaborative preparation. mal_clean/mal_bd: pooled (X,y) clean / backdoor data of attackers."""
    Xc, yc = mal_clean
    Xb, yb = mal_bd
    gbd = _train_gbd(template, g_vec, Xc, yc, Xb, yb, cfg)
    gbd_ba = _ba(gbd, Xb, yb)                                    # how well the local backdoor model learned
    delta_bar = _bc_layers(template, g_vec, gbd, Xb, yb, cfg)    # averaged already via pooled attacker data
    bc = _contiguous(delta_bar, cfg.theta)
    ts = _gradcam_units(gbd, bc, Xb, yb, cfg.rho)
    S = _train_subnetwork(template, g_vec, bc, ts, Xb, yb, Xc, yc, cfg)
    return Prepared(S_bd=S, bc_layers=bc, ts_units={k: v for k, v in ts.items()},
                    delta_bar=delta_bar.tolist(), gbd_ba=gbd_ba)


def inject_fusion(client, base_vec, rnd, prep: Prepared, cfg):
    """Malicious FU update: honest unlearn -> fuse S_bd -> fine-tune clean+backdoor -> delta."""
    unlearned = base_vec + client.honest_fu_round(base_vec, rnd)   # legit clean-data unlearning first
    fused = unlearned + cfg.beta * prep.S_bd                        # Eq.(11)
    net = models.load(client.template, fused)
    Xc, yc = client._retain()
    Xb, yb = client.attack_bd                                       # this client's backdoor data
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    gen = torch.Generator().manual_seed(client._seed(rnd, 7))
    net.train()
    for _ in range(cfg.ft_steps):                                   # Eq.(12)
        ci = torch.randint(0, len(yc), (cfg.batch_size,), generator=gen)
        bi = torch.randint(0, len(yb), (cfg.batch_size,), generator=gen)
        opt.zero_grad()
        loss = F.cross_entropy(net(Xc[ci]), yc[ci]) + cfg.lam * F.cross_entropy(net(Xb[bi]), yb[bi])
        loss.backward()
        opt.step()
    return models.get_vec(net) - base_vec


def inject_baseline(client, base_vec, rnd, cfg):
    """Baseline: honest unlearn, then directly fine-tune on backdoor+clean (no subnetwork)."""
    unlearned = base_vec + client.honest_fu_round(base_vec, rnd)
    net = models.load(client.template, unlearned)
    Xc, yc = client._retain()
    Xb, yb = client.attack_bd
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    gen = torch.Generator().manual_seed(client._seed(rnd, 8))
    net.train()
    for _ in range(cfg.ft_steps):
        ci = torch.randint(0, len(yc), (cfg.batch_size,), generator=gen)
        bi = torch.randint(0, len(yb), (cfg.batch_size,), generator=gen)
        opt.zero_grad()
        loss = F.cross_entropy(net(Xc[ci]), yc[ci]) + cfg.lam * F.cross_entropy(net(Xb[bi]), yb[bi])
        loss.backward()
        opt.step()
    return models.get_vec(net) - base_vec


def inject_signflip(client, base_vec, rnd, cfg):
    """Sign-flip: submit the negated honest unlearning update (untargeted; SubID robustness test)."""
    return -1.0 * client.honest_fu_round(base_vec, rnd)
