"""
espfl_easa.py
=============
EASA: Evolution-Aware Structural Annealing (final architecture, 2026-11-06).

    capacity decision    : how many channels per layer (validation feedback, Sec. 3.2)
    composition decision : which channels (S/E/D gradient-trajectory statistics, Sec. 3.1)

They are coupled only through the target widths K^(ℓ).

Contents
--------
TrajectoryStats       S / E / D per channel (paper Eq. 4), participation means (Eq. 5),
                      stage EMA (Eq. 6), quantile normalisation (Eq. 7).
compose_successor     S-filtered candidate pool (Eq. 10-11) + E(1-D) ranking (Eq. 12) + Top-K (Eq. 13).
CapacityController    'evidence' : level gate + trend intensity  (the new controller, default)
                      'fixed' / 'gate_only' / 'trend_only' : ablations of the two signals
                      'legacy'   : the original implementation's piecewise rules, but fed with
                                   VALIDATION accuracy (pre-experiment "原控制器")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from slim_models import ArchSpec, IndexSet


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory statistics (Eq. 4-7)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryStats:
    """Per-client, per-stage channel statistics on the *current* active channels.

    A channel's parameter block is the union of all tensors whose dim-0 is that
    channel (conv filter + bias + BN affine; for a ResNet stream channel this
    includes every conv/shortcut/BN writing into the stream).  Per local step r:

        S_k = | Σ_j g_kj W_kj |           first-order (Taylor) sensitivity
        E_k = || g_k ||_2                  participation in local optimisation
        D_k = |E_k,r − E_k,r−1| / (sqrt(E_k,r² + E_k,r−1²) + ϵ)   gradient volatility, D_k,1 := 0

    Implementation note: the paper writes S without the absolute value; per-step
    signed sums cancel when averaged over steps, so we use the magnitude (the
    standard first-order Taylor criterion).  Stats are read before optimizer.step().
    """

    def __init__(self, spec: ArchSpec, ema_decay: float = 0.9, eps: float = 1e-8):
        self.spec = spec
        self.groups = list(spec.prunable)
        self.delta = ema_decay
        self.eps = eps
        self.reset()

    # stage-level
    def reset(self) -> None:
        self.ema: Dict[str, Optional[Dict[str, torch.Tensor]]] = {g: None for g in self.groups}
        self.n_participations = 0
        self._tensors = None

    def has_stats(self) -> bool:
        return self.n_participations > 0

    # participation-level
    def begin(self, model: nn.Module) -> None:
        named = dict(model.named_parameters())
        self._tensors = {g: [named[p.key] for p in self.spec.out_params[g]] for g in self.groups}
        dev = next(model.parameters()).device
        self._acc = {g: {'S': torch.zeros(self._tensors[g][0].shape[0], device=dev),
                         'E': torch.zeros(self._tensors[g][0].shape[0], device=dev),
                         'D': torch.zeros(self._tensors[g][0].shape[0], device=dev)}
                     for g in self.groups}
        self._prevE = {g: None for g in self.groups}
        self._steps, self._d_steps = 0, 0

    @torch.no_grad()
    def step(self) -> None:
        if self._tensors is None:
            return
        first_d = None
        for g, tensors in self._tensors.items():
            gw = gg = None
            for p in tensors:
                if p.grad is None:
                    continue
                C = p.shape[0]
                a = (p.grad * p.data).reshape(C, -1).sum(1)
                b = p.grad.pow(2).reshape(C, -1).sum(1)
                gw = a if gw is None else gw + a
                gg = b if gg is None else gg + b
            if gw is None:
                continue
            E = gg.sqrt()
            acc = self._acc[g]
            acc['S'] += gw.abs()
            acc['E'] += E
            prev = self._prevE[g]
            if prev is not None:
                acc['D'] += (E - prev).abs() / (torch.sqrt(E * E + prev * prev) + self.eps)
                first_d = True
            self._prevE[g] = E
        self._steps += 1
        if first_d:
            self._d_steps += 1

    def end(self) -> None:
        if self._tensors is None or self._steps == 0:
            self._tensors = None
            return
        for g in self.groups:
            acc = self._acc[g]
            x = {'S': (acc['S'] / self._steps).cpu(),
                 'E': (acc['E'] / self._steps).cpu(),
                 'D': (acc['D'] / max(self._d_steps, 1)).cpu()}
            if self.ema[g] is None:
                self.ema[g] = x
            else:
                self.ema[g] = {k: self.delta * self.ema[g][k] + (1 - self.delta) * x[k] for k in x}
        self.n_participations += 1
        self._tensors, self._acc, self._prevE = None, None, None


def _qnorm(x: torch.Tensor, q_low: float, q_high: float, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    if x.numel() < 2:
        return torch.ones_like(x)
    lo, hi = torch.quantile(x, q_low), torch.quantile(x, q_high)
    return (x.clamp(lo, hi) - lo) / (hi - lo + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Composition decision (Eq. 10-13)
# ─────────────────────────────────────────────────────────────────────────────

def compose_successor(
    stats: TrajectoryStats,
    current_idx: IndexSet,
    targets: Dict[str, int],
    mode: str = 'sed',
    gamma_vol: float = 1.0,
    quantiles: Tuple[float, float] = (0.05, 0.95),
    weight_norms: Optional[Dict[str, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> IndexSet:
    """Return the pending successor index set (nested in `current_idx`, Eq. 14).

    mode='sed'       : paper rule
    mode='magnitude' : channel L2-norm Top-K (ablation: no trajectory evidence)
    mode='random'    : uniform random subset (ablation)
    """
    new_idx = {g: t.clone() for g, t in current_idx.items()}
    for g, K in targets.items():
        cur = current_idx[g]
        C = cur.numel()
        K = int(min(max(K, 1), C))
        if K >= C:
            continue
        ema = stats.ema.get(g) if stats is not None else None
        if mode == 'sed' and ema is not None and ema['S'].numel() == C:
            S = _qnorm(ema['S'], *quantiles)
            E = _qnorm(ema['E'], *quantiles)
            D = _qnorm(ema['D'], *quantiles)
            kth = torch.topk(S, K).values[-1]
            tau = torch.minimum(torch.quantile(S, 0.5), kth)          # ≥ K candidates guaranteed
            cand = S >= tau
            score = E * (1.0 - D).clamp_min(0) ** gamma_vol
            score = torch.where(cand, score, torch.full_like(score, -1.0))
            sel = torch.topk(score, K).indices
        elif mode == 'magnitude' or (mode == 'sed' and weight_norms is not None):
            # (fallback for 'sed' only if a group has no statistics, which should not happen)
            sel = torch.topk(weight_norms[g].float(), K).indices
        elif mode == 'random':
            sel = torch.randperm(C, generator=generator)[:K]
        else:
            raise RuntimeError(f"composition: no evidence for group {g}")
        new_idx[g] = cur[torch.sort(sel).values]
    return new_idx


def channel_weight_norms(spec: ArchSpec, model: nn.Module) -> Dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    out = {}
    for g in spec.prunable:
        acc = None
        for p in spec.out_params[g]:
            t = named[p.key].detach()
            v = t.pow(2).reshape(t.shape[0], -1).sum(1)
            acc = v if acc is None else acc + v
        out[g] = acc.sqrt().cpu()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Capacity decision
# ─────────────────────────────────────────────────────────────────────────────

def layer_coefficients(spec: ArchSpec, kind: str) -> Dict[str, float]:
    """'uniform': c_ℓ = 1.   'original': the layer multipliers of the original MaskGenerator
    (VGG: conv 0,1 and last three ×0.3, conv 2 ×0.5; ResNet stages ×[0.3,0.7,1.0,0.5]).
    Projector hidden groups (pruned only in the new implementation) keep 1.0."""
    c = {g: 1.0 for g in spec.prunable}
    if kind == 'uniform':
        return c
    if kind != 'original':
        raise ValueError(kind)
    if spec.arch == 'vgg16':
        n = sum(1 for g in spec.groups if g.startswith('c'))
        for g in spec.prunable:
            if g.startswith('c'):
                i = int(g[1:])
                c[g] = 0.3 if (i in (0, 1) or i >= n - 3) else (0.5 if i < 3 else 1.0)
    else:
        stage = {1: 0.3, 2: 0.7, 3: 1.0, 4: 0.5}
        for g in spec.prunable:
            c[g] = stage[int(g[1])]
    return c


@dataclass
class ControllerConfig:
    mode: str = 'evidence'           # evidence | fixed | gate_only | trend_only | legacy
    eps: float = 0.05                # level-gate tolerance ε (relative). With ~45 val samples one
                                     # sample ≈ 2.2pt, so ε=0.02 ≡ ε=0 and the noisy best-so-far
                                     # closes the gate by chance (see ESPFL_IMPLEMENTATION.md §4)
    eta: float = 0.5                 # trend gain η (damped: val-acc ratio noise is ±5-7%)
    eta_up: Optional[float] = None   # gain for ratio>1 (None → η, i.e. exactly the doc formula)
    lam_max: float = 0.2             # max per-transition channel contraction λ_max
    lam_init: Optional[float] = None  # λ_last initial value (default λ_max/2)
    lam_fixed: Optional[float] = None  # λ for 'fixed'/'gate_only' (default λ_init)
    delta_min: float = 0.03          # min effective contraction (parameter ratio)
    phi: float = 0.25                # per-layer floor: fraction of the client's initial width
    k_min: int = 4
    rho_min: float = 0.10            # global parameter-ratio floor
    layer_coef: str = 'uniform'      # uniform | original
    # legacy-only context
    dataset_name: str = 'cifar10'
    noniid_alpha: Optional[float] = None

    def __post_init__(self):
        if self.lam_init is None:
            self.lam_init = self.lam_max / 2
        if self.lam_fixed is None:
            self.lam_fixed = self.lam_init


@dataclass
class CapacityState:
    K0: Dict[str, int]                       # client's initial (budget) widths
    A_best: Optional[float] = None           # best validation acc at earlier observation points
    A_last: Optional[float] = None           # validation acc at the last executed contraction
    lam_last: float = 0.1
    stopped: bool = False
    stop_reason: str = ''
    # legacy controller
    legacy_initial_acc: Optional[float] = None
    legacy_history: List[dict] = field(default_factory=list)


@dataclass
class CapacityDecision:
    executed: bool
    permit: bool
    lam: float
    targets: Optional[Dict[str, int]]
    rho_before: float
    rho_after: float
    shrink: float
    reason: str
    A: float
    A_best_prev: Optional[float]


class CapacityController:
    def __init__(self, spec: ArchSpec, cfg: ControllerConfig):
        self.spec, self.cfg = spec, cfg
        self.coef = layer_coefficients(spec, 'original' if cfg.mode == 'legacy' else cfg.layer_coef)
        if cfg.mode == 'legacy' and spec.arch == 'vgg16':
            for g in spec.prunable:                 # original implementation never pruned the projector
                if g.startswith('h'):
                    self.coef[g] = 0.0

    def new_state(self, K0: Dict[str, int]) -> CapacityState:
        return CapacityState(K0=dict(K0), lam_last=self.cfg.lam_init)

    # ── helpers ──────────────────────────────────────────────────────────
    def _candidate(self, st: CapacityState, cur: Dict[str, int], lam: float) -> Dict[str, int]:
        c = self.cfg
        w = dict(cur)
        for g in self.spec.prunable:
            K = cur[g]
            floor = min(K, max(math.ceil(c.phi * st.K0[g]), c.k_min))
            w[g] = max(floor, K - int(math.floor(lam * self.coef[g] * K)))
        return w

    def _targets(self, cur: Dict[str, int], cand: Dict[str, int]) -> Dict[str, int]:
        return {g: cand[g] for g in self.spec.prunable if cand[g] < cur[g]}

    # ── main entry ───────────────────────────────────────────────────────
    def decide(self, st: CapacityState, A: float, cur: Dict[str, int]) -> CapacityDecision:
        if self.cfg.mode == 'legacy':
            return self._decide_legacy(st, A, cur)
        c, spec = self.cfg, self.spec
        A_best_prev = st.A_best
        rho_before = spec.rho(cur)

        use_gate = c.mode in ('evidence', 'gate_only')
        use_trend = c.mode in ('evidence', 'trend_only')

        # 1. level gate: can we contract?
        permit = (not use_gate) or A_best_prev is None or A >= (1.0 - c.eps) * A_best_prev
        # 2. trend intensity: by how much?  (anchored at the last executed contraction)
        if use_trend:
            ratio = A / st.A_last if (st.A_last is not None and st.A_last > 0) else 1.0
            gain = c.eta if (ratio < 1.0 or c.eta_up is None) else c.eta_up
            lam = float(np.clip(st.lam_last + gain * (ratio - 1.0), 0.0, c.lam_max))
        else:
            lam = c.lam_fixed

        dec = CapacityDecision(False, permit, lam, None, rho_before, rho_before, 0.0, '', A, A_best_prev)
        if not permit:
            dec.reason = 'gate'
        else:
            # 3. feasibility: per-layer floors, then global floor via bisection on λ
            cand = self._candidate(st, cur, lam)
            floor_bound = False
            if spec.rho(cand) < c.rho_min:
                lo, hi = 0.0, lam
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if spec.rho(self._candidate(st, cur, mid)) >= c.rho_min:
                        lo = mid
                    else:
                        hi = mid
                lam, floor_bound = lo, True
                cand = self._candidate(st, cur, lam)
            rho_after = spec.rho(cand)
            shrink = 1.0 - rho_after / rho_before
            dec.lam, dec.rho_after, dec.shrink = lam, rho_after, shrink
            # 4. execute or skip
            if shrink >= c.delta_min:
                dec.executed, dec.targets, dec.reason = True, self._targets(cur, cand), 'contract'
                st.A_last, st.lam_last = A, lam
            elif floor_bound:
                dec.reason, st.stopped, st.stop_reason = 'global_floor', True, 'global_floor'
            elif all(cur[g] <= max(math.ceil(c.phi * st.K0[g]), c.k_min) for g in spec.prunable):
                # every prunable layer already sits at its floor: no contraction can ever be
                # feasible again, so stopping is behaviour-identical and saves validation passes
                dec.reason, st.stopped, st.stop_reason = 'layer_floor', True, 'layer_floor'
            else:
                dec.reason = 'below_delta_min'
        # 5. best-so-far update
        st.A_best = A if A_best_prev is None else max(A_best_prev, A)
        return dec

    # ── original implementation's rules (on validation accuracy) ────────────
    def _decide_legacy(self, st: CapacityState, A: float, cur: Dict[str, int]) -> CapacityDecision:
        c, spec = self.cfg, self.spec
        rho_before = spec.rho(cur)
        A_best_prev = st.A_best
        st.A_best = A if A_best_prev is None else max(A_best_prev, A)
        if st.legacy_initial_acc is None:
            st.legacy_initial_acc = A
        init = st.legacy_initial_acc
        dec = CapacityDecision(False, True, 0.0, None, rho_before, rho_before, 0.0, '', A, A_best_prev)

        # stop rules (original _check_should_stop; thresholds were tuned on accuracy in %)
        stop = None
        if c.dataset_name == 'cifar100':
            if A / max(init, 1e-6) < 0.80: stop = 'perf<80%init'
            elif A < 28.0: stop = 'abs<28'
            elif rho_before < 0.20: stop = 'rho<0.20'
        else:
            min_rho = 0.10
            if c.dataset_name == 'cifar10' and c.noniid_alpha is not None and c.noniid_alpha <= 0.3 \
                    and A >= 82.0 and A / max(init, 1e-6) >= 0.98:
                min_rho = 0.25
            if A / max(init, 1e-6) < 0.9: stop = 'perf<90%init'
            elif A < 70.0: stop = 'abs<70'
            elif rho_before < min_rho: stop = f'rho<{min_rho}'
        if stop:
            dec.permit, dec.reason = False, f'legacy_stop:{stop}'
            return dec

        # sparsity rule (original _compute_sparsity)
        hist = st.legacy_history
        if c.dataset_name == 'cifar100':
            if not hist:
                sp = 0.08 if A >= 45 else 0.06 if A >= 38 else 0.05 if A >= 32 else 0.04 if A >= 28 else 0.03
            else:
                pa, ps = hist[-1]['A'], hist[-1]['sp']
                tr = A / max(pa, 1e-6)
                sp = (min(0.09, ps + 0.015) if tr >= 1.02 else min(0.08, ps + 0.010) if tr >= 0.98
                      else max(0.03, ps - 0.005) if tr >= 0.95 else max(0.02, ps - 0.015))
        else:
            n03 = c.noniid_alpha is not None and c.noniid_alpha <= 0.3
            if not hist:
                if n03:
                    sp = 0.11 if A >= 84 else 0.09 if A >= 79 else 0.07 if A >= 74 else 0.05
                else:
                    sp = 0.12 if A >= 85 else 0.10 if A >= 80 else 0.08 if A >= 75 else 0.06
            else:
                pa, ps = hist[-1]['A'], hist[-1]['sp']
                tr = A / max(pa, 1e-6)
                sp = (min(0.15, ps + 0.04) if tr >= 1.02 else min(0.12, ps + 0.02) if tr >= 0.98
                      else max(0.04, ps - 0.01) if tr >= 0.95 else max(0.02, ps - 0.03))

        mcr = 0.32 if (c.dataset_name == 'cifar10' and c.noniid_alpha is not None and c.noniid_alpha <= 0.3) else 0.4
        kmin_legacy = 8 if spec.arch == 'vgg16' else 4
        cand = dict(cur)
        for g in spec.prunable:
            K = cur[g]
            min_ch = max(kmin_legacy, int(K * mcr))
            cand[g] = K - max(0, min(K - min_ch, int(K * sp * self.coef[g])))
        kept = sum(cand[g] for g in spec.prunable) / max(1, sum(cur[g] for g in spec.prunable))
        min_prune = ([0.02, 0.015, 0.01] if c.dataset_name == 'cifar100' else [0.05, 0.03, 0.02])[min(len(hist), 2)]
        rho_after = spec.rho(cand)
        dec.lam, dec.rho_after, dec.shrink = sp, rho_after, 1.0 - rho_after / rho_before
        if kept > 1.0 - min_prune:
            dec.reason = 'legacy_too_few'
            return dec
        dec.executed, dec.targets, dec.reason = True, self._targets(cur, cand), 'contract'
        hist.append({'A': A, 'sp': sp})
        st.A_last, st.lam_last = A, sp
        return dec
