"""
espfl_method.py
===============
ESPFL (final architecture, 2026-11-06) on top of the FederatedTrainer / BaseMethod interface.

Protocol per selected client (Algorithm 1)
------------------------------------------
    if a pending successor mask exists           -> AKI activation round, then activate it
    else                                         -> regular local training (+ S/E/D statistics)
         if p_i % T_part == 0, s_i < S_total, ρ(M_i) > ρ_min, not stopped
             -> validation accuracy -> capacity decision -> (composition decision) -> cache
                pending mask + teacher; it takes effect at the NEXT participation
    upload Δ on the active coordinates Ω_i (new mask after an activation round)
Server: SPPA (coordinate-wise, normalised over observers O_j) or zero-fill ('w/o SPPA').

Data roles (strict):
    D_tr  : local optimisation, trajectory statistics, AKI
    D_val : capacity decisions only (deterministic evaluation, fixed set)
    D_te  : personalized evaluation only -- never read by any decision in this file
            (local_update logs local test accuracy for monitoring only;
             nothing downstream consumes it).
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod
from slim_models import (ArchSpec, IndexSet, make_arch_spec, extract_state,
                         relative_positions, CoverageAccumulator)
from espfl_easa import (TrajectoryStats, compose_successor, channel_weight_norms,
                        ControllerConfig, CapacityController)


@dataclass
class ESPFLConfig:
    # protocol
    T_part: int = 15
    S_total: int = 4
    local_epochs: int = 5
    # optimiser for local training
    optimizer: str = 'sgd'            # 'sgd' | 'adamw'
    momentum: float = 0.9
    weight_decay: float = 1e-4
    adamw_lr: float = 3e-4
    # budgets
    capacity_distribution: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.00, 0.40), (0.71, 0.30), (0.50, 0.30)])
    assign_seed: int = 42
    prune_fc_hidden: bool = True      # VGG projector hidden units are prunable layers
    # EASA
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    composition: str = 'sed'          # sed | magnitude | random
    ema_decay: float = 0.9
    gamma_vol: float = 1.0
    quantiles: Tuple[float, float] = (0.05, 0.95)
    val_metric: str = 'acc'           # acc | soft_acc (mean prob. of the true class)
    # AKI
    aki_mode: str = 'full'            # full | fixed_gamma | gate_only | kd_only | none
    aki_T: float = 2.5
    aki_xi: float = 0.5
    aki_cmax: float = 2.0
    aki_gamma_fixed: float = 0.5
    aki_dpi_norm: str = 'template'    # Eq. 20 denominator: 'template' (paper) | 'current'
    log_activation_metrics: bool = True
    # SPPA
    aggregation: str = 'sppa'         # sppa | zerofill
    server_lr: float = 1.0
    # logging
    log_local_test_acc: bool = True   # monitoring only (never used for decisions)
    seed: int = 1


def _evaluate(model: nn.Module, loader, device, metric: str = 'acc') -> Tuple[float, float]:
    """Returns (metric in %, mean CE loss). Deterministic given a non-shuffled loader."""
    model.eval()
    n, correct, soft, loss = 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss += F.cross_entropy(out, y, reduction='sum').item()
            correct += (out.argmax(1) == y).sum().item()
            if metric == 'soft_acc':
                soft += F.softmax(out, 1).gather(1, y.view(-1, 1)).sum().item()
            n += y.numel()
    if n == 0:
        return 0.0, 0.0
    val = (soft if metric == 'soft_acc' else correct) / n * 100.0
    return val, loss / n


def _kl_teacher_student(t_logits, s_logits, T: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1), reduction='batchmean')


class ESPFLMethod(BaseMethod):

    def __init__(self, num_clients: int, model_name: str, dataset_name: str,
                 device: torch.device, cfg: ESPFLConfig, val_loaders: Dict[int, object]):
        self.N = num_clients
        self.num_clients = num_clients
        self.model_name, self.dataset_name = model_name.lower(), dataset_name.lower()
        self.device, self.cfg = device, cfg
        if val_loaders is None:
            raise ValueError("ESPFL needs per-client validation loaders (set VAL_RATIO > 0).")
        self.val_loaders = val_loaders

        self.spec: ArchSpec = make_arch_spec(self.model_name, self.dataset_name, cfg.prune_fc_hidden)
        cfg.controller.dataset_name = self.dataset_name
        self.controller = CapacityController(self.spec, cfg.controller)

        self.global_model = self.spec.build(self.spec.full_widths)
        self.capacity_distribution = cfg.capacity_distribution
        self.client_scales = self._assign_client_scales()
        self.client_base_scales = dict(self.client_scales)

        self.idx: Dict[int, IndexSet] = {}
        self.cap = {}
        self.traj: Dict[int, TrajectoryStats] = {}
        for i in range(self.N):
            self.idx[i] = self.spec.prefix_index(self.client_scales[i])
            self.cap[i] = self.controller.new_state(self.spec.widths_of(self.idx[i]))
            self.traj[i] = TrajectoryStats(self.spec, cfg.ema_decay)
        self.stage = {i: 0 for i in range(self.N)}
        self.part = {i: 0 for i in range(self.N)}             # regular participations in stage
        self.pending: Dict[int, Optional[IndexSet]] = {i: None for i in range(self.N)}
        self.teacher: Dict[int, Optional[dict]] = {i: None for i in range(self.N)}
        self.local_state: Dict[int, Optional[dict]] = {i: None for i in range(self.N)}
        self.events: Dict[int, List[dict]] = {i: [] for i in range(self.N)}
        self._uploads: Dict[int, tuple] = {}
        self._round_global: Optional[dict] = None
        self.round_log: List[dict] = []
        self._rng = torch.Generator().manual_seed(cfg.seed)
        self._round = 0

        tiers = {}
        for s in self.client_scales.values():
            tiers[s] = tiers.get(s, 0) + 1
        print(f"[ESPFL] {self.model_name}/{self.dataset_name}: template d={self.spec.d:,}, "
              f"tiers={tiers}, prunable groups={len(self.spec.prunable)}")
        for s in sorted(tiers, reverse=True):
            print(f"[ESPFL]   scale {s:.2f}: rho0={self.spec.rho(self.spec.scale_widths(s)):.3f}")
        cc = cfg.controller
        print(f"[ESPFL] protocol T_part={cfg.T_part}, S_total={cfg.S_total}; controller={cc.mode} "
              f"(eps={cc.eps}, eta={cc.eta}, lam_max={cc.lam_max}, delta_min={cc.delta_min}, "
              f"phi={cc.phi}, k_min={cc.k_min}, rho_min={cc.rho_min}, coef={cc.layer_coef}); "
              f"composition={cfg.composition}; AKI={cfg.aki_mode}; aggregation={cfg.aggregation}; "
              f"optimizer={cfg.optimizer}")

    # ─────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────
    def _assign_client_scales(self) -> Dict[int, float]:
        """Assign ESPFL budget tiers using a fixed seed, independently of client IDs."""
        order = list(range(self.N))
        np.random.default_rng(seed=self.cfg.assign_seed).shuffle(order)
        scales, k = {}, 0
        dist = self.cfg.capacity_distribution
        for t, (s, ratio) in enumerate(dist):
            cnt = self.N - k if t == len(dist) - 1 else min(int(round(self.N * ratio)), self.N - k)
            for _ in range(cnt):
                scales[order[k]] = s
                k += 1
        for i in range(self.N):
            scales.setdefault(i, 1.0)
        return scales

    def _widths(self, i: int) -> Dict[str, int]:
        return self.spec.widths_of(self.idx[i])

    def _model_from(self, widths: Dict[str, int], state: dict) -> nn.Module:
        m = self.spec.build(widths)
        ref = m.state_dict()
        m.load_state_dict({k: state[k].to(ref[k].dtype) for k in ref}, strict=True)
        return m.to(self.device)

    def _optimizer(self, model: nn.Module, lr: float):
        if self.cfg.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=self.cfg.momentum,
                                   weight_decay=self.cfg.weight_decay)
        if self.cfg.optimizer == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=self.cfg.adamw_lr,
                                     weight_decay=self.cfg.weight_decay)
        raise ValueError(self.cfg.optimizer)

    @staticmethod
    def _cpu(state: dict, half: bool = False) -> dict:
        return {k: (v.detach().cpu().half() if half else v.detach().cpu().float()).clone()
                for k, v in state.items()}

    def _global_state(self) -> dict:
        if self._round_global is None:
            self._round_global = self._cpu(self.global_model.state_dict())
        return self._round_global

    def _can_decide(self, i: int) -> bool:
        return (self.stage[i] < self.cfg.S_total and not self.cap[i].stopped
                and self.pending[i] is None
                and self.spec.rho(self._widths(i)) > self.cfg.controller.rho_min)

    # ─────────────────────────────────────────────────────────────────────
    # BaseMethod interface
    # ─────────────────────────────────────────────────────────────────────
    def distribute(self, selected_clients: List[int]) -> None:
        # Clients slice Θ_G on their own active coordinates inside local_update.
        self._round_global = self._cpu(self.global_model.state_dict())

    def local_update(self, selected_clients, train_loaders, test_loaders, round_idx, lr):
        self._round = round_idx
        stats = {}
        for i in selected_clients:
            t0 = time.time()
            if self.pending[i] is not None:
                info = self._activation_round(i, train_loaders[i], lr)
            else:
                info = self._regular_round(i, train_loaders[i], lr)
            acc = float('nan')
            if self.cfg.log_local_test_acc:      # monitoring only
                m = self._model_from(self._widths(i), self.local_state[i])
                acc, _ = _evaluate(m, test_loaders[i], self.device)
                del m
            params = self.spec.count(self._widths(i))
            stats[i] = {'accuracy': acc, 'params': params, 'compression': params / self.spec.d}
            print(f"  Client {i:3d} [s={self.stage[i]} p={self.part[i]:2d} "
                  f"rho={params / self.spec.d:.3f}] {info} | local-test(monitor)={acc:.2f}% "
                  f"| {time.time() - t0:.1f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats

    def aggregate(self, selected_clients, train_loaders) -> None:
        g_state = self._global_state()
        acc = CoverageAccumulator(self.spec, g_state)
        uplink = 0
        for i in selected_clients:
            if i not in self._uploads:
                continue
            delta, idx = self._uploads.pop(i)
            acc.add(delta, idx, float(len(train_loaders[i].dataset)))
            uplink += self.spec.count(self.spec.widths_of(idx))
        new_state, coverage = acc.apply(g_state, self.cfg.aggregation, self.cfg.server_lr)
        self.global_model.load_state_dict(new_state, strict=True)
        self._round_global = None
        self.round_log.append({'round': self._round + 1, 'uplink_params': uplink,
                               'n_clients': len(selected_clients), 'coverage': coverage,
                               'mean_rho_all': float(np.mean([self.spec.rho(self._widths(j))
                                                              for j in range(self.N)]))})
        del acc
        gc.collect()

    def get_global_model(self) -> nn.Module:
        return self.global_model

    def set_global_model(self, model: nn.Module) -> None:
        self.global_model = model

    def evaluate_local_clients(self, client_ids, test_loaders, device) -> Dict[int, float]:
        res = {}
        g = None
        for i in client_ids:
            state = self.local_state[i]
            if state is None:                    # never participated: its slice of Θ_G
                g = g or self._cpu(self.global_model.state_dict())
                state = extract_state(self.spec, g, self.idx[i])
            m = self._model_from(self._widths(i), state)
            res[i], _ = _evaluate(m, test_loaders[i], device)
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res

    # ─────────────────────────────────────────────────────────────────────
    # regular round (+ observation point)
    # ─────────────────────────────────────────────────────────────────────
    def _regular_round(self, i: int, loader, lr: float) -> str:
        g_state = self._global_state()
        idx, widths = self.idx[i], self._widths(i)
        start = extract_state(self.spec, g_state, idx)
        model = self._model_from(widths, start)
        opt = self._optimizer(model, lr)
        collect = self._can_decide(i)
        tr = self.traj[i]
        if collect:
            tr.begin(model)
        model.train()
        for _ in range(self.cfg.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                if collect:
                    tr.step()
                opt.step()
        if collect:
            tr.end()
        self.part[i] += 1

        trained = self._cpu(model.state_dict())
        self._uploads[i] = ({k: trained[k] - start[k] for k in trained}, idx)
        self.local_state[i] = self._cpu(trained, half=True)
        info = 'train'

        if collect and self.part[i] % self.cfg.T_part == 0:
            info = self._observation_point(i, model, trained)
        del model, opt
        return info

    def _observation_point(self, i: int, model: nn.Module, trained: dict) -> str:
        A, A_loss = _evaluate(model, self.val_loaders[i], self.device, self.cfg.val_metric)
        widths = self._widths(i)
        st = self.cap[i]
        dec = self.controller.decide(st, A, widths)
        ev = {'round': self._round + 1, 'type': 'obs', 'stage': self.stage[i], 'p': self.part[i],
              'A_val': A, 'val_loss': A_loss, 'A_best_prev': dec.A_best_prev, 'permit': dec.permit,
              'lam': dec.lam, 'rho_before': dec.rho_before, 'rho_after': dec.rho_after,
              'shrink': dec.shrink, 'executed': dec.executed, 'reason': dec.reason}
        if dec.executed:
            wn = channel_weight_norms(self.spec, model) if self.cfg.composition != 'sed' else None
            new_idx = compose_successor(self.traj[i], self.idx[i], dec.targets, self.cfg.composition,
                                        self.cfg.gamma_vol, self.cfg.quantiles, wn, self._rng)
            self.pending[i] = new_idx
            self.teacher[i] = self._cpu(trained, half=True)
            ev['pending_widths'] = self.spec.widths_of(new_idx)
            ev['rho_pending'] = self.spec.rho(self.spec.widths_of(new_idx))
        self.events[i].append(ev)
        tag = (f"OBS A_val={A:.2f}% best={dec.A_best_prev if dec.A_best_prev is not None else float('nan'):.2f} "
               f"permit={int(dec.permit)} lam={dec.lam:.3f} -> {dec.reason}")
        if dec.executed:
            tag += f" rho {dec.rho_before:.3f}->{dec.rho_after:.3f} (pending)"
        return tag

    # ─────────────────────────────────────────────────────────────────────
    # AKI activation round
    # ─────────────────────────────────────────────────────────────────────
    def _activation_round(self, i: int, loader, lr: float) -> str:
        cfg, spec = self.cfg, self.spec
        g_state = self._global_state()
        old_idx, new_idx = self.idx[i], self.pending[i]
        old_w, new_w = spec.widths_of(old_idx), spec.widths_of(new_idx)
        rel = relative_positions(old_idx, new_idx)

        mode = cfg.aki_mode
        use_gate = mode in ('full', 'fixed_gamma', 'gate_only')
        use_kd = mode in ('full', 'fixed_gamma', 'kd_only')

        # Δπ (Eq. 20) from the actual masks
        removed_ch = sum(old_w[g] - new_w[g] for g in spec.prunable)
        denom = (sum(spec.full_widths[g] for g in spec.prunable) if cfg.aki_dpi_norm == 'template'
                 else sum(old_w[g] for g in spec.prunable))
        d_pi = removed_ch / max(denom, 1)

        # student Θ_S ← Θ_G (on the old support when gating, else directly on the successor)
        s_idx, s_w = (old_idx, old_w) if use_gate else (new_idx, new_w)
        student = self._model_from(s_w, extract_state(spec, g_state, s_idx))
        removed_mask = {}
        if use_gate:
            for g in spec.prunable:
                if new_w[g] < old_w[g]:
                    keep = torch.zeros(old_w[g], dtype=torch.bool)
                    keep[rel[g]] = True
                    removed_mask[g] = (~keep).to(self.device)

        def set_gate(value: float):
            if not use_gate:
                return
            gates = {}
            for g, rm in removed_mask.items():
                gt = torch.ones(rm.numel(), device=self.device)
                gt[rm] = value
                gates[g] = gt
            student.set_gates(gates)

        teacher = None
        if use_kd or cfg.log_activation_metrics:
            teacher = self._model_from(old_w, self.teacher[i])
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        # Δout (Eq. 21): initialized student under the final successor mask, one local batch
        d_out = 0.0
        set_gate(0.0)
        if use_kd:
            x0, _ = next(iter(loader))
            with torch.no_grad():
                student.eval()
                kl = _kl_teacher_student(teacher(x0.to(self.device)), student(x0.to(self.device)), cfg.aki_T)
            d_out = float(min(max(kl.item(), 0.0), cfg.aki_cmax) / cfg.aki_cmax)
        if mode == 'full':
            gamma = float(np.clip(1.0 - d_pi + cfg.aki_xi * d_out, 0.1, 1.0))
        elif mode in ('fixed_gamma', 'kd_only'):
            gamma = cfg.aki_gamma_fixed if mode == 'fixed_gamma' else float(
                np.clip(1.0 - d_pi + cfg.aki_xi * d_out, 0.1, 1.0))
        else:
            gamma = 1.0

        metrics = {}
        if cfg.log_activation_metrics:
            val = self.val_loaders[i]
            metrics['teacher_val_acc'], metrics['teacher_val_loss'] = _evaluate(teacher, val, self.device)
            metrics['init_val_acc'], metrics['init_val_loss'] = _evaluate(student, val, self.device)

        # activation-round training: progressive gates 1 - r/R on removed channels (Eq. 15-16)
        opt = self._optimizer(student, lr)
        R = max(1, cfg.local_epochs * len(loader))
        r = 0
        student.train()
        for _ in range(cfg.local_epochs):
            for x, y in loader:
                r += 1
                set_gate(1.0 - r / R)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = student(x)
                loss = F.cross_entropy(out, y)
                if use_kd and gamma < 1.0:
                    with torch.no_grad():
                        t_out = teacher(x)
                    loss = gamma * loss + (1 - gamma) * cfg.aki_T ** 2 * _kl_teacher_student(t_out, out, cfg.aki_T)
                loss.backward()
                opt.step()
        student.set_gates(None)

        s_state = self._cpu(student.state_dict())
        final = extract_state(spec, s_state, rel) if use_gate else s_state
        start_new = extract_state(spec, g_state, new_idx)
        self._uploads[i] = ({k: final[k] - start_new[k] for k in final}, new_idx)

        # activate
        self.idx[i] = new_idx
        self.stage[i] += 1
        self.part[i] = 0
        self.traj[i].reset()
        self.pending[i], self.teacher[i] = None, None
        self.local_state[i] = self._cpu(final, half=True)

        if cfg.log_activation_metrics:
            m = self._model_from(new_w, final)
            metrics['final_val_acc'], metrics['final_val_loss'] = _evaluate(m, self.val_loaders[i], self.device)
            del m
        self.events[i].append({'round': self._round + 1, 'type': 'activate', 'stage': self.stage[i],
                               'd_pi': d_pi, 'd_out': d_out, 'gamma': gamma,
                               'rho': spec.rho(new_w), 'widths': new_w, **metrics})
        del student, teacher, opt
        msg = f"AKI({mode}) dpi={d_pi:.3f} dout={d_out:.3f} gamma={gamma:.2f} -> stage {self.stage[i]}"
        if metrics:
            msg += (f" | val acc teacher={metrics['teacher_val_acc']:.1f} init={metrics['init_val_acc']:.1f} "
                    f"final={metrics['final_val_acc']:.1f}")
        return msg

    # ─────────────────────────────────────────────────────────────────────
    # reporting
    # ─────────────────────────────────────────────────────────────────────
    def structure_report(self) -> dict:
        clients = {}
        for i in range(self.N):
            clients[str(i)] = {
                'tier_scale': self.client_scales[i],
                'stage': self.stage[i],
                'rho': self.spec.rho(self._widths(i)),
                'rho0': self.spec.rho(self.spec.scale_widths(self.client_scales[i])),
                'widths': self._widths(i),
                'stopped': self.cap[i].stopped, 'stop_reason': self.cap[i].stop_reason,
                'pending_at_end': self.pending[i] is not None,
                'indices': {g: t.tolist() for g, t in self.idx[i].items() if g in self.spec.prunable},
                'events': self.events[i],
            }
        cfg = asdict(self.cfg)
        return {'config': cfg, 'template_params': self.spec.d,
                'prunable_groups': self.spec.prunable, 'clients': clients, 'rounds': self.round_log}
