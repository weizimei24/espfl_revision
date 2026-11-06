"""
Correctness tests for the ESPFL implementation.

    python tests/test_espfl.py          (or: pytest tests/test_espfl.py)

Covers: template == models.build_model; index slicing == gating removed channels to 0;
parameter counting; SPPA / zero-fill arithmetic; nested extraction; capacity controller
branches; composition guarantees; an end-to-end smoke run of the full protocol.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from slim_models import (make_arch_spec, extract_state, relative_positions,
                         CoverageAccumulator, check_against_reference)
from espfl_easa import (TrajectoryStats, compose_successor, ControllerConfig,
                        CapacityController)

torch.manual_seed(0)


def _random_nested(spec, outer, frac=0.6, protect=('input', 'classes')):
    inner = {}
    for g, t in outer.items():
        if g in protect or not spec.groups[g].prunable:
            inner[g] = t.clone()
            continue
        k = max(1, int(t.numel() * frac))
        sel = torch.sort(torch.randperm(t.numel())[:k]).values
        inner[g] = t[sel]
    return inner


def test_template_matches_reference():
    for m, d in [('vgg16', 'cifar10'), ('resnet18', 'cifar10'), ('resnet18', 'cifar100')]:
        check_against_reference(m, d)


def test_slicing_equals_gating_and_counts():
    for arch in ('vgg16', 'resnet18'):
        spec = make_arch_spec(arch, 'cifar10')
        full = spec.build(spec.full_widths).eval()
        g_state = {k: v.clone() for k, v in full.state_dict().items()}
        idx = _random_nested(spec, spec.full_index(), 0.5)
        sub = spec.build(spec.widths_of(idx)).eval()
        sub.load_state_dict(extract_state(spec, g_state, idx), strict=True)
        assert sum(p.numel() for p in sub.parameters()) == spec.count(spec.widths_of(idx))
        gates = {}
        for g in spec.prunable:
            gt = torch.zeros(spec.full_widths[g])
            gt[idx[g]] = 1.0
            gates[g] = gt
        full.set_gates(gates)
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            a, b = full(x), sub(x)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (arch, (a - b).abs().max())


def test_nested_extraction():
    spec = make_arch_spec('resnet18', 'cifar10')
    g_state = spec.build(spec.full_widths).state_dict()
    old = _random_nested(spec, spec.prefix_index(0.71), 0.8)
    new = _random_nested(spec, old, 0.7)
    rel = relative_positions(old, new)
    two_step = extract_state(spec, extract_state(spec, g_state, old), rel)
    direct = extract_state(spec, g_state, new)
    for k in direct:
        assert torch.equal(two_step[k], direct[k]), k


def test_sppa_and_zerofill():
    spec = make_arch_spec('vgg16', 'cifar10')
    g_state = {k: torch.zeros_like(v) for k, v in spec.build(spec.full_widths).state_dict().items()}
    idx_a = spec.prefix_index(1.0)
    idx_b = spec.prefix_index(0.5)
    da = {k: torch.ones_like(v) for k, v in extract_state(spec, g_state, idx_a).items()}
    db = {k: 3 * torch.ones_like(v) for k, v in extract_state(spec, g_state, idx_b).items()}
    for mode, exp_shared, exp_only_a in [('sppa', (1 * 1 + 3 * 3) / 4, 1.0),
                                         ('zerofill', (1 * 1 + 3 * 3) / 4, 1 * 1 / 4)]:
        acc = CoverageAccumulator(spec, g_state)
        acc.add(da, idx_a, 1.0)
        acc.add(db, idx_b, 3.0)
        new, cov = acc.apply(g_state, mode)
        w = new['features.3.weight']               # conv 1: (64, 64, 3, 3)
        assert math.isclose(w[0, 0, 0, 0].item(), exp_shared, rel_tol=1e-6)
        assert math.isclose(w[63, 63, 0, 0].item(), exp_only_a, rel_tol=1e-6), (mode, w[63, 63, 0, 0])
        assert cov == 1.0
    # an unobserved coordinate keeps its value under SPPA
    acc = CoverageAccumulator(spec, g_state)
    acc.add(db, idx_b, 1.0)
    new, cov = acc.apply(g_state, 'sppa')
    assert new['features.3.weight'][63, 63, 0, 0].item() == 0.0 and cov < 1.0


def test_controller_branches():
    spec = make_arch_spec('resnet18', 'cifar10')
    cfg = ControllerConfig(eps=0.02, eta=1.0, lam_max=0.2, delta_min=0.03, phi=0.25, rho_min=0.10)
    ctl = CapacityController(spec, cfg)
    w0 = spec.scale_widths(1.0)
    st = ctl.new_state(w0)
    d = ctl.decide(st, 80.0, w0)                    # first point: permitted, ratio=1 -> λ = λ_init
    assert d.permit and d.executed and abs(d.lam - 0.1) < 1e-9 and st.A_last == 80.0
    w1 = {**w0, **d.targets}
    d = ctl.decide(st, 76.0, w1)                    # 76 < 0.98*80 -> gate closes, λ_last unchanged
    assert not d.permit and not d.executed and st.lam_last == 0.1 and st.A_best == 80.0
    d = ctl.decide(st, 84.0, w1)                    # recovered above best; trend +5% -> λ = 0.15
    assert d.permit and d.executed and abs(d.lam - 0.15) < 1e-9
    # global floor: S-tier client near ρ_min -> λ reduced by bisection, or stop
    ws = spec.scale_widths(0.5)
    cfg2 = ControllerConfig(rho_min=spec.rho(ws) * 0.99, delta_min=0.03)
    ctl2 = CapacityController(spec, cfg2)
    st2 = ctl2.new_state(ws)
    d = ctl2.decide(st2, 70.0, ws)
    assert not d.executed and st2.stopped and d.reason == 'global_floor'
    # per-layer floors are respected and nestedness of widths holds
    st3 = ctl.new_state(w0)
    cur = dict(w0)
    for _ in range(30):
        st3.lam_last = 0.2
        st3.A_last = None
        dd = ctl.decide(st3, 90.0, cur)
        if not dd.executed:
            break
        for g, K in dd.targets.items():
            assert K < cur[g] and K >= max(math.ceil(0.25 * w0[g]), 4)
            cur[g] = K
    assert spec.rho(cur) >= cfg.rho_min


def test_composition():
    spec = make_arch_spec('vgg16', 'cifar10')
    model = spec.build(spec.scale_widths(0.71))
    cur = spec.prefix_index(0.71)
    tr = TrajectoryStats(spec)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    for _ in range(2):                              # two participations of 3 steps
        tr.begin(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            tr.step()
            opt.step()
        tr.end()
    targets = {g: max(1, int(cur[g].numel() * 0.8)) for g in spec.prunable}
    new = compose_successor(tr, cur, targets)
    for g in spec.prunable:
        assert new[g].numel() == targets[g]
        assert set(new[g].tolist()) <= set(cur[g].tolist())
        assert torch.all(new[g][1:] > new[g][:-1])


def test_three_way_split_keeps_old_test_set():
    import numpy as np
    from data_fed import _three_way_split
    for n in (7, 50, 523, 1400):
        idx = list(np.random.default_rng(n).permutation(10_000)[:n])
        split = max(1, min(int(n * 0.9), n - 1))
        old_train, old_test = idx[:split], idx[split:]
        tr, va, te = _three_way_split(idx, 0.9, 0.1)
        assert te == old_test                               # identical local test set
        assert tr + va == old_train and len(va) >= 1 and len(tr) >= 1
        assert not (set(tr) & set(va)) and not (set(va) & set(te))
        tr0, va0, te0 = _three_way_split(idx, 0.9, 0.0)
        assert tr0 == old_train and va0 == [] and te0 == old_test


def test_cli_config():
    import run_espfl
    ap_args = ['--preset', 'new_asym', '--lam_max', '0.3', '--eps', '0.07']
    import argparse
    # reuse the parser by calling build_config through main's parser
    parser_ns = argparse.Namespace(**{**{k: None for k in [
        'eps', 'eta', 'eta_up', 'lam_max', 'lam_fixed', 'delta_min', 'phi', 'rho_min', 'k_min',
        'T_part', 'S_total', 'layer_coef', 'controller', 'optimizer', 'val_metric', 'aki_mode',
        'aggregation', 'composition']},
        'preset': 'new_asym', 'lam_max': 0.3, 'eps': 0.07, 'alpha': 0.3, 'epochs': 5, 'seed': 1,
        'no_prune_fc_hidden': False, 'no_monitor': False})
    cfg = run_espfl.build_config(parser_ns)
    c = cfg.controller
    assert c.eta == 1.0 and c.eta_up == 0.1 and c.eps == 0.07 and c.lam_max == 0.3
    assert abs(c.lam_init - 0.15) < 1e-12 and c.noniid_alpha == 0.3


def test_end_to_end_smoke():
    """Tiny federation on synthetic data: all protocol branches must run."""
    from espfl_method import ESPFLConfig, ESPFLMethod
    from federated_trainer import FederatedTrainer
    N = 4
    def loader(n, shuffle):
        ds = TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))
        return DataLoader(ds, batch_size=8, shuffle=shuffle)
    train = {i: loader(16, True) for i in range(N)}
    val = {i: loader(8, False) for i in range(N)}
    test = {i: loader(8, False) for i in range(N)}
    glob = loader(16, False)
    for arch, agg, aki in [('resnet18', 'sppa', 'full'), ('vgg16', 'zerofill', 'none')]:
        cfg = ESPFLConfig(T_part=1, S_total=2, local_epochs=1, aggregation=agg, aki_mode=aki,
                          controller=ControllerConfig(mode='fixed', delta_min=0.01))
        m = ESPFLMethod(N, arch, 'cifar10', torch.device('cpu'), cfg, val)
        tr = FederatedTrainer(m, N, num_rounds=6, client_selection_fraction=1.0,
                              device=torch.device('cpu'), seed=0, probe_client_ids=[0, 1], eval_interval=3)
        tr.run(train, test, glob)
        assert all(m.stage[i] == 2 for i in range(N)), m.stage
        for i in range(N):
            acts = [e for e in m.events[i] if e['type'] == 'activate']
            assert len(acts) == 2
            assert m.spec.rho(m._widths(i)) < m.spec.rho(m.spec.scale_widths(m.client_scales[i]))
        rep = m.structure_report()
        assert len(rep['rounds']) == 6


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
