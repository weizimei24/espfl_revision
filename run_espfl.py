"""
run_espfl.py
============
Command-line entry for ESPFL runs (pre-experiments, main results, ablations, sensitivity).

Examples
--------
# pre-experiment trio (doc §0): new controller / original controller / new + original layer coefficients
python run_espfl.py --dataset cifar10 --model vgg16 --alpha 0.3 --preset new
python run_espfl.py --dataset cifar10 --model vgg16 --alpha 0.3 --preset legacy
python run_espfl.py --dataset cifar10 --model vgg16 --alpha 0.3 --preset new_origcoef

# ablations
python run_espfl.py ... --preset fixed          # no level gate, no trend (constant λ)
python run_espfl.py ... --preset no_aki         # direct switch, CE only
python run_espfl.py ... --preset zerofill       # w/o SPPA (zero-fill aggregation)
python run_espfl.py ... --preset random_comp    # random channel composition
python run_espfl.py ... --preset no_easa        # structures fixed at the budget tier

# sensitivity
python run_espfl.py ... --preset new --eps 0.05 --eta 0.5

Outputs (in --out_dir/<tag>/): *_output.log, *_results.pkl, *_structure.json, *_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import asdict
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from data_fed import build_federated_dataloaders
from espfl_easa import ControllerConfig
from espfl_method import ESPFLConfig, ESPFLMethod
from federated_trainer import FederatedTrainer
from probe_utils import select_probe_clients, print_probe_summary


PRESETS = {
    # name           : overrides applied before explicit CLI flags
    'new':           {},
    'legacy':        {'controller.mode': 'legacy'},
    'new_origcoef':  {'controller.layer_coef': 'original'},
    # candidate fix for the training-progress confound (changes Eq. cap_intensity if adopted):
    # decreases of the val ratio act with η, increases only with η_up
    'new_asym':      {'controller.eta': 1.0, 'controller.eta_up': 0.1},
    'fixed':         {'controller.mode': 'fixed'},
    'gate_only':     {'controller.mode': 'gate_only'},
    'trend_only':    {'controller.mode': 'trend_only'},
    'no_easa':       {'S_total': 0},
    'no_aki':        {'aki_mode': 'none'},
    'aki_fixed':     {'aki_mode': 'fixed_gamma'},
    'aki_gate_only': {'aki_mode': 'gate_only'},
    'aki_kd_only':   {'aki_mode': 'kd_only'},
    'zerofill':      {'aggregation': 'zerofill'},
    'random_comp':   {'composition': 'random'},
    'magnitude_comp': {'composition': 'magnitude'},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> ESPFLConfig:
    cc = ControllerConfig(noniid_alpha=None if args.alpha <= 0 else args.alpha)
    cfg = ESPFLConfig(controller=cc, local_epochs=args.epochs, seed=args.seed)
    for key, val in PRESETS[args.preset].items():
        obj, attr = (cfg.controller, key.split('.', 1)[1]) if key.startswith('controller.') else (cfg, key)
        setattr(obj, attr, val)
    # explicit flags override the preset
    for flag, target in [('eps', 'controller.eps'), ('eta', 'controller.eta'),
                         ('eta_up', 'controller.eta_up'),
                         ('lam_max', 'controller.lam_max'), ('delta_min', 'controller.delta_min'),
                         ('phi', 'controller.phi'), ('k_min', 'controller.k_min'),
                         ('rho_min', 'controller.rho_min'), ('layer_coef', 'controller.layer_coef'),
                         ('controller', 'controller.mode'), ('lam_fixed', 'controller.lam_fixed'),
                         ('T_part', 'T_part'), ('S_total', 'S_total'), ('optimizer', 'optimizer'),
                         ('val_metric', 'val_metric'), ('aki_mode', 'aki_mode'),
                         ('aggregation', 'aggregation'), ('composition', 'composition')]:
        v = getattr(args, flag)
        if v is None:
            continue
        if target.startswith('controller.'):
            setattr(cfg.controller, target.split('.', 1)[1], v)
        else:
            setattr(cfg, target, v)
    cfg.controller.lam_init = cfg.controller.lam_max / 2 if args.lam_max is not None else cfg.controller.lam_init
    if args.lam_fixed is None and args.lam_max is not None:
        cfg.controller.lam_fixed = cfg.controller.lam_init
    cfg.prune_fc_hidden = not args.no_prune_fc_hidden
    cfg.log_local_test_acc = not args.no_monitor
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--model', default='vgg16')
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--alpha', type=float, default=0.3, help='Dirichlet alpha; <=0 means IID')
    ap.add_argument('--clients', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--frac', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--local_train_ratio', type=float, default=0.9)
    ap.add_argument('--preset', default='new', choices=sorted(PRESETS))
    for f, t in [('eps', float), ('eta', float), ('eta_up', float), ('lam_max', float), ('lam_fixed', float),
                 ('delta_min', float), ('phi', float), ('rho_min', float), ('k_min', int),
                 ('T_part', int), ('S_total', int)]:
        ap.add_argument(f'--{f}', type=t, default=None)
    ap.add_argument('--layer_coef', choices=['uniform', 'original'], default=None)
    ap.add_argument('--controller', choices=['evidence', 'fixed', 'gate_only', 'trend_only', 'legacy'], default=None)
    ap.add_argument('--optimizer', choices=['sgd', 'adamw'], default=None)
    ap.add_argument('--val_metric', choices=['acc', 'soft_acc'], default=None)
    ap.add_argument('--aki_mode', choices=['full', 'fixed_gamma', 'gate_only', 'kd_only', 'none'], default=None)
    ap.add_argument('--aggregation', choices=['sppa', 'zerofill'], default=None)
    ap.add_argument('--composition', choices=['sed', 'magnitude', 'random'], default=None)
    ap.add_argument('--no_prune_fc_hidden', action='store_true')
    ap.add_argument('--no_monitor', action='store_true', help='skip per-round local-test monitoring')
    ap.add_argument('--eval_interval', type=int, default=10)
    ap.add_argument('--n_probe', type=int, default=20)
    ap.add_argument('--out_dir', default='./runs_espfl')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    cfg = build_config(args)
    alpha_tag = 'iid' if args.alpha <= 0 else f'a{args.alpha:g}'
    tag = args.tag or f"{args.dataset}_{args.model}_{alpha_tag}_{args.preset}_s{args.seed}"
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # tee stdout into a log file
    log_f = open(os.path.join(out, f'{stamp}_output.log'), 'a', encoding='utf-8')

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, m):
            for x in self.s: x.write(m)
        def flush(self):
            for x in self.s: x.flush()
    sys.stdout = _Tee(sys.__stdout__, log_f)
    print(f"[run_espfl] args={vars(args)}")
    print(f"[run_espfl] config={json.dumps(asdict(cfg), default=str)}")

    device = torch.device(args.device)
    train_l, val_l, test_l, global_l = build_federated_dataloaders(
        num_clients=args.clients, dataset_name=args.dataset, data_root=args.data_root,
        iid=args.alpha <= 0, noniid_alpha=args.alpha if args.alpha > 0 else 0.5,
        batch_size=args.bs, local_train_ratio=args.local_train_ratio, augment_train=True,
        seed=args.seed, num_workers=args.num_workers, val_ratio=args.val_ratio,
        fix_eval_transform=True, return_val=True)
    sizes = [len(train_l[i].dataset) for i in range(args.clients)]
    vsizes = [len(val_l[i].dataset) for i in range(args.clients)]
    print(f"[data] train/client mean={np.mean(sizes):.1f} min={min(sizes)}; "
          f"val/client mean={np.mean(vsizes):.1f} min={min(vsizes)}")

    method = ESPFLMethod(args.clients, args.model, args.dataset, device, cfg, val_l)
    probe_ids = select_probe_clients(method.client_scales, n_probe=args.n_probe,
                                     ratio_sml=(0.30, 0.30, 0.40), seed=42)
    print_probe_summary(probe_ids, method.client_scales)

    trainer = FederatedTrainer(method=method, num_clients=args.clients, num_rounds=args.rounds,
                               client_selection_fraction=args.frac, device=device, seed=args.seed,
                               probe_client_ids=probe_ids, eval_interval=args.eval_interval)
    results = trainer.run(train_loaders=train_l, test_loaders=test_l, global_testloader=global_l)

    # final evaluation of ALL clients' personalized models on their local test sets (paper Eq. 34)
    final_local = method.evaluate_local_clients(list(range(args.clients)), test_l, device)
    results['final_local_all'] = final_local
    rep = method.structure_report()
    tiers = {}
    for i in range(args.clients):
        tiers.setdefault(method.client_scales[i], []).append(i)
    rl = rep['rounds']
    summary = {
        'tag': tag, 'preset': args.preset,
        'local_mean_all': float(np.mean(list(final_local.values()))),
        'local_by_tier': {str(s): float(np.mean([final_local[i] for i in ids])) for s, ids in tiers.items()},
        'global_final': results['round_accuracies'][-1] if results['round_accuracies'] else None,
        'global_best': max(results['round_accuracies']) if results['round_accuracies'] else None,
        'comm_avg_params_per_participant_round':
            float(sum(r['uplink_params'] for r in rl) / max(1, sum(r['n_clients'] for r in rl))),
        'comm_avg_ratio_vs_full':
            float(sum(r['uplink_params'] for r in rl) / max(1, sum(r['n_clients'] for r in rl)) / method.spec.d),
        'final_rho_by_tier': {str(s): float(np.mean([method.spec.rho(method._widths(i)) for i in ids]))
                              for s, ids in tiers.items()},
        'initial_rho_by_tier': {str(s): method.spec.rho(method.spec.scale_widths(s)) for s in tiers},
        'transitions_mean': float(np.mean([method.stage[i] for i in range(args.clients)])),
        'transitions_hist': {str(k): int(sum(1 for i in range(args.clients) if method.stage[i] == k))
                             for k in range(cfg.S_total + 1)},
        'probe_final_avg': results['probe_local_eval']['avg'][-1] if results['probe_local_eval']['avg'] else None,
    }
    print(f"\n[summary] {json.dumps(summary, indent=2)}")
    with open(os.path.join(out, f'{stamp}_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(out, f'{stamp}_structure.json'), 'w') as f:
        json.dump(rep, f, default=str)
    with open(os.path.join(out, f'{stamp}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    main()
