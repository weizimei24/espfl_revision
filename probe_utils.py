"""
probe_utils.py
==============
固定评估子集（Probe Clients）的选取工具。

设计目标：
    - 固定 ESPFL 评估客户端，便于跨实验比较
    - 按 S/M/L 容量比例 3:3:4 采样，覆盖不同资源约束的客户端
    - 由 seed 唯一决定，只要 seed / num_clients / capacity_assignment 不变，
      每次调用返回完全相同的 ID 列表

使用方式：
    from probe_utils import select_probe_clients, print_probe_summary

    # 在 main.py 中，method 初始化之后调用（此时 client_scales 已确定）
    probe_ids = select_probe_clients(
        client_scales   = method.client_scales,   # {cid: scale}
        n_probe         = 20,
        ratio_sml       = (0.30, 0.30, 0.40),     # S:M:L = 3:3:4
        seed            = 42,                      # 固定种子，所有实验相同
    )
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


# ESPFL 的 S/M/L 容量档位阈值
_SCALE_L = 0.99   # scale >= 0.99 → Large
_SCALE_M = 0.65   # 0.65 <= scale < 0.99 → Medium
                  # scale < 0.65 → Small


def _classify_scale(scale: float) -> str:
    if scale >= _SCALE_L:
        return 'L'
    elif scale >= _SCALE_M:
        return 'M'
    else:
        return 'S'


def select_probe_clients(
    client_scales: Dict[int, float],
    n_probe: int = 20,
    ratio_sml: Tuple[float, float, float] = (0.30, 0.30, 0.40),
    seed: int = 42,
) -> List[int]:
    """
    从 client_scales 中按 S:M:L 比例随机选取 n_probe 个固定探针客户端。

    参数
    ----
    client_scales : {client_id: scale}，来自 method.client_scales
    n_probe       : 总探针客户端数（默认 20）
    ratio_sml     : (S比例, M比例, L比例)，默认 (0.30, 0.30, 0.40) 即 3:3:4
    seed          : 随机种子，保证可复现

    返回
    ----
    probe_ids : 排序后的客户端 ID 列表，长度为 n_probe
    """
    rng = np.random.default_rng(seed=seed)

    r_s, r_m, r_l = ratio_sml
    assert abs(r_s + r_m + r_l - 1.0) < 1e-6, "ratio_sml 之和必须为 1.0"

    n_s = round(n_probe * r_s)
    n_m = round(n_probe * r_m)
    n_l = n_probe - n_s - n_m   # 余数给 L，保证总数精确

    # 按 scale 分组
    group_l = [cid for cid, s in client_scales.items() if _classify_scale(s) == 'L']
    group_m = [cid for cid, s in client_scales.items() if _classify_scale(s) == 'M']
    group_s = [cid for cid, s in client_scales.items() if _classify_scale(s) == 'S']

    # 防御：若某组客户端不足，从其他组补充
    def _safe_sample(pool: List[int], n: int) -> List[int]:
        if len(pool) == 0:
            return []
        n = min(n, len(pool))
        return rng.choice(pool, size=n, replace=False).tolist()

    sampled_l = _safe_sample(sorted(group_l), n_l)
    sampled_m = _safe_sample(sorted(group_m), n_m)
    sampled_s = _safe_sample(sorted(group_s), n_s)

    # 若某组不足，从其他组补足
    total_sampled = len(sampled_l) + len(sampled_m) + len(sampled_s)
    if total_sampled < n_probe:
        already = set(sampled_l + sampled_m + sampled_s)
        remaining = [cid for cid in client_scales if cid not in already]
        extra = _safe_sample(remaining, n_probe - total_sampled)
        sampled_l += extra

    probe_ids = sorted(sampled_l + sampled_m + sampled_s)
    return probe_ids


def print_probe_summary(
    probe_ids: List[int],
    client_scales: Dict[int, float],
) -> None:
    """打印探针客户端的组成统计，便于日志核查。"""
    groups = {'L': [], 'M': [], 'S': []}
    for cid in probe_ids:
        g = _classify_scale(client_scales.get(cid, 1.0))
        groups[g].append(cid)

    print(f"[ProbeClients] Total={len(probe_ids)} | "
          f"L={len(groups['L'])} {groups['L']} | "
          f"M={len(groups['M'])} {groups['M']} | "
          f"S={len(groups['S'])} {groups['S']}")
