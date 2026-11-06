"""
federated_trainer.py
====================
ESPFL 联邦训练骨架。

职责：
    - 维护 round 循环
    - 随机选择客户端
    - 按顺序调用 method 的标准接口（distribute / local_update /
      post_round_hook / aggregate）
    - 执行全局模型评估
    - 每隔 eval_interval 轮对固定的 probe_clients 做本地测试集评估
    - 统一记录 results

不应包含：
    - 任何算法细节（聚合策略、剪枝逻辑、蒸馏参数等）
    - 对具体模型结构的感知（VGG / ResNet 等）
    - 对 ESPFL 客户端内部状态的直接访问

由 main.py 或 run_espfl.py 传入 ESPFLMethod 实例。
"""

from __future__ import annotations

import gc
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional

from base_method import BaseMethod
from train_utils import test_model


class FederatedTrainer:
    """
    ESPFL 联邦训练骨架。

    参数：
        method                   : 实现 BaseMethod 接口的 ESPFLMethod 实例
        num_clients              : 客户端总数
        num_rounds               : 联邦训练总轮数
        client_selection_fraction: 每轮随机选取的客户端比例（如 0.1 = 10%）
        device                   : 全局模型评估所用设备
        seed                     : 客户端选取的随机种子，None 表示不固定
        probe_client_ids         : 固定评估子集的客户端 ID 列表（跨实验保持不变）
                                   若为 None，则不做定期本地评估
        eval_interval            : 每隔多少轮对 probe_clients 做一次本地评估（默认 10）
    """

    def __init__(
        self,
        method: BaseMethod,
        num_clients: int,
        num_rounds: int,
        client_selection_fraction: float,
        device: torch.device,
        seed: Optional[int] = None,
        probe_client_ids: Optional[List[int]] = None,
        eval_interval: int = 10,
    ):
        self.method          = method
        self.num_clients     = num_clients
        self.num_rounds      = num_rounds
        self.num_selected    = max(1, int(num_clients * client_selection_fraction))
        self.device          = device
        self.probe_clients   = list(probe_client_ids) if probe_client_ids else []
        self.eval_interval   = eval_interval

        if seed is not None:
            np.random.seed(seed)

        # ── results 结构 ────────────────────────────────────────────────────
        self.results: Dict = {
            # 每轮全局测试集准确率
            'round_accuracies':          [],

            # 每个客户端参与训练时的本地测试准确率（训练轮次记录，稀疏）
            'client_accuracies':         {i: [] for i in range(num_clients)},
            'client_param_counts':       {i: [] for i in range(num_clients)},
            'client_compression_ratios': {i: [] for i in range(num_clients)},
            'selected_clients':          [],

            # 固定 probe 子集的定期本地评估结果
            # 格式：{'rounds': [10, 20, ...], 'per_client': {cid: [acc@r10, acc@r20, ...]}, 'avg': [...]}
            'probe_local_eval': {
                'rounds':     [],          # 触发评估的轮次编号（1-indexed）
                'per_client': {cid: [] for cid in self.probe_clients},
                'avg':        [],          # 每次评估时 probe_clients 的平均准确率
            },
        }

        print(
            f"[FederatedTrainer] "
            f"clients={num_clients}, rounds={num_rounds}, "
            f"select={self.num_selected}/round ({client_selection_fraction:.0%}), "
            f"device={device}"
        )
        if self.probe_clients:
            print(
                f"[FederatedTrainer] Probe clients ({len(self.probe_clients)}): "
                f"{self.probe_clients}  |  eval_interval={eval_interval} rounds"
            )

    # =========================================================================
    # 主训练循环
    # =========================================================================

    def run(
        self,
        train_loaders: dict,
        test_loaders: dict,
        global_testloader,
    ) -> Dict:
        """
        执行完整的联邦训练流程。

        参数：
            train_loaders     : {client_id: DataLoader}，客户端训练集
            test_loaders      : {client_id: DataLoader}，客户端本地测试集
            global_testloader : DataLoader，全局测试集（用于评估全局模型）

        返回：
            results 字典
        """
        client_sizes = {
            cid: len(train_loaders[cid].dataset)
            for cid in range(self.num_clients)
        }

        for round_idx in tqdm(range(self.num_rounds), desc="FL Training", unit="round"):
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1} / {self.num_rounds}")
            print(f"{'='*60}")

            # ── Step 1: 随机选取客户端 ────────────────────────────────────
            selected = np.random.choice(
                self.num_clients, self.num_selected, replace=False
            ).tolist()
            self.results['selected_clients'].append(selected)
            print(f"Selected {len(selected)} clients: {selected}")

            # ── Step 2: 全局模型分发到客户端 ──────────────────────────────
            self.method.distribute(selected)

            # ── Step 3: 客户端本地训练 ────────────────────────────────────
            lr = 3e-4 * (0.998 ** round_idx)
            client_stats = self.method.local_update(
                selected, train_loaders, test_loaders, round_idx, lr
            )

            for cid, stats in client_stats.items():
                self.results['client_accuracies'][cid].append(stats['accuracy'])
                self.results['client_param_counts'][cid].append(stats['params'])
                self.results['client_compression_ratios'][cid].append(stats['compression'])

            # ── Step 4: 服务器端聚合 ──────────────────────────────────────
            # 先聚合本轮 local_update 后的客户端模型。
            # 这样 global_acc 只反映“本轮本地训练 + 本轮聚合”的全局效果，
            # 不会混入当轮刚发生的个性化剪枝影响。
            self.method.aggregate(selected, train_loaders)
            gc.collect()
            torch.cuda.empty_cache()

            # ── Step 5: 全局模型评估 ──────────────────────────────────────
            # 此时还没有执行当轮个性化，因此全局准确率更干净。
            global_acc = self._evaluate_global_model(
                selected, train_loaders, global_testloader
            )
            self.results['round_accuracies'].append(global_acc)

            # ── Step 6: Probe 子集本地评估 ────────────────────────────────
            # ProbeEval 放在个性化之前：
            # 评估的是“本轮 local_update 后、当轮新个性化前”的客户端状态。
            # 避免刚剪枝但尚未经过下一轮训练/KD 恢复的模型影响 probe 指标。
            current_round = round_idx + 1   # 1-indexed
            if self.probe_clients and (current_round % self.eval_interval == 0):
                self._evaluate_probe_clients(
                    test_loaders  = test_loaders,
                    current_round = current_round,
                )

            # ── Step 7: 打印本轮统计 ──────────────────────────────────────
            # 此时 global_acc / ProbeLocalAvg 都已经记录，且都没有被当轮个性化污染。
            self._print_round_summary(round_idx, selected, global_acc, client_sizes)

            # ── Step 8: 轮后个性化检查 ───────────────────────────────────
            # 个性化只更新客户端本地状态，用于下一轮训练。
            # 不参与本轮聚合，也不影响本轮 global_acc / probe_eval。
            self.method.post_round_hook(selected, train_loaders, test_loaders, round_idx)

        return self.results

    # =========================================================================
    # Probe 子集评估
    # =========================================================================

    def _evaluate_probe_clients(
        self,
        test_loaders: dict,
        current_round: int,
    ) -> None:
        """
        对固定的 probe_clients 调用 method.evaluate_local_clients()，
        记录每个客户端的本地测试准确率和均值。
        """
        print(f"\n  [ProbeEval] Round {current_round}: evaluating "
              f"{len(self.probe_clients)} probe clients...")

        per_client_acc = self.method.evaluate_local_clients(
            client_ids  = self.probe_clients,
            test_loaders= test_loaders,
            device      = self.device,
        )

        # 写入 results
        probe_results = self.results['probe_local_eval']
        probe_results['rounds'].append(current_round)

        accs = []
        for cid in self.probe_clients:
            acc = per_client_acc.get(cid, 0.0)
            probe_results['per_client'][cid].append(acc)
            accs.append(acc)

        avg_acc = float(np.mean(accs)) if accs else 0.0
        probe_results['avg'].append(avg_acc)

        print(f"  [ProbeEval] Round {current_round}: "
              f"avg local acc = {avg_acc:.2f}%  "
              f"(min={min(accs):.2f}%, max={max(accs):.2f}%)")

    # =========================================================================
    # 内部辅助
    # =========================================================================

    def _evaluate_global_model(
        self,
        selected: list,
        train_loaders: dict,
        global_testloader,
    ) -> float:
        global_model = self.method.get_global_model().to(self.device)
        global_acc   = test_model(global_model, global_testloader, self.device)
        self.method.set_global_model(global_model.cpu())
        torch.cuda.empty_cache()
        return global_acc

    def _print_round_summary(
        self,
        round_idx: int,
        selected: list,
        global_acc: float,
        client_sizes: dict,
    ) -> None:
        avg_sel_acc = np.mean([
            self.results['client_accuracies'][i][-1]
            for i in selected
            if self.results['client_accuracies'][i]
        ])
        avg_sel_comp = np.mean([
            self.results['client_compression_ratios'][i][-1]
            for i in selected
            if self.results['client_compression_ratios'][i]
        ])
        all_last_comp = [
            self.results['client_compression_ratios'][i][-1]
            for i in range(self.num_clients)
            if self.results['client_compression_ratios'][i]
        ]
        avg_all_comp = float(np.mean(all_last_comp)) if all_last_comp else float('nan')

        # 若本轮恰好是 probe 轮，附加打印 probe avg
        probe_tag = ""
        if (self.probe_clients
                and self.results['probe_local_eval']['rounds']
                and self.results['probe_local_eval']['rounds'][-1] == round_idx + 1):
            probe_avg = self.results['probe_local_eval']['avg'][-1]
            probe_tag = f"  |  ProbeLocalAvg: {probe_avg:.2f}%"

        print(
            f"\n  [Round {round_idx + 1}] "
            f"Global Acc: {global_acc:.2f}%  |  "
            f"Avg Selected Acc: {avg_sel_acc:.2f}%  |  "
            f"Avg Selected Compression: {avg_sel_comp:.3f}  |  "
            f"Avg All Compression: {avg_all_comp:.3f}"
            f"{probe_tag}"
        )