"""
main.py
=======
ESPFL 实验入口：在 main() 中配置数据、模型和 ESPFL 超参数后直接运行。
命令行实验与消融配置见 run_espfl.py。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import logging
import pickle
import numpy as np
import torch
import psutil
from datetime import datetime
import json

from data_fed import build_federated_dataloaders
from federated_trainer import FederatedTrainer
from probe_utils import select_probe_clients, print_probe_summary
from espfl_easa import ControllerConfig
from espfl_method import ESPFLConfig, ESPFLMethod


# ─────────────────────────────────────────────────────────────────────────────
# 日志工具
# ─────────────────────────────────────────────────────────────────────────────

class RedirectStdoutToLog:
    def __init__(self, logger: logging.Logger):
        self.logger   = logger
        self.terminal = sys.stdout

    def write(self, message: str) -> None:
        self.terminal.write(message)
        if message.strip():
            self.logger.info(message.strip())

    def flush(self) -> None:
        self.terminal.flush()


def setup_logging(save_dir: str, timestamp: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{timestamp}_output.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_path,
        filemode="a",
    )
    sys.stdout = RedirectStdoutToLog(logging.getLogger())
    print(f"[Logging] Log file: {log_path}")


def print_memory(tag: str = "") -> None:
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[Memory{' ' + tag if tag else ''}] RAM={rss:.2f}GB | VRAM={vram:.2f}GB")
    else:
        print(f"[Memory{' ' + tag if tag else ''}] RAM={rss:.2f}GB")


# ─────────────────────────────────────────────────────────────────────────────
# 结果打印
# ─────────────────────────────────────────────────────────────────────────────

def print_final_results(results: dict, num_clients: int) -> None:
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")

    if results['round_accuracies']:
        print(f"Final global accuracy : {results['round_accuracies'][-1]:.2f}%")
        print(f"Best  global accuracy : {max(results['round_accuracies']):.2f}%")

    participated = [i for i in range(num_clients) if results['client_accuracies'][i]]
    if participated:
        accs  = [results['client_accuracies'][i][-1] for i in participated]
        comps = [results['client_compression_ratios'][i][-1] for i in participated]
        print(f"\nClients participated  : {len(participated)}/{num_clients}")
        print(f"Avg client accuracy   : {np.mean(accs):.2f}%")
        print(f"Avg compression ratio : {np.mean(comps):.3f}")

    print(f"\nSample Client Statistics (first 10 participated):")
    shown = 0
    for i in range(num_clients):
        if not results['client_accuracies'][i]:
            continue
        print(
            f"  Client {i:3d}: "
            f"Acc={results['client_accuracies'][i][-1]:.2f}%, "
            f"Params={results['client_param_counts'][i][-1]:,}, "
            f"Compression={results['client_compression_ratios'][i][-1]:.3f}"
        )
        shown += 1
        if shown >= 10:
            break


# ─────────────────────────────────────────────────────────────────────────────
# 模型结构摘要
# ─────────────────────────────────────────────────────────────────────────────
from models import build_model
def print_model_summary(
    model_name: str,
    dataset_name: str,
    capacity_distribution: list,
) -> None:
    """
    打印 S / M / L 三档模型的结构摘要与参数量对比。

    参数：
        model_name            : 'vgg16' | 'resnet18'
        dataset_name          : 'cifar10' | 'cifar100' | 'femnist' ...
        capacity_distribution : [(scale, ratio), ...]，按 scale 从大到小排列
    """
    # 以 scale=1.0（L）作为基准参数量
    ref_model = build_model(model_name, dataset_name, scale=1.0)
    ref_params = sum(p.numel() for p in ref_model.parameters())
    del ref_model

    # 标签映射（按 scale 大小打标签，不依赖字符串）
    sorted_tiers = sorted(capacity_distribution, reverse=True)
    tier_labels = ['L', 'M', 'S'] + [f'T{i}' for i in range(4, len(sorted_tiers) + 1)]

    sep = '─' * 68
    print(f"\n{sep}")
    print(f"  模型结构摘要  |  {model_name.upper()}  ×  {dataset_name.upper()}")
    print(sep)
    print(f"  {'档位':<6} {'scale':>7}  {'客户端占比':>10}  {'参数量':>14}  {'相对 L':>8}")
    print(sep)

    for label, (scale, ratio) in zip(tier_labels, sorted_tiers):
        m = build_model(model_name, dataset_name, scale=scale)
        total = sum(p.numel() for p in m.parameters())
        del m
        rel = total / ref_params
        print(
            f"  {label:<6} {scale:>7.2f}  {ratio * 100:>9.1f}%  "
            f"{total:>14,}  {rel:>7.2%}"
        )

    print(sep)
    print(f"  基准 (L, scale=1.0): {ref_params:,} 参数")
    print(f"  track_running_stats = False  （所有 BN 层）")
    print(f"{sep}\n")



# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # =========================================================================
    # ESPFL 实验配置
    # =========================================================================
    # ── 数据集与模型 ──────────────────────────────────────────────────────────
    DATASET_NAME  = 'cifar10'   # 'cifar10' | 'cifar100' | 'femnist'
    MODEL_NAME    = 'vgg16'     # 'vgg16'   | 'resnet18'
    DATA_ROOT     = r'E:\awzm\federated_learning\AdaptiveFL-master\data'

    # ── 联邦学习设置 ──────────────────────────────────────────────────────────
    NUM_CLIENTS               = 100
    NUM_ROUNDS                = 1000
    CLIENT_SELECTION_FRACTION = 0.1    # 每轮选取 10% 客户端0.1 
    LOCAL_EPOCHS              = 5
    BATCH_SIZE                = 50
    LOCAL_TRAIN_RATIO         = 0.9
    # 本地验证集比例（相对客户端分片）：train/val/test ≈ 80/10/10。
    # 验证集仅用于容量决策，测试集仅用于评估；VAL_RATIO 必须大于 0。
    VAL_RATIO                 = 0.1
    # True：本地 val/test 使用无增强 transform（修复旧版本地测试集误用训练增强）
    FIX_EVAL_TRANSFORM        = True

    # ── 数据分布 ──────────────────────────────────────────────────────────────
    IID          = False
    NONIID_ALPHA = 0.3    # Dirichlet α，越小越非 IID   0.6   0.3

    # ── 其他 ──────────────────────────────────────────────────────────────────
    DATA_AUGMENT = True
    SEED         = 1

    # ── Probe 子集评估配置 ────────────────────────────────────────────────────
    N_PROBE        = 20          # 固定评估的客户端数
    PROBE_RATIO    = (0.30, 0.30, 0.40)   # S:M:L 采样比例（3:3:4）
    PROBE_SEED     = 42          # 固定种子，便于跨实验比较
    EVAL_INTERVAL  = 15          # 每隔多少轮触发一次 probe 评估

    # =========================================================================
    # ESPFL 超参数
    # =========================================================================

    # ── ESPFL（最终架构，2026-11-06）：详细超参见 espfl_method.ESPFLConfig ─────
    #     预实验 / 消融建议用 run_espfl.py 的 --preset
    ESPFL_CFG = ESPFLConfig(
        T_part=15, S_total=5, local_epochs=LOCAL_EPOCHS, seed=SEED,
        controller=ControllerConfig(
            mode='evidence', eps=0.05, eta=0.5, lam_max=0.2, delta_min=0.03,
            phi=0.25, k_min=4, rho_min=0.10, layer_coef='uniform',
            noniid_alpha=None if IID else NONIID_ALPHA),
        optimizer='sgd',          # momentum=0.9, wd=1e-4，学习率由 trainer 提供
        aki_mode='full', aggregation='sppa', composition='sed',
    )

    # =========================================================================
    # 日志 & 设备
    # =========================================================================

    alpha_tag   = 0.0 if IID else NONIID_ALPHA
    save_log    = f'./{DATASET_NAME}_train/alpha_{alpha_tag:.2f}/client_{NUM_CLIENTS}_{MODEL_NAME}_espfl'
    timestamp   = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    setup_logging(save_log, timestamp)

    RESULT_PATH = os.path.join(
        save_log,
        f"{timestamp}_espfl_{DATASET_NAME}_{MODEL_NAME}_alpha{alpha_tag:.2f}_r{NUM_ROUNDS}.pkl"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import random as _random
    _random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    print(f"[Config] method=espfl, device={device}, "
          f"dataset={DATASET_NAME}, model={MODEL_NAME}")
    print(f"[Config] clients={NUM_CLIENTS}, rounds={NUM_ROUNDS}, "
          f"local_epochs={LOCAL_EPOCHS}, iid={IID}, alpha={NONIID_ALPHA}")

    print_memory("startup")

    # =========================================================================
    # 数据加载
    # =========================================================================

    train_loaders, val_loaders, test_loaders, global_testloader = build_federated_dataloaders(
        num_clients       = NUM_CLIENTS,
        dataset_name      = DATASET_NAME,
        data_root         = DATA_ROOT,
        iid               = IID,
        noniid_alpha      = NONIID_ALPHA,
        batch_size        = BATCH_SIZE,
        local_train_ratio = LOCAL_TRAIN_RATIO,
        augment_train     = DATA_AUGMENT,
        plot_distribution = False,
        seed              = SEED,
        val_ratio         = VAL_RATIO,
        fix_eval_transform= FIX_EVAL_TRANSFORM,
        return_val        = True,
    )
    print_memory("after data loading")

    # =========================================================================
    # 初始化 ESPFL、打印模型摘要并选取固定评估客户端
    # =========================================================================

    method = ESPFLMethod(
        num_clients=NUM_CLIENTS,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        device=device,
        cfg=ESPFL_CFG,
        val_loaders=val_loaders,
    )
    print_model_summary(MODEL_NAME, DATASET_NAME, method.capacity_distribution)

    probe_ids = select_probe_clients(
        client_scales=method.client_scales,
        n_probe=N_PROBE,
        ratio_sml=PROBE_RATIO,
        seed=PROBE_SEED,
    )
    print_probe_summary(probe_ids, method.client_scales)

    # =========================================================================
    # 联邦训练骨架
    # =========================================================================

    trainer = FederatedTrainer(
        method                    = method,
        num_clients               = NUM_CLIENTS,
        num_rounds                = NUM_ROUNDS,
        client_selection_fraction = CLIENT_SELECTION_FRACTION,
        device                    = device,
        seed                      = SEED,
        probe_client_ids          = probe_ids,
        eval_interval             = EVAL_INTERVAL,
    )

    # =========================================================================
    # 训练
    # =========================================================================

    print_memory("before training")
    results = trainer.run(
        train_loaders     = train_loaders,
        test_loaders      = test_loaders,
        global_testloader = global_testloader,
    )
    print_memory("after training")

    # =========================================================================
    # 保存 & 打印
    # =========================================================================

    with open(RESULT_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[Results] Saved to {RESULT_PATH}")

    print_final_results(results, NUM_CLIENTS)

    # ── 最终 Local 指标：全部 N 个客户端的个性化模型在本地测试集上的平均（论文 Eq. 34）──
    final_local = method.evaluate_local_clients(list(range(NUM_CLIENTS)), test_loaders, device)
    _vals = [v for v in final_local.values() if v == v]
    print(f"\n[FinalLocal] mean over {len(_vals)} clients = {np.mean(_vals):.2f}%")
    with open(os.path.join(save_log, f"{timestamp}_final_local_espfl.json"), 'w') as f:
        json.dump({str(k): v for k, v in final_local.items()}, f, indent=2)
    with open(os.path.join(save_log, f"{timestamp}_structure_espfl.json"), 'w') as f:
        json.dump(method.structure_report(), f, default=str)


    # ── 额外保存 probe 评估结果为可读 JSON ────────────────────────────
    probe_data = results['probe_local_eval']
    if probe_data['rounds']:
        probe_save = {
            'method':       'espfl',
            'dataset':      DATASET_NAME,
            'model':        MODEL_NAME,
            'eval_interval': EVAL_INTERVAL,
            'n_probe':      N_PROBE,
            'probe_ids':    probe_ids,
            'rounds':       probe_data['rounds'],
            'avg':          probe_data['avg'],
            'per_client':   {str(cid): accs for cid, accs in probe_data['per_client'].items()},
        }
        probe_save_path = os.path.join(
            save_log, f"{timestamp}_probe_eval_espfl.json"
        )
        os.makedirs(save_log, exist_ok=True)
        with open(probe_save_path, 'w') as f:
            json.dump(probe_save, f, indent=2)
        print(f"[ProbeEval] Saved to {probe_save_path}")


if __name__ == "__main__":
    main()
