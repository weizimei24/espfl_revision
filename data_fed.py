"""
Federated data utilities
========================
支持的数据集：
  - CIFAR-10
  - CIFAR-100
  - MNIST
  - FEMNIST  (依赖 LEAF 项目预处理，见下方说明)

数据划分策略：
  - IID:     按类别分层分配（每个类别均匀分到各客户端）
  - Non-IID: Dirichlet 分布划分（alpha 越小越异构）
  - FEMNIST: 天然按用户划分，直接使用 LEAF 输出的 per-user json

本地 train/test 切分：
  - 所有数据集：随机 9:1（非分层），可通过 seed 复现

FEMNIST 数据准备方法：
  1. git clone https://github.com/TalwalkarLab/leaf
  2. cd leaf/data/femnist
  3. ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
     （--sf 控制采样比例，可按需调整）
  4. 生成的 data/train/*.json 和 data/test/*.json 放入：
     <data_root>/femnist/data/train/ 和 <data_root>/femnist/data/test/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# 数据集全局配置
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "num_classes": 100,
        "in_channels": 3,
        "input_size": 32,
        "mean": [0.5071, 0.4865, 0.4409],
        "std":  [0.2673, 0.2564, 0.2762],
    },
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "mean": [0.1307],
        "std":  [0.3081],
    },
    "femnist": {
        "num_classes": 62,   # 10数字 + 26大写 + 26小写
        "in_channels": 1,
        "input_size": 28,
        "mean": [0.9641],
        "std":  [0.1592],
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    """获取数据集配置，不存在时抛出友好错误。"""
    name = dataset_name.lower()
    if name not in DATASET_CONFIGS:
        raise NotImplementedError(
            f"不支持的数据集: '{dataset_name}'。"
            f"当前支持: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[name]


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具类
# ─────────────────────────────────────────────────────────────────────────────

class DatasetSplit(Dataset):
    """按索引切分数据集的包装器（适用于 CIFAR/MNIST 等 torchvision 数据集）。"""

    def __init__(self, dataset: Dataset, idxs: List[int]):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int):
        x, y = self.dataset[self.idxs[item]]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# FEMNIST 专用 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FemnistUserDataset(Dataset):
    """
    单个 FEMNIST 用户的数据集。
    LEAF 的 json 格式：{'x': [[pixel...], ...], 'y': [label, ...]}
    每个 x 样本是长度为 784 的归一化像素向量（0~1）。
    """

    def __init__(self, user_data: dict, transform=None):
        # x: list of list[float], shape (N, 784)
        # y: list of int
        self.x = user_data["x"]
        self.y = user_data["y"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        # 重建为 (1, 28, 28) 的 Tensor
        img = torch.tensor(self.x[idx], dtype=torch.float32).view(1, 28, 28)
        label = int(self.y[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# IID / Non-IID 划分（适用于 CIFAR / MNIST）
# ─────────────────────────────────────────────────────────────────────────────

def iid_stratified_partition(
    dataset,
    num_users: int,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    严格 IID 划分：对每个类别，将其索引均匀分配给各客户端。
    适用于任何含 `.targets` 属性的 torchvision 数据集。
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    dict_users: Dict[int, List[int]] = {i: [] for i in range(num_users)}

    for c in range(num_classes):
        idxs_c = np.where(labels == c)[0].copy()
        rng.shuffle(idxs_c)

        base = len(idxs_c) // num_users
        rem  = len(idxs_c) % num_users

        start = 0
        for u in range(num_users):
            take = base + (1 if u < rem else 0)
            if take > 0:
                dict_users[u].extend(idxs_c[start : start + take].tolist())
            start += take

    for u in range(num_users):
        rng.shuffle(dict_users[u])

    return dict_users


def noniid_dirichlet_partition(
    dataset,
    num_users: int,
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    Dirichlet Non-IID 划分：alpha 越小异构程度越高。
    适用于任何含 `.targets` 属性的 torchvision 数据集。
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    class_idxs = [np.where(labels == k)[0] for k in range(num_classes)]

    for k in range(num_classes):
        idxs_k = class_idxs[k]
        rng.shuffle(idxs_k)

        proportions = rng.dirichlet([alpha] * num_users)
        proportions = proportions / proportions.sum()

        splits = (proportions * len(idxs_k)).astype(int)

        diff = len(idxs_k) - splits.sum()
        for i in range(abs(diff)):
            splits[i % num_users] += int(np.sign(diff))

        idx_start = 0
        for i in range(num_users):
            dict_users[i] = np.concatenate((dict_users[i], idxs_k[idx_start:idx_start + splits[i]]), axis=0)
            idx_start += splits[i]

    for i in range(num_users):
        rng.shuffle(dict_users[i])
        dict_users[i] = dict_users[i].tolist()

    return dict_users


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR / MNIST：加载数据集 + 划分
# ─────────────────────────────────────────────────────────────────────────────

def _build_transforms(
    cfg: dict,
    augment_train: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    根据数据集配置构建训练/测试 transform。
    CIFAR 系列默认加入随机裁剪和水平翻转，MNIST/FEMNIST 不加。
    """
    mean, std = cfg["mean"], cfg["std"]
    input_size = cfg["input_size"]
    in_channels = cfg["in_channels"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if augment_train and in_channels == 3 and input_size == 32:
        # CIFAR 风格数据增强
        trans_train = transforms.Compose([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return trans_train, trans_test


def _get_torchvision_dataset(
    dataset_name: str,
    data_root: str,
    augment_train: bool = True,
):
    """加载 torchvision 内置数据集（CIFAR-10/100, MNIST）。"""
    cfg = get_dataset_config(dataset_name)
    trans_train, trans_test = _build_transforms(cfg, augment_train=augment_train)

    DS_MAP = {
        "cifar10":  torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
        "mnist":    torchvision.datasets.MNIST,
    }
    DS = DS_MAP[dataset_name.lower()]

    dataset_train = DS(data_root, train=True,  download=True, transform=trans_train)
    dataset_test  = DS(data_root, train=False, download=True, transform=trans_test)

    return dataset_train, dataset_test


def _get_torchvision_train_eval_view(dataset_name: str, data_root: str):
    """训练集的「无增强」视图（test transform），用于本地验证集 / 本地测试集评估。"""
    cfg = get_dataset_config(dataset_name)
    _, trans_test = _build_transforms(cfg, augment_train=False)
    DS = {
        "cifar10":  torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
        "mnist":    torchvision.datasets.MNIST,
    }[dataset_name.lower()]
    return DS(data_root, train=True, download=True, transform=trans_test)


def get_partitioned_dataset(
    dataset_name: str,
    data_root: str,
    num_users: int,
    iid: bool = False,
    noniid_alpha: float = 0.5,
    augment_train: bool = True,
    seed: Optional[int] = None,
):
    """
    加载 torchvision 数据集并返回联邦划分字典。

    返回：
        dataset_train, dataset_test, dict_users
        其中 dict_users: {client_id: [sample_indices, ...]}
    """
    dataset_train, dataset_test = _get_torchvision_dataset(
        dataset_name, data_root, augment_train=augment_train
    )

    if iid:
        dict_users = iid_stratified_partition(dataset_train, num_users, seed=seed)
    else:
        dict_users = noniid_dirichlet_partition(
            dataset_train, num_users, alpha=noniid_alpha, seed=seed
        )

    return dataset_train, dataset_test, dict_users


# ─────────────────────────────────────────────────────────────────────────────
# FEMNIST：加载 LEAF 格式数据
# ─────────────────────────────────────────────────────────────────────────────

def _load_leaf_json_files(json_dir: Path) -> Dict[str, dict]:
    """
    读取一个目录下所有 LEAF json 文件，返回 {user_id: {'x': ..., 'y': ...}}。
    LEAF json 结构：
        {
          "users": ["f0000_14", ...],
          "num_samples": [N, ...],
          "user_data": {"f0000_14": {"x": [[...], ...], "y": [...]}, ...}
        }
    """
    user_data: Dict[str, dict] = {}
    json_files = sorted(json_dir.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"在 {json_dir} 下没有找到任何 json 文件。\n"
            "请先按照文件顶部注释中的步骤使用 LEAF 预处理 FEMNIST 数据。"
        )

    for f in json_files:
        with open(f, "r") as fp:
            d = json.load(fp)
        for user, data in d["user_data"].items():
            user_data[user] = data

    return user_data


def _get_femnist_loaders(
    data_root: str,
    num_clients: int,
    batch_size: int = 32,
    local_train_ratio: float = 0.9,
    seed: Optional[int] = None,
    num_workers: int = 0,
    val_ratio: float = 0.0,
):
    """
    从 LEAF 预处理好的 FEMNIST json 构建联邦 DataLoader。
    返回 (train_loaders, val_loaders or None, test_loaders, global_testloader)。
    验证集从该用户的训练数据中按 val_ratio 划出（固定种子）。

    目录结构期望：
        <data_root>/femnist/data/train/*.json
        <data_root>/femnist/data/test/*.json

    返回：
        train_loaders, test_loaders, global_testloader
        global_testloader 是所有客户端 test 数据合并后的 DataLoader。
    """
    cfg = get_dataset_config("femnist")
    # FEMNIST 像素值已归一化到 [0,1]，这里只做通道级标准化
    transform = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])

    femnist_root = Path(data_root) / "femnist" / "data"
    train_user_data = _load_leaf_json_files(femnist_root / "train")
    test_user_data  = _load_leaf_json_files(femnist_root / "test")

    all_users = sorted(train_user_data.keys())
    rng = np.random.default_rng(seed)

    # 按需采样 num_clients 个用户
    if len(all_users) < num_clients:
        print(
            f"[Warning] FEMNIST 中只有 {len(all_users)} 个用户，"
            f"但请求了 {num_clients} 个客户端，将使用全部用户。"
        )
        num_clients = len(all_users)
        selected_users = all_users
    else:
        selected_users = rng.choice(all_users, num_clients, replace=False).tolist()

    train_loaders: Dict[int, DataLoader] = {}
    val_loaders:   Dict[int, DataLoader] = {}
    test_loaders:  Dict[int, DataLoader] = {}
    all_test_x, all_test_y = [], []

    for cid, user in enumerate(selected_users):
        # ── 训练集 ──────────────────────────────────────────────
        train_ds = FemnistUserDataset(train_user_data[user], transform=transform)

        # 如果该用户有独立的 test json 则使用，否则从 train 数据切分
        if user in test_user_data:
            test_ds = FemnistUserDataset(test_user_data[user], transform=transform)
        else:
            # 从训练数据随机切分 local_train_ratio
            n = len(train_ds)
            indices = np.arange(n)
            client_rng = np.random.default_rng(
                None if seed is None else (seed + cid)
            )
            client_rng.shuffle(indices)
            split_idx = max(1, min(int(n * local_train_ratio), n - 1))
            train_ds = FemnistUserDataset(
                {
                    "x": [train_user_data[user]["x"][i] for i in indices[:split_idx]],
                    "y": [train_user_data[user]["y"][i] for i in indices[:split_idx]],
                },
                transform=transform,
            )
            test_ds = FemnistUserDataset(
                {
                    "x": [train_user_data[user]["x"][i] for i in indices[split_idx:]],
                    "y": [train_user_data[user]["y"][i] for i in indices[split_idx:]],
                },
                transform=transform,
            )

        if val_ratio > 0 and len(train_ds) >= 3:
            n_all = len(train_ds) + len(test_ds)
            n_val = max(1, min(int(round(n_all * val_ratio)), len(train_ds) - 1))
            perm = np.random.default_rng(
                None if seed is None else (seed + 10_000 + cid)
            ).permutation(len(train_ds))
            src_x, src_y = train_ds.x, train_ds.y
            val_ds = FemnistUserDataset(
                {"x": [src_x[j] for j in perm[:n_val]], "y": [src_y[j] for j in perm[:n_val]]},
                transform=transform,
            )
            train_ds = FemnistUserDataset(
                {"x": [src_x[j] for j in perm[n_val:]], "y": [src_y[j] for j in perm[n_val:]]},
                transform=transform,
            )
            val_loaders[cid] = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        train_loaders[cid] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loaders[cid] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 收集 global test 数据
        if user in test_user_data:
            all_test_x.extend(test_user_data[user]["x"])
            all_test_y.extend(test_user_data[user]["y"])

    # 全局测试集
    if all_test_x:
        global_test_ds = FemnistUserDataset(
            {"x": all_test_x, "y": all_test_y}, transform=transform
        )
        global_testloader = DataLoader(
            global_test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        global_testloader = None
        print("[Warning] 未找到 FEMNIST 独立 test json，global_testloader 为 None。")

    return train_loaders, (val_loaders or None), test_loaders, global_testloader


def _three_way_split(indices: List[int], local_train_ratio: float, val_ratio: float):
    """
    在「已按客户端种子打乱」的索引上做 train / val / test 三分。

    关键约束：test 部分与旧版 9:1 切分逐样本一致（仍是打乱后的最后 1-local_train_ratio），
    验证集从原训练部分的尾部划出。因此：
        - 所有方法的本地测试集与旧实验完全相同；
        - val_ratio=0 时完全退化为旧切分。
    val_ratio 以客户端整个数据分片为基数（0.1 → 约 80/10/10）。
    """
    n = len(indices)
    split_idx = max(1, min(int(n * local_train_ratio), n - 1))
    train_part, test_idx = indices[:split_idx], indices[split_idx:]
    n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
    if val_ratio > 0:
        n_val = max(1, min(n_val, len(train_part) - 1))
    if n_val > 0:
        return train_part[:-n_val], train_part[-n_val:], test_idx
    return train_part, [], test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 主入口：统一构建联邦 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_federated_dataloaders(
    num_clients: int = 100,
    dataset_name: str = "cifar10",
    data_root: str = "./data",       
    iid: bool = False,
    noniid_alpha: float = 0.6,
    batch_size: int = 128,
    local_train_ratio: float = 0.9,
    augment_train: bool = True,         # 是否对训练集做数据增强（CIFAR 默认开启）
    plot_distribution: bool = False,
    seed: Optional[int] = None,
    num_workers: int = 0,
    val_ratio: float = 0.0,
    fix_eval_transform: bool = True,
    return_val: bool = False,
):
    """
    统一的联邦学习数据加载入口。

    新增参数（ESPFL 返修）：
        val_ratio          : 本地验证集占客户端分片的比例（0 = 旧版 9:1 切分）。
                             验证集只服务于 ESPFL 的容量决策，不参与本地训练。
        fix_eval_transform : True（默认）→ 本地 val/test 使用无增强的 test transform。
                             旧版本地测试集误用了训练增强（RandomCrop + Flip），评估带随机性。
                             False 仅用于复现旧日志。
        return_val         : True → 返回 (train, val, test, global)；False → 旧版三元组。

    参数：
        num_clients        : 客户端数量
        dataset_name       : 数据集名称，支持 cifar10 / cifar100 / mnist / femnist
        data_root          : 数据根目录（默认 './data'）
                             FEMNIST 需要 <data_root>/femnist/data/{train,test}/*.json
        iid                : True = IID 分层划分；False = Dirichlet Non-IID 划分
                             （FEMNIST 忽略此参数，天然 Non-IID）
        noniid_alpha       : Dirichlet alpha，越小越 Non-IID（仅 Non-IID 时生效）
        batch_size         : DataLoader batch size
        local_train_ratio  : 本地 train/test 切分比例（默认 0.9）
        augment_train      : CIFAR 类数据是否加数据增强（随机裁剪 + 水平翻转）
        plot_distribution  : 是否绘制数据分布图（需要 plot_figure.py）
        seed               : 随机种子，None 表示不固定
        num_workers        : DataLoader num_workers

    返回：
        train_loaders      : {client_id: DataLoader}
        test_loaders       : {client_id: DataLoader}
        global_testloader  : DataLoader（全局测试集）或 None（FEMNIST 无独立 test 时）
    """
    name = dataset_name.lower()

    # ── FEMNIST 走独立分支 ──────────────────────────────────────
    if name == "femnist":
        tr, va, te, gl = _get_femnist_loaders(
            data_root=data_root,
            num_clients=num_clients,
            batch_size=batch_size,
            local_train_ratio=local_train_ratio,
            seed=seed,
            num_workers=num_workers,
            val_ratio=val_ratio,
        )
        return (tr, va, te, gl) if return_val else (tr, te, gl)

    # ── CIFAR / MNIST 通用流程 ──────────────────────────────────
    dataset_train, dataset_test, dict_users = get_partitioned_dataset(
        dataset_name=name,
        data_root=data_root,
        num_users=num_clients,
        iid=iid,
        noniid_alpha=noniid_alpha,
        augment_train=augment_train,
        seed=seed,
    )

    if plot_distribution:
        try:
            from plot_figure import plot_client_distribution
            plot_client_distribution(dataset_train, dict_users)
        except Exception as e:
            print(f"[Warning] plot_client_distribution 失败: {e}")

    # 全局测试集
    global_testloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 本地 val / test 的数据源：默认使用无增强视图（修复旧版评估带增强的问题）
    dataset_eval = (
        _get_torchvision_train_eval_view(name, data_root) if fix_eval_transform else dataset_train
    )

    train_loaders: Dict[int, DataLoader] = {}
    val_loaders:   Dict[int, DataLoader] = {}
    test_loaders:  Dict[int, DataLoader] = {}

    for client_id in range(num_clients):
        indices = list(dict_users[client_id])

        # 与旧版相同的客户端种子与打乱方式 → 本地测试集逐样本一致
        client_rng = np.random.default_rng(
            None if seed is None else (seed + client_id)
        )
        client_rng.shuffle(indices)

        train_indices, val_indices, test_indices = _three_way_split(
            indices, local_train_ratio, val_ratio
        )

        train_loaders[client_id] = DataLoader(
            DatasetSplit(dataset_train, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loaders[client_id] = DataLoader(
            DatasetSplit(dataset_eval, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_indices:
            # 固定集合 + 不打乱 + 无增强 → 确定性验证评估
            val_loaders[client_id] = DataLoader(
                DatasetSplit(dataset_eval, val_indices),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

    if return_val:
        return train_loaders, (val_loaders or None), test_loaders, global_testloader
    return train_loaders, test_loaders, global_testloader


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：获取数据集信息（供外部模块查询 num_classes / input_size 等）
# ─────────────────────────────────────────────────────────────────────────────

def get_num_classes(dataset_name: str) -> int:
    return get_dataset_config(dataset_name)["num_classes"]


def get_input_channels(dataset_name: str) -> int:
    return get_dataset_config(dataset_name)["in_channels"]


def get_input_size(dataset_name: str) -> int:
    return get_dataset_config(dataset_name)["input_size"]


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    """返回 (in_channels, height, width)"""
    cfg = get_dataset_config(dataset_name)
    return (cfg["in_channels"], cfg["input_size"], cfg["input_size"])