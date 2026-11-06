"""ESPFL 与联邦训练器之间的接口约定。"""
from abc import ABC, abstractmethod
import torch
from typing import Dict, List


class BaseMethod(ABC):

    @abstractmethod
    def distribute(self, selected_clients: list):
        """将全局模型分发到选中的客户端"""
        ...

    @abstractmethod
    def local_update(
        self,
        selected_clients: list,
        train_loaders: dict,
        test_loaders: dict,
        round_idx: int,
        lr: float,
    ) -> dict:
        """
        客户端本地训练。
        返回格式：
        {
            client_id: {
                'accuracy': float,
                'params': int,
                'compression': float
            },
            ...
        }
        """
        ...

    @abstractmethod
    def aggregate(self, selected_clients: list, train_loaders: dict) -> None:
        """服务器端聚合，结果更新内部 global_model"""
        ...

    @abstractmethod
    def get_global_model(self):
        """返回当前全局模型（CPU上）"""
        ...

    @abstractmethod
    def set_global_model(self, model) -> None:
        """更新全局模型"""
        ...

    def post_round_hook(
        self,
        selected_clients: list,
        train_loaders: dict,
        test_loaders: dict,
        round_idx: int,
    ) -> None:
        """每轮聚合后的可选操作，默认空实现，子类按需重写"""
        pass

    @abstractmethod
    def evaluate_local_clients(
        self,
        client_ids: List[int],
        test_loaders: dict,
        device: torch.device,
    ) -> Dict[int, float]:
        """评估 ESPFL 客户端当前子网，返回 {client_id: 准确率百分比}。"""
        ...
