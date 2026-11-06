"""ESPFL 联邦训练器使用的模型评估工具。"""

import torch
import torch.nn as nn


def test_model(model, testloader, device, return_loss=False):
    """
    在 testloader 上评估模型准确率。

    返回：
        accuracy: float，百分比形式（如 83.24）
    """
    model.eval()
    correct = 0
    total   = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total      += targets.size(0)
            correct    += predicted.eq(targets).sum().item()
            if return_loss:
                total_loss += criterion(outputs, targets).item()

    accuracy = 100.0 * correct / total
    if return_loss:
        return accuracy, total_loss / total
    return accuracy

