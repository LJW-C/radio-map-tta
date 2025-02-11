from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import math
import os
from torch.optim import SGD
from torch.optim import Adam
__all__ = ["setup"]

def pad_or_truncate(tensor, target_size):
    # 如果张量大小小于目标大小，则进行填充
    if tensor.numel() < target_size:
        padding = torch.zeros(target_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding])
    # 如果张量大小大于目标大小，则进行截断
    else:
        return tensor[:target_size]

class MemoryBank:
    def __init__(self, K):
        self.K = K  # 记忆库的最大容量
        self.memory = deque(maxlen=K)  # 使用双端队列实现先进先出

    def add(self, q, v):
        # 将特征和预测结果添加到记忆库中
        for i in range(q.size(0)):
                q_i = q[i].detach().clone().view(-1).unsqueeze(0)
                v_i = v[i].detach().clone().view(-1).unsqueeze(0)
                self.memory.append((q_i, v_i))

    def retrieve(self, q, D):
        # for idx, item in enumerate(self.memory):
            # print(f"Item {idx} shape: {item[0].view(-1).shape}")

        # 检索与输入特征 q 最相似的 D 个样本
        if len(self.memory) == 0:
            return []

        # 展平特征向量
        batch_size = q.size(0)

        keys = torch.stack([item[0] for item in self.memory]).mean(0)
        q = q.view(batch_size, -1)
        distances = F.cosine_similarity(keys, q, dim=1)
        D = min(D, distances.shape[0])
        nearest_indices = distances.topk(D, largest=False)[1]
        support_set = []
        for i in nearest_indices:
            data = self.memory[i][0]
            support_set.append(
                data
            )
        return support_set

class RMEGANTTA(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.steps = 1
        self.episodic = False
        self.model, self.optimizer = prepare_model_and_optimizer(model)
        self._his = [0]
        self.margin_e = 0
        self.base_lr = 0.00002
        self.memory_bank = MemoryBank(K=100)  # 你可以根据需要调整 K 的大小
        self.dist_list = []
        self.adjusted_lr_list = []
        self.retrieval_size = 5
    # 1
    def forward(self, inputs, target):

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(inputs, target)
        return outputs



    @torch.enable_grad()
    def rmse(self, x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

    @torch.enable_grad()
    def nmse(self, x, y):
        mse = torch.mean((x - y) ** 2)
        mse_reference = torch.mean((y - 0) ** 2)
        epsilon = 1e-8
        if mse_reference.item() < epsilon:
            mse_reference = epsilon
        nmse = mse / mse_reference
        return nmse


    def reset(self):
        if self.model_state is None:
            raise Exception("cannot reset without saved model/optimizer state")

        load_model_and_optimizer(self.model, self.optimizer ,self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, inputs, target):

        self.optimizer.zero_grad()

        feats, outputs = self.model(inputs, return_features=True)

        rmse_loss = self.rmse(outputs, target)
        nmse_loss = self.nmse(outputs, target)
        loss = (rmse_loss + nmse_loss) / 1

        print(f"RMSE Loss: {rmse_loss.item()}, NMSE Loss: {nmse_loss.item()}, Total Loss: {loss.item()}")

        if not (torch.isnan(feats).any() or torch.isinf(feats).any() or
                torch.isnan(outputs).any() or torch.isinf(outputs).any()):
            self.memory_bank.add(feats, outputs)
        else:
            print("Features or outputs contain nan or inf values. Skipping adding to memory bank.")
        # ====================================
        support_set = self.memory_bank.retrieve(feats, D=self.retrieval_size)
        centers = torch.stack([feats for feats in support_set]).mean(0).to(outputs.device)
        centers = F.normalize(centers)
        feats = F.normalize(feats).mean(0)
        dist = 1 - F.cosine_similarity(feats.view(-1).unsqueeze(0), centers, dim=1)
        dist = dist.mean()
        self.dist_list.append(dist.item())
        tau = 0.01
        weight_lr = torch.exp(-dist * tau)
        if len(support_set) == 0:
            adjusted_lr = self.base_lr
        else:
            adjusted_lr = self.base_lr * weight_lr

        self.adjusted_lr_list.append(adjusted_lr.item())

        # print('adjusted_lr', adjusted_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr.item()
            # print(param_group['lr'])
        loss.backward()

        self.optimizer.step()

        print(f"Adjusted Learning Rate: {adjusted_lr}")
        return outputs


    def save(self, ): pass
    def print(self, ): pass


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.Conv2d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
            m.requires_grad_(True)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
        elif isinstance(m, nn.Conv2d):
            m.train()
            m.requires_grad_(True)
    return model


def prepare_model_and_optimizer(model):

    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = optim.Adam(
        params,
        lr=float(0.000002),
        betas=(0.9, 0.999),
        weight_decay=float(0.)
    )
    return model, optimizer


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

def setup(model):
    # model = configure_model(model)
    TTA_model = RMEGANTTA(
        model
    )
    return TTA_model

