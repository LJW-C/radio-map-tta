from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from collections import deque
import numpy as np
import math
import os
import logging

__all__ = ["setup"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pad_or_truncate(tensor, target_size):
    if tensor.numel() < target_size:
        padding = torch.zeros(target_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding])
    else:
        return tensor[:target_size]

class MemoryBank:
    def __init__(self, K):
        self.K = K
        self.memory = deque(maxlen=K)

    def add(self, q, v):
        for i in range(q.size(0)):
            q_i = q[i].detach().clone().view(-1).unsqueeze(0)
            v_i = v[i].detach().clone().view(-1).unsqueeze(0)
            self.memory.append((q_i, v_i))

    def retrieve(self, q, D):
        if len(self.memory) == 0:
            return []
        batch_size = q.size(0)
        keys = torch.cat([item[0] for item in self.memory], dim=0)
        q = q.view(batch_size, -1)
        distances = F.cosine_similarity(keys, q, dim=1)
        D = min(D, distances.shape[0])
        nearest_indices = distances.topk(D, largest=True)[1]
        support_set = [self.memory[i][0] for i in nearest_indices]
        return support_set

class RMEGANTTA(nn.Module):
    def __init__(self, model, base_lr=0.00002, tau=0.01, lambda_anchor=0.1):
        super().__init__()
        self.steps = 1
        self.episodic = False
        self.model, self.optimizer = prepare_model_and_optimizer(model)
        self.frozen_model = deepcopy(model).eval()
        self._his = [0]
        self.margin_e = 0
        self.base_lr = base_lr
        self.memory_bank = MemoryBank(K=100)
        self.dist_list = []
        self.adjusted_lr_list = []
        self.retrieval_size = 5
        self.dist_min = float('inf')
        self.dist_max = float('-inf')
        self.epsilon = 1e-8
        self.initial_params = {name: param.clone().detach | param in model.named_parameters()}

    def forward(self, inputs, target=None):
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
        if mse_reference.item() < self.epsilon:
            mse_reference = torch.tensor(self.epsilon, device=mse.device)
        nmse = mse / mse_reference
        return nmse

    def reset(self):
        if self.model_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, inputs, target=None):
        self.optimizer.zero_grad()
        with torch.no_grad():
            feats, outputs_pre = self.frozen_model(inputs, return_features=True)
        feats, outputs = self.model(inputs, return_features=True)
        rmse_loss = self.rmse(outputs, outputs_pre.detach())
        nmse_loss = self.nmse(outputs, outputs_pre.detach())
        self_cons_loss = 0.5 * rmse_loss + 0.5 * nmse_loss
        anchor_loss = 0.0
        for name, param in self.model.named_parameters():
            anchor_loss += (param - self.initial_params[name]).pow(2).sum()
        loss = self_cons_loss + self.lambda_anchor * anchor_loss
        if not (torch.isnan(feats).any() or torch.isinf(feats).any() or
                torch.isnan(outputs).any() or torch.isinf(outputs).any()):
            self.memory_bank.add(feats, outputs)
        else:
            logger.warning("特征或输出包含无效值，跳过添加到记忆库")
        support_set = self.memory_bank.retrieve(feats, D=self.retrieval_size)
        if support_set:
            centers = torch.stack([feats for feats in support_set]).mean(0).to(outputs.device)
            centers = F.normalize(centers, dim=-1)
            feats = F.normalize(feats, dim=-1).mean(0)
            dist = 1 - F.cosine_similarity(feats.view(-1).unsqueeze(0), centers, dim=1)
            dist = dist.mean()
            self.dist_list.append(dist.item())
            self.dist_min = min(self.dist_min, dist.item())
            self.dist_max = max(self.dist_max, dist.item())
            dist_normalized = (dist - self.dist_min) / (self.dist_max - self.dist_min + self.epsilon)
            weight_lr = torch.exp(self.tau * dist_normalized)
            adjusted_lr = self.base_lr * weight_lr
        else:
            adjusted_lr = self.base_lr
        self.adjusted_lr_list.append(adjusted_lr.item())
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr.item()
        if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
            loss.backward()
            self.optimizer.step()
        else:
            logger.warning("跳过更新，因损失值无效")
        logger.info(f"RMSE 损失: {rmse_loss.item():.6f}, NMSE 损失: {nmse_loss.item():.6f}, 总损失: {loss.item():.6f}")
        logger.info(f"调整后学习率: {adjusted_lr.item():.6f}")
        return outputs

    def save(self):
        pass

    def print(self):
        pass

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
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
            m.train()
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
    optimizer = torch.optim.Adam(
        params,
        lr=float(0.00002),
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
    TTA_model = RMEGANTTA(model)
    return TTA_model
