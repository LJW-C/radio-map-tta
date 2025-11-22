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
import logging
from torch.optim import SGD
from torch.optim import Adam

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["setup"]

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
        # 【显存优化】务必 detach，否则显存会爆炸
        for i in range(q.size(0)):
            q_i = q[i].detach().clone().view(-1).unsqueeze(0)
            v_i = v[i].detach().clone().view(-1).unsqueeze(0)
            self.memory.append((q_i, v_i))

    def retrieve(self, q, D):
        if len(self.memory) == 0:
            return []

        batch_size = q.size(0)

        with torch.no_grad():
            # keys: (Memory_Size, Feature_Dim)
            keys = torch.cat([item[0] for item in self.memory], dim=0) 
            q_mean = q.view(batch_size, -1).mean(dim=0, keepdim=True)
            
            distances = F.cosine_similarity(keys, q_mean, dim=1)
            D = min(D, distances.shape[0])
            nearest_indices = distances.topk(D, largest=True)[1]
            support_set = [self.memory[i][0] for i in nearest_indices]
            
        return support_set

class OA_Supervised(nn.Module):

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.steps = 1
        
        # 1. 准备可训练的学生模型
        self.model, self.optimizer = prepare_model_and_optimizer(model, args)
        
        # 【移除】不再需要 frozen_model，因为我们直接用 target 计算 Loss
          
        # 2. 保存初始参数用于计算 Anchor Loss (依然保留，防止遗忘)
        self.initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        self.base_lr = 0.000008 
        self.memory_bank = MemoryBank(K=50) 

        self.dist_list = []
        self.adjusted_lr_list = []
        self.retrieval_size = 7

        self.dist_min = 0.0
        self.dist_max = 1.0
        self.first_pass = True
        self.tau = 0.5 
        self.lambda_anchor = 0.1 
        self.epsilon = 1e-8

    def forward(self, inputs, target):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(inputs, target)
        return outputs

    @torch.enable_grad()
    def rmse(self, x, y):
        return torch.sqrt(torch.mean((x - y) ** 2) + self.epsilon)

    @torch.enable_grad()
    def nmse(self, x, y):
        mse = torch.mean((x - y) ** 2)
        mse_reference = torch.mean((y - 0) ** 2)
        if mse_reference.item() < self.epsilon:
            mse_reference = torch.tensor(self.epsilon, device=mse.device)
        nmse = mse / mse_reference
        return nmse

    def reset(self):
        pass

    @torch.enable_grad()
    def forward_and_adapt(self, inputs, target):
        self.optimizer.zero_grad()

        # 1. 获取当前模型输出
        feats, outputs = self.model(inputs, return_features=True)

        # 2. 【关键修改】计算有监督 Loss (使用真实 target)
        rmse_loss = self.rmse(outputs, target)
        nmse_loss = self.nmse(outputs, target)
        
        # 在有监督场景下，Loss 就是真实的误差
        supervised_loss = (rmse_loss + nmse_loss) / 1.0 

        # 3. Anchor Loss (防止对当前 batch 过拟合导致遗忘旧知识)
        anchor_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                anchor_loss += (param - self.initial_params[name]).pow(2).sum()

        loss = supervised_loss + self.lambda_anchor * anchor_loss

        # 4. Memory Bank 与 动态学习率 (逻辑保持不变)
        # 即使有 target，动态学习率依然有用：
        # 当遇到分布差异大的样本（dist大）时，说明这是一个很难/很新的样本，我们可能希望增大 LR 快速学会它
        # 或者反过来，如果认为它是异常值，减小 LR (取决于你的策略，目前代码是增大 LR)
        if not (torch.isnan(feats).any() or torch.isinf(feats).any()):
            self.memory_bank.add(feats, outputs)

        adjusted_lr = self.base_lr
        support_set = self.memory_bank.retrieve(feats, D=self.retrieval_size)
        
        if support_set:
            with torch.no_grad():
                centers = torch.stack([f for f in support_set]).mean(0).to(outputs.device)
                centers = F.normalize(centers, dim=-1)
                feats_norm = F.normalize(feats.detach(), dim=-1).mean(0)
              
                dist = 1 - F.cosine_similarity(feats_norm.view(-1).unsqueeze(0), centers, dim=1)
                dist_val = dist.mean().item()
                self.dist_list.append(dist_val)
                if self.first_pass:
                    self.dist_min = dist_val
                    self.dist_max = dist_val + 1e-6
                    self.first_pass = False
                else:
                    self.dist_min = min(self.dist_min, dist_val)
                    self.dist_max = max(self.dist_max, dist_val)

                dist_normalized = (dist_val - self.dist_min) / (self.dist_max - self.dist_min + self.epsilon)
                
                weight_lr = math.exp(self.tau * dist_normalized)
                adjusted_lr = self.base_lr * weight_lr

        self.adjusted_lr_list.append(adjusted_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Loss is NaN/Inf. Skipping update.")
            self.optimizer.zero_grad()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return outputs

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

def prepare_model_and_optimizer(model, args):
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = optim.Adam(
        params,
        lr=float(0.000002),
        betas=(0.9, 0.999),
        weight_decay=float(0.)
    )
    return model, optimizer

def setup(model, args):
    # 这里改名为 OA_Supervised
    TTA_model = OA_Supervised(
        args,
        model
    )
    return TTA_model
