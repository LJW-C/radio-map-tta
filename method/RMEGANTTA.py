from copy import deepcopy
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
        
        with torch.no_grad():
            keys = torch.cat([item[0] for item in self.memory], dim=0)
            q_mean = q.view(batch_size, -1).mean(dim=0, keepdim=True)
            
            distances = F.cosine_similarity(keys, q_mean, dim=1)
            D = min(D, distances.shape[0])
            
            nearest_indices = distances.topk(D, largest=True)[1]
            support_set = [self.memory[i][0] for i in nearest_indices]
            
        return support_set

class RMEGANOA(nn.Module):
    def __init__(self, model, base_lr=0.00002, tau=0.5, lambda_anchor=0.1):
        super().__init__()
        self.steps = 1
        self.episodic = False
        
        self.model, self.optimizer = prepare_model_and_optimizer(model)
        
        self.frozen_model = deepcopy(model).eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        self._his = [0]
        self.margin_e = 0
        self.base_lr = base_lr
        self.lambda_anchor = lambda_anchor
        self.tau = tau
        
        self.memory_bank = MemoryBank(K=20)
        self.dist_list = []
        self.adjusted_lr_list = []
        self.retrieval_size = 5
        self.epsilon = 1e-8
        
        self.dist_min = 0.0
        self.dist_max = 1.0
        self.first_pass = True

        self.initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def forward(self, inputs, target=None):
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
        if self.model_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, inputs, target=None):
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            _, outputs_pre = self.frozen_model(inputs, return_features=True)
            outputs_pre = outputs_pre.detach()

        feats, outputs = self.model(inputs, return_features=True)

        rmse_loss = self.rmse(outputs, outputs_pre)
        nmse_loss = self.nmse(outputs, outputs_pre)
        
        self_cons_loss = 1.0 * rmse_loss + 0.1 * nmse_loss

        anchor_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                anchor_loss += (param - self.initial_params[name]).pow(2).sum()
        
        loss = self_cons_loss + self.lambda_anchor * anchor_loss

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
            self.optimizer.zero_grad()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

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
    TTA_model = RMEGANOA(model)
    return TTA_model
