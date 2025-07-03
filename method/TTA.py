import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque
import logging

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

class TTA(nn.Module):
    def __init__(self, model, base_lr=0.000005, tau=0.5, lambda_anchor=0.1):
        super().__init__()
        self.steps = 1
        self.model, self.optimizer = prepare_model_and_optimizer(model, base_lr)
        self.frozen_model = copy.deepcopy(model).eval()
        self.memory_bank = MemoryBank(K=100)
        self.base_lr = base_lr
        self.tau = tau
        self.lambda_anchor = lambda_anchor
        self.retrieval_size = 5
        self.dist_min = float('inf')
        self.dist_max = float('-inf')
        self.epsilon = 1e-8
        self.initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def forward(self, antenna, builds, target=None):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(antenna, builds, target)
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

    @torch.enable_grad()
    def forward_and_adapt(self, antenna, builds, target=None):
        self.optimizer.zero_grad()
        with torch.no_grad():
            feats, outputs_pre = self.frozen_model(antenna, builds, return_features=True)
        feats, outputs = self.model(antenna, builds, return_features=True)

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

        support_set = self.memory_bank.retrieve(feats, D=self.retrieval_size)
        if support_set:
            centers = torch.stack([feats for feats in support_set]).mean(0).to(outputs.device)
            centers = F.normalize(centers, dim=-1)
            feats = F.normalize(feats, dim=-1).mean(0)
            dist = 1 - F.cosine_similarity(feats.view(-1).unsqueeze(0), centers, dim=1)
            dist = dist.mean()
            self.dist_min = min(self.dist_min, dist.item())
            self.dist_max = max(self.dist_max, dist.item())
            dist_normalized = (dist - self.dist_min) / (self.dist_max - self.dist_min + self.epsilon)
            weight_lr = torch.exp(self.tau * dist_normalized)
            adjusted_lr = self.base_lr * weight_lr
        else:
            adjusted_lr = self.base_lr

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

def prepare_model_and_optimizer(model, base_lr):
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=float(base_lr), betas=(0.9, 0.999), weight_decay=0.0)
    return model, optimizer

def setup(model):
    tta_model = TTA(model)
    return tta_model
