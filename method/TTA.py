import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque
import logging
import math
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["setup"]

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

class OA(nn.Module):
    def __init__(self, model, base_lr=0.000005, tau=0.5, lambda_anchor=0.1):
        super().__init__()
        self.steps = 1
        
        self.model, self.optimizer = prepare_model_and_optimizer(model, base_lr)
            
        self.initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        self.memory_bank = MemoryBank(K=20) 
        
        self.base_lr = base_lr
        self.tau = tau
        self.lambda_anchor = lambda_anchor
        self.retrieval_size = 5
        self.epsilon = 1e-8
        
        self.dist_min = 0.0
        self.dist_max = 1.0 
        self.first_pass = True

    def forward(self, antenna, builds, target=None):
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(antenna, builds, target)
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

    @torch.enable_grad()
    def forward_and_adapt(self, antenna, builds, target):
        self.optimizer.zero_grad()
        feats, outputs = self.model(antenna, builds, return_features=True)
        rmse_loss = self.rmse(outputs, target)
        nmse_loss = self.nmse(outputs, target)

        supervised_loss = 1.0 * rmse_loss + 0.1 * nmse_loss

        anchor_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                anchor_loss += (param - self.initial_params[name]).pow(2).sum()
        
        loss = supervised_loss + self.lambda_anchor * anchor_loss

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
        else:
            adjusted_lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Loss is NaN/Inf. Skipping step.")
            self.optimizer.zero_grad() 
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return outputs

def prepare_model_and_optimizer(model, base_lr):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
             m.track_running_stats = False 
             
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=float(base_lr), betas=(0.9, 0.999), weight_decay=0.0)
    return model, optimizer

def setup(model):
    tta_model = OA(model)
    return tta_model
