from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

__all__ = ["setup"]


def pad_or_truncate(tensor, target_size):
    """Pads or truncates a tensor to a target size."""
    if tensor.numel() < target_size:
        padding = torch.zeros(target_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding])
    else:
        return tensor[:target_size]


class MemoryBank:
    """
    A memory bank to store and retrieve similar feature-prediction pairs.
    """

    def __init__(self, K):
        """
        Initializes the MemoryBank.

        Args:
            K (int): The maximum capacity of the memory bank.
        """
        self.K = K
        self.memory = deque(maxlen=K)

    def add(self, q, v):
        """
        Adds a feature vector and its corresponding prediction to the memory bank.

        Args:
            q (torch.Tensor): The feature vector.
            v (torch.Tensor): The prediction corresponding to the feature vector.
        """
        for i in range(q.size(0)):
            q_i = q[i].detach().clone().view(-1).unsqueeze(0)
            v_i = v[i].detach().clone().view(-1).unsqueeze(0)
            self.memory.append((q_i, v_i))

    def retrieve(self, q, D):
        """
        Retrieves the D most similar feature vectors from the memory bank.

        Args:
            q (torch.Tensor): The feature vector to compare against.
            D (int): The number of most similar feature vectors to retrieve.

        Returns:
            list: A list of the D most similar feature vectors from the memory bank.
        """
        if len(self.memory) == 0:
            return []

        batch_size = q.size(0)

        keys = torch.stack([item[0] for item in self.memory]).mean(0)
        q = q.view(batch_size, -1)
        distances = F.cosine_similarity(keys, q, dim=1)
        D = min(D, distances.shape[0])
        nearest_indices = distances.topk(D, largest=False)[1]
        support_set = []
        for i in nearest_indices:
            data = self.memory[i][0]
            support_set.append(data)
        return support_set


class pmnettta(nn.Module):
    """
    Implements Test-Time Adaptation (TTA) for PMnet models.
    """

    def __init__(self, args, model, base_lr=0.00002):
        """
        Initializes the pmnettta module.

        Args:
            args (object): Configuration arguments.
            model (nn.Module): The PMnet model to adapt.
        """
        super().__init__()
        self.args = args
        self.steps = 1
        self.model, self.optimizer = self.prepare_model_and_optimizer(model, base_lr)
        self.base_lr = base_lr

        self.memory_bank = MemoryBank(K=100)

        self.retrieval_size = 7

    def forward(self, inputs, target):
        """
        Performs forward pass and test-time adaptation.

        Args:
            inputs (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Model output after adaptation.
        """
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(inputs, target)
        return outputs

    @torch.enable_grad()
    def rmse(self, x, y):
        """Computes the Root Mean Squared Error (RMSE)."""
        return torch.sqrt(torch.mean((x - y) ** 2))

    @torch.enable_grad()
    def nmse(self, x, y):
        """Computes the Normalized Mean Squared Error (NMSE)."""
        mse = torch.mean((x - y) ** 2)
        mse_reference = torch.mean((y - 0) ** 2)
        epsilon = 1e-8
        if mse_reference.item() < epsilon:
            mse_reference = epsilon
        nmse = mse / mse_reference
        return nmse

    @torch.enable_grad()
    def forward_and_adapt(self, inputs, target):
        """
        Performs a forward pass, calculates loss, and adapts the model.

        Args:
            inputs (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Model output after adaptation.
        """
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
        if support_set:
            centers = torch.stack([feats for feats in support_set]).mean(0).to(outputs.device)
            centers = F.normalize(centers)
            feats = F.normalize(feats).mean(0)
            dist = 1 - F.cosine_similarity(feats.view(-1).unsqueeze(0), centers, dim=1)
            dist = dist.mean()
            tau = 0.05
            weight_lr = torch.exp(-dist * tau)
            adjusted_lr = self.base_lr * weight_lr
        else:
            adjusted_lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr.item()
        loss.backward()
        self.optimizer.step()
        print(f"Adjusted Learning Rate: {adjusted_lr}")

        return outputs

    def save(self, ):
        pass

    def print(self, ):
        pass


def collect_params(model):
    """Collects parameters from specific layers (BatchNorm, LayerNorm, GroupNorm, Conv2d)."""
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
    """Configures the model for test-time adaptation by enabling gradients for specific layers."""
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


def prepare_model_and_optimizer(model, base_lr):
    """Prepares the model and optimizer for test-time adaptation."""
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = optim.Adam(
        params,
        lr=float(base_lr),
        betas=(0.9, 0.999),
        weight_decay=float(0.)
    )
    return model, optimizer


def setup(model, args):
    """Sets up the TTA model."""
    TTA_model = pmnettta(
        args,
        model
    )
    return TTA_model