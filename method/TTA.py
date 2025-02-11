import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

def pad_or_truncate(tensor, target_size):
    """Pads or truncates a tensor to a target size.

    If the tensor is smaller than the target size, it pads with zeros.
    If the tensor is larger, it truncates.
    """
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
        self.memory = deque(maxlen=K) # Use deque for efficient appending and popping

    def add(self, q, v):
        """
        Adds a feature vector and its corresponding prediction to the memory bank.

        Args:
            q (torch.Tensor): The feature vector.
            v (torch.Tensor): The prediction corresponding to the feature vector.
        """
        for i in range(q.size(0)):
            q_i = q[i].detach().clone().view(-1).unsqueeze(0) # Detach and clone for memory efficiency
            v_i = v[i].detach().clone().view(-1).unsqueeze(0) # Detach and clone for memory efficiency
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
        keys = torch.stack([item[0] for item in self.memory]).mean(0) # Stack all feature vectors
        q = q.view(batch_size, -1) # Reshape the query vector
        distances = F.cosine_similarity(keys, q, dim=1) # Calculate cosine similarity
        D = min(D, distances.shape[0]) # Ensure D is not larger than the memory size
        nearest_indices = distances.topk(D, largest=False)[1] # Find indices of top D nearest vectors
        support_set = [self.memory[i][0] for i in nearest_indices] # Collect the top D nearest vectors
        return support_set

class TTA(nn.Module):
    """
    Implements Test-Time Adaptation (TTA).
    """
    def __init__(self, model, base_lr=0.000005):
        """
        Initializes the TTA module.

        Args:
            model (nn.Module): The model to adapt.
            base_lr (float): Base learning rate for adaptation.
        """
        super().__init__()
        self.steps = 1 # Number of adaptation steps
        self.model, self.optimizer = prepare_model_and_optimizer(model,base_lr) # Prepare model and optimizer
        self.memory_bank = MemoryBank(K=100) # Initialize the memory bank
        self.base_lr = base_lr # Store the base learning rate
        self.retrieval_size = 5 # Number of samples to retrieve from the memory bank

    # RME-GAN
    # def forward(self, inputs,  target):

    # ACT-GAN/RadioUNet/REMNet+
    def forward(self, antenna, builds, target):
        """
        Performs forward pass and test-time adaptation.

        Args:
            antenna (torch.Tensor): Input antenna data.
            builds (torch.Tensor): Input builds data.
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Model output after adaptation.
        """
        for _ in range(self.steps):
            # RME-GAN
            # outputs = self.forward_and_adapt(inputs, target) # Perform forward and adapt step
            # ACT-GAN
            outputs = self.forward_and_adapt(antenna,builds, target) # Perform forward and adapt step
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


    # RME-GAN
    # @torch.enable_grad()
    # def forward_and_adapt(self, inputs, target):
    #
    #     self.optimizer.zero_grad()
    #
    #     feats, outputs = self.model(inputs, return_features=True)


    # ACT-GAN/RadioUNet/REMNet+
    @torch.enable_grad()
    def forward_and_adapt(self, antenna, builds, target):
        """
        Performs a forward pass, calculates loss, and adapts the model.

        Args:
            antenna (torch.Tensor): Input antenna data.
            builds (torch.Tensor): Input builds data.
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Model output after adaptation.
        """
        self.optimizer.zero_grad() # Reset the gradients
        feats, outputs = self.model(antenna, builds, return_features=True) # Get the features and outputs

        rmse_loss = self.rmse(outputs, target) # Calculate RMSE loss
        nmse_loss = self.nmse(outputs, target) # Calculate NMSE loss
        loss = (rmse_loss + nmse_loss) / 1 # Calculate the total loss

        print(f"RMSE Loss: {rmse_loss.item()}, NMSE Loss: {nmse_loss.item()}, Total Loss: {loss.item()}")

        if not (torch.isnan(feats).any() or torch.isinf(feats).any() or
                torch.isnan(outputs).any() or torch.isinf(outputs).any()):
            self.memory_bank.add(feats, outputs) # Add features and outputs to the memory bank

        support_set = self.memory_bank.retrieve(feats, D=self.retrieval_size) # Retrieve similar samples from memory
        if support_set:
            centers = torch.stack([feats for feats in support_set]).mean(0).to(outputs.device) # Calculate the center of retrieved features
            centers = F.normalize(centers) # Normalize the center
            feats = F.normalize(feats).mean(0) # Normalize the input features
            dist = 1 - F.cosine_similarity(feats.view(-1).unsqueeze(0), centers, dim=1) # Calculate cosine distance
            dist = dist.mean()

            tau = 0.5
            weight_lr = torch.exp(-dist * tau) # Calculate weight for learning rate adjustment
            adjusted_lr = self.base_lr * weight_lr # Adjust the learning rate
        else:
            adjusted_lr = self.base_lr # Use the base learning rate if no support set is found

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr.item() # Set the learning rate for the optimizer

        loss.backward() # Backpropagate the loss
        self.optimizer.step() # Update the model parameters

        print(f"Adjusted Learning Rate: {adjusted_lr}")

        return outputs

def prepare_model_and_optimizer(model,base_lr):
    """
    Prepares the model and optimizer for TTA.

    Args:
        model (nn.Module): The model to adapt.

    Returns:
        tuple: Model and optimizer.
    """
    params = list(model.parameters()) # Get the model parameters
    optimizer = optim.Adam(params, lr=float(base_lr), betas=(0.9, 0.999), weight_decay=float(0.)) # Initialize the Adam optimizer
    return model, optimizer

def setup(model):
    """
    Sets up the TTA model.

    Args:
        model (nn.Module): The model to adapt.

    Returns:
        TTA: The TTA model.
    """
    TTA_model = TTA(model) # Create the TTA model
    return TTA_model