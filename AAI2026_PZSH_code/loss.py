import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb


class PZSHAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, exclude_self=True):
        super().__init__()
        self.temperature = temperature
        self.exclude_self = exclude_self
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, real_features, synthetic_features, labels):
        real_features = F.normalize(real_features, dim=-1)
        synthetic_features = F.normalize(synthetic_features, dim=-1)
        logits = torch.matmul(real_features, synthetic_features.T) / self.temperature
        log_probs = self.log_softmax(logits)

        with torch.no_grad():
            positive_mask = (labels @ labels.T > 0).float()
            if self.exclude_self:
                positive_mask.fill_diagonal_(0.0)
            positive_mask = positive_mask / (positive_mask.sum(dim=1, keepdim=True) + 1e-6)

        return -(positive_mask * log_probs).sum(dim=1).mean()


class PZSHProxyHashLoss(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.config = config
        self.bit = bit
        self.device = config["device"]
        self.hash_center = self._generate_hash_centers(bit, config["n_class"]).to(self.device)
        self.label_center = torch.eye(config["n_class"], dtype=torch.float32, device=self.device)
        self.hash_memory = torch.randn(config["num_train"], bit, device=self.device)
        self.label_memory = torch.zeros(config["num_train"], config["n_class"], device=self.device)

    def forward(self, query_hash, momentum_hash, labels, indices, epoch=0):
        self.hash_memory[indices, :] = momentum_hash.detach()
        self.label_memory[indices, :] = labels.detach()

        center_loss = self._center_loss(query_hash, labels)
        quantization_loss = (query_hash.abs() - 1).pow(2).mean()
        proxy_loss = 0.0 if epoch < self.config["proxy_warmup"] else self._proxy_loss(query_hash, labels)
        return center_loss + self.config["lambda_quant"] * quantization_loss + self.config["beta"] * proxy_loss

    def _center_loss(self, hash_codes, labels):
        logits = torch.matmul(F.normalize(hash_codes), F.normalize(self.hash_center).t())
        logits = (self.bit ** 0.5) * logits
        targets = (labels @ self.label_center.t()).float()
        probs = torch.softmax(logits, dim=1).clamp(min=1e-6, max=1 - 1e-6)
        loss = targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)
        return -loss.mean()

    def _proxy_loss(self, hash_codes, labels):
        hash_codes = F.normalize(hash_codes)
        memory_hash = F.normalize(self.hash_memory)
        positive_mask = (labels @ self.label_memory.t() > 0).float()
        similarity = hash_codes @ memory_hash.t()
        loss = positive_mask * torch.log1p(torch.exp(0.5 * (1 - similarity)))
        return loss.sum() / (positive_mask.sum() + 1e-6)

    def _generate_hash_centers(self, bit, n_class, seed=42):
        capacity = (2 ** bit) / n_class
        self._get_margin(bit, capacity)
        rng = np.random.default_rng(seed)
        centers = np.zeros((n_class, bit), dtype=np.int8)

        for _ in range(30):
            for class_idx in range(n_class):
                center = np.ones(bit, dtype=np.int8)
                center[rng.choice(bit, size=bit // 2, replace=False)] = -1
                centers[class_idx] = center

            distances = []
            for i in range(n_class):
                for j in range(i + 1, n_class):
                    distances.append(np.sum(centers[i] != centers[j]))
            distances = np.asarray(distances)
            if distances.min() > bit // 4 and distances.mean() >= bit / 2:
                break

        return torch.tensor(centers, dtype=torch.float32)

    @staticmethod
    def _get_margin(bit, capacity):
        d_min, d_max = 0, 0
        for dim in range(2 * bit + 4):
            if sum(comb(bit, i) for i in range((dim - 1) // 2 + 1)) <= capacity:
                if sum(comb(bit, i) for i in range(dim // 2 + 1)) > capacity:
                    d_min = dim
        for dim in range(2 * bit + 4):
            if sum(comb(bit, i) for i in range(dim)) >= capacity:
                if sum(comb(bit, i) for i in range(dim - 1)) < capacity:
                    d_max = dim
        return d_min, d_max
