import numpy as np
import torch

class Normalizer:
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def from_file(cls, path):
        stats = np.load(path, allow_pickle=True).item()
        return cls(stats)

    def normalize(self, data, key):
        if key in self.stats:
            mean = torch.from_numpy(self.stats[key]["mean"]).to(data.device)
            std = torch.from_numpy(self.stats[key]["std"]).to(data.device)
            return (data - mean) / std
        return data

    def unnormalize(self, data, key):
        if key in self.stats:
            mean = torch.from_numpy(self.stats[key]["mean"]).to(data.device)
            std = torch.from_numpy(self.stats[key]["std"]).to(data.device)
            return data * std + mean
        return data

    def apply(self, data):
        for key, value in data.items():
            if key in self.stats:
                data[key] = self.normalize(value, key)
        return data

    def unapply(self, data):
        for key, value in data.items():
            if key in self.stats:
                data[key] = self.unnormalize(value, key)
        return data
