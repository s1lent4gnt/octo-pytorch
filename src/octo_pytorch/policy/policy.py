from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from octo_pytorch.model.modeling_octo import OctoModel
from octo_pytorch.utils.normalization import Normalizer

DATASET_STATS = "dataset_statistics.npy"


@dataclass
class OctoPolicy:
    model: OctoModel
    normalizer: Optional[Normalizer] = None
    device: str = "cuda"

    @classmethod
    def from_pretrained(cls, repo_or_path: str, device="cuda"):
        model = OctoModel.from_pretrained(repo_or_path).to(device).eval()
        stats_path = Path(repo_or_path) / DATASET_STATS
        normalizer = Normalizer.from_file(stats_path) if stats_path.exists() else None
        return cls(model=model, normalizer=normalizer, device=device)

    def preprocess(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Map your env/dataset keys to model feature names and normalize if needed
        x = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
        if self.normalizer:
            x = self.normalizer.apply(x)
        return x

    def postprocess(self, out: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        act = out["action_pred"]
        if self.normalizer:
            act = self.normalizer.unapply({"action": act})["action"]
        return {"action": act}

    @torch.no_grad()
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        features = self.preprocess(observations)
        outputs = self.model(features, is_training=False)
        return self.postprocess(outputs)
