from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

from octo_pytorch.model.modeling_octo import OctoModel
from octo_pytorch.utils.normalization import Normalizer


@dataclass
class OctoPolicy:
    model: OctoModel
    normalizer: Optional[Normalizer] = None
    dataset_statistics: Optional[Dict[str, Any]] = None
    device: str = "cuda"

    @classmethod
    def from_pretrained(cls, repo_or_path: str, device="cuda"):
        from octo_pytorch.model.configuration_octo import OctoConfig
        from huggingface_hub import hf_hub_download
        import os
        
        if "/" in repo_or_path and not os.path.exists(repo_or_path):
            repo_id = repo_or_path
            
            # Load config from HuggingFace
            config = OctoConfig.from_pretrained(repo_id)
            
            # Create model with config
            model = OctoModel.from_pretrained(repo_id)
            model = model.to(device).eval()
            
            from transformers import T5EncoderModel
            t5_encoder = T5EncoderModel.from_pretrained("t5-base", torch_dtype=torch.float32)
            model.octo_transformer.task_tokenizers.language_instruction.t5_encoder = t5_encoder.to(device)
            
            # Freeze T5 encoder if needed
            if not model.octo_transformer.task_tokenizers.language_instruction.finetune_encoder:
                for param in model.octo_transformer.task_tokenizers.language_instruction.t5_encoder.parameters():
                    param.requires_grad = False
            
            # Try to download dataset statistics if available
            try:
                stats_file = hf_hub_download(repo_id, "dataset_statistics.npy")
                dataset_statistics = np.load(stats_file, allow_pickle=True).item()
                normalizer = Normalizer.from_file(stats_file) if hasattr(Normalizer, 'from_file') else None
            except:
                dataset_statistics = None
                normalizer = None
        else:
            config = OctoConfig.from_pretrained(Path(repo_or_path).parent)
            model = OctoModel(config)
            model.load_state_dict(torch.load(repo_or_path))
            model = model.to(device).eval()
            stats_path = Path(repo_or_path).parent / f"dataset_statistics_{model.config.model_name}.npy"
            if stats_path.exists():
                dataset_statistics = np.load(stats_path, allow_pickle=True).item()
                normalizer = Normalizer.from_file(stats_path) if hasattr(Normalizer, 'from_file') else None
            else:
                dataset_statistics = None
                normalizer = None
            
        return cls(model=model, normalizer=normalizer, dataset_statistics=dataset_statistics, device=device)

    def preprocess(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Map your env/dataset keys to model feature names and normalize if needed
        x = {}
        for k, v in obs.items():
            if k == "task":
                x[k] = v
                continue

            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, device=self.device)
            else:
                v = v.to(self.device)

            if "image" in k:
                if v.ndim == 3:
                    v = v.unsqueeze(0)
                v = v.float()
            
            x[k] = v

        if self.normalizer:
            x = self.normalizer.apply(x)
        return x

    def postprocess(self, out: torch.Tensor, unnormalization_statistics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        act = out
        if unnormalization_statistics is not None:
            mask = unnormalization_statistics.get(
                "mask",
                np.ones_like(unnormalization_statistics["mean"]),
            )
            mask = torch.from_numpy(mask).to(act.device).bool()
            act = act[..., : len(mask)]
            mean = torch.from_numpy(unnormalization_statistics["mean"]).to(act.device).float()
            std = torch.from_numpy(unnormalization_statistics["std"]).to(act.device).float()
            act = torch.where(mask, act * std + mean, act)
        elif self.normalizer:
            act = self.normalizer.unapply({"action": act})["action"]
        return {"action": act}

    @torch.no_grad()
    def get_action(self, observations: Dict[str, Any], unnormalization_statistics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        features = self.preprocess(observations)
        tasks = self.model.create_tasks(texts=features.pop("task"))
        timestep_pad_mask = features.pop("timestep_pad_mask")
        outputs = self.model(features, tasks, timestep_pad_mask)
        return self.postprocess(outputs, unnormalization_statistics)
