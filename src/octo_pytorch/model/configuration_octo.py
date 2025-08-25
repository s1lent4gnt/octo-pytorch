from transformers import PretrainedConfig


class OctoConfig(PretrainedConfig):
    model_type: str = "octo"
    model_name: str = "octo-base"
    token_embedding_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    mlp_dim: int = 0
    action_dim: int = 7
    action_horizon: int = 4
    action_max_horizon: int = 10

    tokenizer_name: str = "t5-base"
    max_text_len: int = 16
    repeat_task_tokens: bool = True

    compute_dtype: str = "float32"

    PRESETS = {
        "octo-base":  dict(token_embedding_size=768, num_layers=12, num_heads=12, mlp_dim=3072),
        "octo-small": dict(token_embedding_size=384, num_layers=12, num_heads=6,  mlp_dim=1536),
    }

    def __init__(self, model_name: str = "octo-base", action_dim: int = 7,
                 action_horizon: int = 4, compute_dtype: str = "float32", **kwargs):
        super().__init__(**kwargs)  # absorb unknown keys for HF compatibility
        if model_name not in self.PRESETS:
            raise ValueError(f"Unknown model name: {model_name}")
        self.model_name = model_name
        preset = self.PRESETS[model_name]
        self.token_embedding_size = preset["token_embedding_size"]
        self.num_layers = preset["num_layers"]
        self.num_heads = preset["num_heads"]
        self.mlp_dim = preset["mlp_dim"]
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.compute_dtype = compute_dtype
