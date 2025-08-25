import torch
from octo_pytorch.model.components.block_transformer import TimestepGroup
from octo_pytorch.model.modeling_octo import DiffusionActionHead


def test_diffusion_loss():
    batch_size = 2
    window_size = 2
    action_dim = 7
    action_horizon = 4
    embedding_size = 768

    action_head = DiffusionActionHead(
        readout_key="readout_action",
        use_map=False,
        input_dim=768,
        action_dim=7,
        action_horizon=4,
    )

    transformer_outputs = {
        "readout_action": TimestepGroup(
            tokens=torch.randn(batch_size, window_size, 1, embedding_size),
            mask=torch.ones(batch_size, window_size, 1, dtype=torch.bool),
            name="readout_action",
            attention_rules={},
        )
    }

    actions = torch.randn(batch_size, window_size, action_horizon, action_dim)
    timestep_pad_mask = torch.ones(batch_size, window_size, dtype=torch.bool)
    action_pad_mask = torch.ones(
        batch_size, window_size, action_horizon, action_dim, dtype=torch.bool
    )

    loss, metrics = action_head.loss(
        transformer_outputs, actions, timestep_pad_mask, action_pad_mask
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "mse" in metrics
    assert metrics["mse"].item() >= 0
    print("Loss is working!")


if __name__ == "__main__":
    test_diffusion_loss()
