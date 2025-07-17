import torch
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.core.frozen_dict import unfreeze

# Define the JAX attention module
class JaxAttention(nn.Module):
    num_heads: int
    d_model: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None):
        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=True,
            dropout_rate=0.0,
            use_bias=True, # Explicitly use bias to match PyTorch
        )(x, x, mask=mask)

# Define the PyTorch attention module
class TorchAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        # The mask should be correctly formatted before being passed to this function.
        # The attention layer expects a boolean mask where True indicates a masked position.
        output, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        return output

def main():
    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    
    print("Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print("-" * 30)

    # 1. Create dummy input
    dummy_input_np = np.random.rand(batch_size, seq_len, d_model).astype(np.float32)
    x_torch = torch.from_numpy(dummy_input_np)
    x_jax = jnp.array(dummy_input_np)
    
    # Create a causal mask for testing
    causal_mask_np = np.tril(np.ones((batch_size, 1, seq_len, seq_len), dtype=np.float32))
    mask_jax = jnp.array(causal_mask_np)
    # PyTorch MHA mask needs to be boolean. True values are positions that will be prevented from attending.
    # The error message suggests a shape of (N * num_heads, L, S).
    
    # Start with a (N, L, S) boolean mask where True means "mask".
    py_mask_3d = torch.from_numpy(causal_mask_np.squeeze(1) == 0)
    
    # Expand to (N * num_heads, L, S) by repeating the mask for each head.
    mask_torch = py_mask_3d.repeat_interleave(num_heads, dim=0)


    # 2. Initialize JAX model and get weights
    jax_model = JaxAttention(num_heads=num_heads, d_model=d_model)
    key = jax.random.PRNGKey(0)
    params = jax_model.init(key, x_jax)['params']
    
    print("Initialized JAX model and generated weights.")

    # 3. Initialize PyTorch model
    torch_model = TorchAttention(num_heads=num_heads, d_model=d_model)
    torch_model.eval() # Set to evaluation mode
    
    print("Initialized PyTorch model.")

    # 4. Convert and load weights
    print("\nConverting JAX weights to PyTorch format...")
    
    # Use unfreeze to make the dictionary mutable
    jax_params = unfreeze(params)
    mha_params = jax_params['MultiHeadDotProductAttention_0']

    # JAX stores weights for Q, K, V in a way that separates heads.
    # We need to reshape and transpose them to match PyTorch's format.
    # JAX QKV kernel shape: (d_model, num_heads, head_dim) -> reshaped to (d_model, d_model)
    # JAX QKV bias shape: (num_heads, head_dim) -> reshaped to (d_model,)
    # PyTorch in_proj_weight shape: (3 * d_model, d_model)
    # PyTorch in_proj_bias shape: (3 * d_model)

    # Process Q, K, V weights
    q_w = np.array(mha_params['query']['kernel']).reshape(d_model, d_model).T
    k_w = np.array(mha_params['key']['kernel']).reshape(d_model, d_model).T
    v_w = np.array(mha_params['value']['kernel']).reshape(d_model, d_model).T
    in_proj_weight = torch.from_numpy(np.concatenate([q_w, k_w, v_w], axis=0))

    # Process Q, K, V biases
    q_b = np.array(mha_params['query']['bias']).reshape(d_model)
    k_b = np.array(mha_params['key']['bias']).reshape(d_model)
    v_b = np.array(mha_params['value']['bias']).reshape(d_model)
    in_proj_bias = torch.from_numpy(np.concatenate([q_b, k_b, v_b], axis=0))

    # Process output projection weights
    # JAX out kernel shape: (num_heads, head_dim, d_model) -> reshaped to (d_model, d_model)
    out_proj_weight = np.array(mha_params['out']['kernel']).reshape(d_model, d_model).T
    out_proj_weight = torch.from_numpy(out_proj_weight)
    
    # JAX out bias shape: (d_model,)
    out_proj_bias = torch.from_numpy(np.array(mha_params['out']['bias']))

    # Load into PyTorch model's state_dict
    new_state_dict = {
        'attention.in_proj_weight': in_proj_weight,
        'attention.in_proj_bias': in_proj_bias,
        'attention.out_proj.weight': out_proj_weight,
        'attention.out_proj.bias': out_proj_bias,
    }
    torch_model.load_state_dict(new_state_dict)
    
    print("Weight conversion and loading complete.")

    # 5. Run inference on both models
    print("\nRunning inference...")
    # Run with mask
    jax_output = jax_model.apply({'params': params}, x_jax, mask=mask_jax)
    torch_output = torch_model(x_torch, mask=mask_torch)
    
    # Convert JAX output to torch tensor for comparison
    jax_output_torch = torch.from_numpy(np.array(jax_output))

    # 6. Compare outputs
    print("Comparing outputs...")
    
    # Print shapes
    print(f"  JAX output shape: {jax_output.shape}")
    print(f"  PyTorch output shape: {torch_output.shape}")

    # Check for closeness
    are_close = torch.allclose(jax_output_torch, torch_output, atol=1e-6)
    
    print("-" * 30)
    if are_close:
        print("✅ SUCCESS: Outputs are identical.")
    else:
        print("❌ FAILURE: Outputs differ.")
        
    # Print difference statistics
    difference = torch.abs(jax_output_torch - torch_output)
    print(f"  Max absolute difference: {difference.max().item():.6f}")
    print(f"  Mean absolute difference: {difference.mean().item():.6f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
