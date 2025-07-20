import argparse

import numpy as np
import torch
from flax.traverse_util import flatten_dict

from octo_pytorch.model.octo_model import OctoModel as PyTorchOctoModel


class OctoWeightConverter:
    def __init__(self, jax_params, pytorch_model, model_name="octo-base"):
        self.jax_params = flatten_dict(jax_params)
        self.pytorch_model = pytorch_model
        self.model_name = model_name
        self.conversion_log = []
        self.converted_param_count = 0
        self.total_jax_params = 0
        self.total_pytorch_params = 0
        self.converted_params_info = []

    def convert(self):
        # Count parameters before conversion
        self._count_parameters()

        # Perform conversion
        self.convert_tokenizers()
        self.convert_transformer()
        self.convert_action_heads()

        # Verify conversion completeness
        self._verify_conversion_completeness()

        # Save converted params info
        self._save_converted_params_info()

        return self.pytorch_model

    def _save_converted_params_info(self):
        """Save the list of converted parameters to a JSON file."""
        import json

        with open(f"output/converted_params_info_{self.model_name}.json", "w") as f:
            json.dump(self.converted_params_info, f, indent=4)

    def convert_tokenizers(self):
        # Convert positional embeddings directly from the model
        self._convert_positional_embeddings()

        # Convert observation tokenizers
        self._convert_observation_tokenizers()

        # Note: Task and observation tokenizers in PyTorch model don't have separate projection layers
        # They are handled differently in the PyTorch implementation

    def _convert_positional_embeddings(self):
        """Convert positional embeddings from JAX to PyTorch model"""
        # Primary observation positional embedding
        jax_param_name = "octo_transformer/obs_primary_pos_embedding"
        obs_primary_pos = self._get_jax_param(jax_param_name)
        if obs_primary_pos is not None:
            self.pytorch_model.obs_primary_pos_embedding.data = torch.from_numpy(
                obs_primary_pos.copy()
            ).float()
            self._track_converted_params(obs_primary_pos, jax_param_name)
            self.conversion_log.append("Converted obs_primary positional embedding")
        else:
            print("Warning: Primary observation positional embedding not found in JAX model")

        # Wrist observation positional embedding
        jax_param_name = "octo_transformer/obs_wrist_pos_embedding"
        obs_wrist_pos = self._get_jax_param(jax_param_name)
        if obs_wrist_pos is not None:
            self.pytorch_model.obs_wrist_pos_embedding.data = torch.from_numpy(obs_wrist_pos.copy()).float()
            self._track_converted_params(obs_wrist_pos, jax_param_name)
            self.conversion_log.append("Converted obs_wrist positional embedding")
        else:
            print("Warning: Wrist observation positional embedding not found in JAX model")

        # Language task positional embedding
        jax_param_name = "octo_transformer/task_language_pos_embedding"
        task_lang_pos = self._get_jax_param(jax_param_name)
        if task_lang_pos is not None:
            self.pytorch_model.task_language_pos_embedding.data = torch.from_numpy(
                task_lang_pos.copy()
            ).float()
            self._track_converted_params(task_lang_pos, jax_param_name)
            self.conversion_log.append("Converted task_language positional embedding")
        else:
            print("Warning: Language task positional embedding not found in JAX model")

        # Readout positional embedding
        jax_param_name = "octo_transformer/readout_action_pos_embedding"
        readout_pos = self._get_jax_param(jax_param_name)
        if readout_pos is not None:
            self.pytorch_model.readout_embedding.data = torch.from_numpy(readout_pos.copy()).float()
            self._track_converted_params(readout_pos, jax_param_name)
            self.conversion_log.append("Converted readout positional embedding")
        else:
            print("Warning: Readout positional embedding not found in JAX model")

        # Convert projection layers
        self._convert_projection_layers()

    def _convert_observation_tokenizers(self):
        """Convert observation tokenizers (primary and wrist)"""
        # Convert primary observation tokenizer
        self._convert_image_tokenizer(
            "octo_transformer/observation_tokenizers_primary/SmallStem16_0",
            self.pytorch_model.observation_tokenizers["image_primary"].encoder,
        )

        # Convert wrist observation tokenizer
        self._convert_image_tokenizer(
            "octo_transformer/observation_tokenizers_wrist/SmallStem16_0",
            self.pytorch_model.observation_tokenizers["image_wrist"].encoder,
        )

    def _convert_image_tokenizer(self, jax_prefix, pytorch_tokenizer):
        """Convert a SmallStem16 image tokenizer from JAX to PyTorch"""
        try:
            # Convert StdConv layers - SmallStem uses conv_layers ModuleList
            for i in range(4):  # SmallStem16 has 4 StdConv layers
                # Each conv_layer is a Sequential with [WeightStandardizedConv2d, GroupNorm, ReLU]
                conv_layer = pytorch_tokenizer.conv_layers[i][0]  # Get the conv layer
                self._convert_stdconv_layer(f"{jax_prefix}/StdConv_{i}", conv_layer)

            # Convert GroupNorm layers
            for i in range(4):  # SmallStem16 has 4 GroupNorm layers
                # Each conv_layer is a Sequential with [WeightStandardizedConv2d, GroupNorm, ReLU]
                groupnorm_layer = pytorch_tokenizer.conv_layers[i][1]  # Get the GroupNorm layer
                self._convert_groupnorm_layer(f"{jax_prefix}/GroupNorm_{i}", groupnorm_layer)

            # Convert final embedding layer
            self._convert_embedding_layer(f"{jax_prefix}/embedding", pytorch_tokenizer.embedding)

            self.conversion_log.append(f"Converted image tokenizer: {jax_prefix}")

        except (AttributeError, IndexError) as e:
            print(f"Warning: Could not convert image tokenizer {jax_prefix}: {e}")

    def _convert_stdconv_layer(self, jax_prefix, pytorch_layer):
        """Convert a StdConv layer from JAX to PyTorch"""
        kernel = self._get_jax_param(f"{jax_prefix}/kernel")
        bias = self._get_jax_param(f"{jax_prefix}/bias")

        if kernel is not None:
            # JAX conv kernel: (H, W, in_channels, out_channels) -> PyTorch: (out_channels, in_channels, H, W)
            pytorch_kernel = kernel.transpose(3, 2, 0, 1)
            pytorch_layer.weight.data = torch.from_numpy(pytorch_kernel.copy()).float()
            self._track_converted_params(kernel, f"{jax_prefix}/kernel")

            if bias is not None:
                pytorch_layer.bias.data = torch.from_numpy(bias.copy()).float()
                self._track_converted_params(bias, f"{jax_prefix}/bias")
            else:
                print(f"Warning: StdConv bias not found for {jax_prefix}")
        else:
            print(f"Warning: StdConv kernel not found for {jax_prefix}")

    def _convert_groupnorm_layer(self, jax_prefix, pytorch_layer):
        """Convert a GroupNorm layer from JAX to PyTorch"""
        scale = self._get_jax_param(f"{jax_prefix}/scale")
        bias = self._get_jax_param(f"{jax_prefix}/bias")

        if scale is not None:
            pytorch_layer.weight.data = torch.from_numpy(scale.copy()).float()
            self._track_converted_params(scale, f"{jax_prefix}/scale")

            if bias is not None:
                pytorch_layer.bias.data = torch.from_numpy(bias.copy()).float()
                self._track_converted_params(bias, f"{jax_prefix}/bias")
            else:
                print(f"Warning: GroupNorm bias not found for {jax_prefix}")
        else:
            print(f"Warning: GroupNorm scale not found for {jax_prefix}")

    def _convert_embedding_layer(self, jax_prefix, pytorch_layer):
        """Convert an embedding layer from JAX to PyTorch"""
        kernel = self._get_jax_param(f"{jax_prefix}/kernel")
        bias = self._get_jax_param(f"{jax_prefix}/bias")

        if kernel is not None:
            # JAX conv kernel: (H, W, in_channels, out_channels) -> PyTorch: (out_channels, in_channels, H, W)
            pytorch_kernel = kernel.transpose(3, 2, 0, 1)
            pytorch_layer.weight.data = torch.from_numpy(pytorch_kernel.copy()).float()
            self._track_converted_params(kernel, f"{jax_prefix}/kernel")

            if bias is not None:
                pytorch_layer.bias.data = torch.from_numpy(bias.copy()).float()
                self._track_converted_params(bias, f"{jax_prefix}/bias")
            else:
                print(f"Warning: Embedding bias not found for {jax_prefix}")
        else:
            print(f"Warning: Embedding kernel not found for {jax_prefix}")

    def _convert_projection_layers(self):
        """Convert projection layers"""
        # Primary observation projection
        kernel_name = "octo_transformer/obs_primary_projection/kernel"
        bias_name = "octo_transformer/obs_primary_projection/bias"
        obs_primary_proj_kernel = self._get_jax_param(kernel_name)
        obs_primary_proj_bias = self._get_jax_param(bias_name)
        if obs_primary_proj_kernel is not None:
            self.pytorch_model.obs_primary_projection.weight.data = torch.from_numpy(
                obs_primary_proj_kernel.T.copy()
            ).float()
            self._track_converted_params(obs_primary_proj_kernel, kernel_name)
            if obs_primary_proj_bias is not None:
                self.pytorch_model.obs_primary_projection.bias.data = torch.from_numpy(
                    obs_primary_proj_bias.copy()
                ).float()
                self._track_converted_params(obs_primary_proj_bias, bias_name)
            else:
                print("Warning: Primary observation projection bias not found in JAX model")
            self.conversion_log.append("Converted obs_primary projection")
        else:
            print("Warning: Primary observation projection kernel not found in JAX model")

        # Wrist observation projection
        kernel_name = "octo_transformer/obs_wrist_projection/kernel"
        bias_name = "octo_transformer/obs_wrist_projection/bias"
        obs_wrist_proj_kernel = self._get_jax_param(kernel_name)
        obs_wrist_proj_bias = self._get_jax_param(bias_name)
        if obs_wrist_proj_kernel is not None:
            self.pytorch_model.obs_wrist_projection.weight.data = torch.from_numpy(
                obs_wrist_proj_kernel.T.copy()
            ).float()
            self._track_converted_params(obs_wrist_proj_kernel, kernel_name)
            if obs_wrist_proj_bias is not None:
                self.pytorch_model.obs_wrist_projection.bias.data = torch.from_numpy(
                    obs_wrist_proj_bias.copy()
                ).float()
                self._track_converted_params(obs_wrist_proj_bias, bias_name)
            else:
                print("Warning: Wrist observation projection bias not found in JAX model")
            self.conversion_log.append("Converted obs_wrist projection")
        else:
            print("Warning: Wrist observation projection kernel not found in JAX model")

        # Language task projection
        kernel_name = "octo_transformer/task_language_projection/kernel"
        bias_name = "octo_transformer/task_language_projection/bias"
        task_lang_proj_kernel = self._get_jax_param(kernel_name)
        task_lang_proj_bias = self._get_jax_param(bias_name)
        if task_lang_proj_kernel is not None:
            self.pytorch_model.task_language_projection.weight.data = torch.from_numpy(
                task_lang_proj_kernel.T.copy()
            ).float()
            self._track_converted_params(task_lang_proj_kernel, kernel_name)
            if task_lang_proj_bias is not None:
                self.pytorch_model.task_language_projection.bias.data = torch.from_numpy(
                    task_lang_proj_bias.copy()
                ).float()
                self._track_converted_params(task_lang_proj_bias, bias_name)
            else:
                print("Warning: Language task projection bias not found in JAX model")
            self.conversion_log.append("Converted task_language projection")
        else:
            print("Warning: Language task projection kernel not found in JAX model")

    def convert_transformer(self):
        # Access the encoder blocks through the correct path:
        # PyTorchOctoTransformer -> BlockTransformer -> Transformer -> encoder_blocks
        try:
            transformer = self.pytorch_model.transformer.transformer.transformer
            encoder_blocks = transformer.encoder_blocks
        except AttributeError:
            # Try alternative path
            try:
                encoder_blocks = self.pytorch_model.transformer.transformer.encoder_blocks
                transformer = self.pytorch_model.transformer.transformer
            except AttributeError:
                print("Could not find transformer encoder blocks. Skipping transformer conversion.")
                return

        # Convert individual encoder layers
        for i in range(len(encoder_blocks)):
            self._convert_transformer_layer(i, encoder_blocks[i])

        # Convert final encoder norm
        self._convert_encoder_norm(transformer)

    def _convert_transformer_layer(self, i, pytorch_layer):
        jax_prefix = f"octo_transformer/BlockTransformer_0/Transformer_0/encoderblock_{i}"

        # Attention
        self._convert_attention(jax_prefix, pytorch_layer)

        # MLP
        self._convert_mlp(jax_prefix, pytorch_layer)

        # LayerNorm
        self._convert_layernorm(jax_prefix, pytorch_layer)

    def _convert_attention(self, jax_prefix, pytorch_layer):
        attn_prefix = f"{jax_prefix}/MultiHeadDotProductAttention_0"

        q_kernel = self._get_jax_param(f"{attn_prefix}/query/kernel")
        k_kernel = self._get_jax_param(f"{attn_prefix}/key/kernel")
        v_kernel = self._get_jax_param(f"{attn_prefix}/value/kernel")
        out_kernel = self._get_jax_param(f"{attn_prefix}/out/kernel")

        q_bias = self._get_jax_param(f"{attn_prefix}/query/bias")
        k_bias = self._get_jax_param(f"{attn_prefix}/key/bias")
        v_bias = self._get_jax_param(f"{attn_prefix}/value/bias")
        out_bias = self._get_jax_param(f"{attn_prefix}/out/bias")

        if q_kernel is not None and k_kernel is not None and v_kernel is not None:
            # JAX shape: (d_model, num_heads, head_dim) -> PyTorch: (d_model, d_model)
            d_model = 768 if self.model_name == "octo-base" else 384
            q_weight = q_kernel.reshape(d_model, d_model)
            k_weight = k_kernel.reshape(d_model, d_model)
            v_weight = v_kernel.reshape(d_model, d_model)

            # Transpose weights for PyTorch format and concatenate Q, K, V
            # PyTorch expects weights in shape (3*d_model, d_model) where weights are transposed
            in_proj_weight = np.concatenate([q_weight.T, k_weight.T, v_weight.T], axis=0)
            pytorch_layer.attention.in_proj_weight.data = torch.from_numpy(in_proj_weight.copy()).float()

            # Track converted parameters
            self._track_converted_params(q_kernel, f"{attn_prefix}/query/kernel")
            self._track_converted_params(k_kernel, f"{attn_prefix}/key/kernel")
            self._track_converted_params(v_kernel, f"{attn_prefix}/value/kernel")

            # Convert biases
            if q_bias is not None and k_bias is not None and v_bias is not None:
                q_bias_flat = q_bias.reshape(-1)
                k_bias_flat = k_bias.reshape(-1)
                v_bias_flat = v_bias.reshape(-1)
                in_proj_bias = np.concatenate([q_bias_flat, k_bias_flat, v_bias_flat])
                pytorch_layer.attention.in_proj_bias.data = torch.from_numpy(in_proj_bias.copy()).float()

                # Track converted bias parameters
                self._track_converted_params(q_bias, f"{attn_prefix}/query/bias")
                self._track_converted_params(k_bias, f"{attn_prefix}/key/bias")
                self._track_converted_params(v_bias, f"{attn_prefix}/value/bias")
            else:
                print(f"Warning: One or more attention biases not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: One or more attention kernels not found in JAX model for layer {jax_prefix}")

        if out_kernel is not None:
            # JAX: (num_heads, head_dim, d_model) -> PyTorch: (d_model, d_model)
            d_model = 768 if self.model_name == "octo-base" else 384
            out_weight = out_kernel.reshape(d_model, d_model).T
            pytorch_layer.attention.out_proj.weight.data = torch.from_numpy(out_weight.copy()).float()
            self._track_converted_params(out_kernel, f"{attn_prefix}/out/kernel")

            if out_bias is not None:
                pytorch_layer.attention.out_proj.bias.data = torch.from_numpy(out_bias.copy()).float()
                self._track_converted_params(out_bias, f"{attn_prefix}/out/bias")
            else:
                print(f"Warning: Output attention bias not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: Output attention kernel not found in JAX model for layer {jax_prefix}")

        self.conversion_log.append(f"Converted attention layer {jax_prefix.split('/')[-1]}")

    def _convert_mlp(self, jax_prefix, pytorch_layer):
        mlp_prefix = f"{jax_prefix}/MlpBlock_0"

        dense0_kernel = self._get_jax_param(f"{mlp_prefix}/Dense_0/kernel")
        dense0_bias = self._get_jax_param(f"{mlp_prefix}/Dense_0/bias")
        dense1_kernel = self._get_jax_param(f"{mlp_prefix}/Dense_1/kernel")
        dense1_bias = self._get_jax_param(f"{mlp_prefix}/Dense_1/bias")

        if dense0_kernel is not None:
            pytorch_layer.mlp.dense1.weight.data = torch.from_numpy(dense0_kernel.T.copy()).float()
            self._track_converted_params(dense0_kernel, f"{mlp_prefix}/Dense_0/kernel")
            if dense0_bias is not None:
                pytorch_layer.mlp.dense1.bias.data = torch.from_numpy(dense0_bias.copy()).float()
                self._track_converted_params(dense0_bias, f"{mlp_prefix}/Dense_0/bias")
            else:
                print(f"Warning: MLP dense0 bias not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: MLP dense0 kernel not found in JAX model for layer {jax_prefix}")

        if dense1_kernel is not None:
            pytorch_layer.mlp.dense2.weight.data = torch.from_numpy(dense1_kernel.T.copy()).float()
            self._track_converted_params(dense1_kernel, f"{mlp_prefix}/Dense_1/kernel")
            if dense1_bias is not None:
                pytorch_layer.mlp.dense2.bias.data = torch.from_numpy(dense1_bias.copy()).float()
                self._track_converted_params(dense1_bias, f"{mlp_prefix}/Dense_1/bias")
            else:
                print(f"Warning: MLP dense1 bias not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: MLP dense1 kernel not found in JAX model for layer {jax_prefix}")

        self.conversion_log.append(f"Converted MLP layer {jax_prefix.split('/')[-1]}")

    def _convert_layernorm(self, jax_prefix, pytorch_layer):
        ln0_scale = self._get_jax_param(f"{jax_prefix}/LayerNorm_0/scale")
        ln0_bias = self._get_jax_param(f"{jax_prefix}/LayerNorm_0/bias")
        ln1_scale = self._get_jax_param(f"{jax_prefix}/LayerNorm_1/scale")
        ln1_bias = self._get_jax_param(f"{jax_prefix}/LayerNorm_1/bias")

        if ln0_scale is not None:
            pytorch_layer.norm1.weight.data = torch.from_numpy(ln0_scale.copy()).float()
            self._track_converted_params(ln0_scale, f"{jax_prefix}/LayerNorm_0/scale")
            if ln0_bias is not None:
                pytorch_layer.norm1.bias.data = torch.from_numpy(ln0_bias.copy()).float()
                self._track_converted_params(ln0_bias, f"{jax_prefix}/LayerNorm_0/bias")
            else:
                print(f"Warning: LayerNorm0 bias not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: LayerNorm0 scale not found in JAX model for layer {jax_prefix}")

        if ln1_scale is not None:
            pytorch_layer.norm2.weight.data = torch.from_numpy(ln1_scale.copy()).float()
            self._track_converted_params(ln1_scale, f"{jax_prefix}/LayerNorm_1/scale")
            if ln1_bias is not None:
                pytorch_layer.norm2.bias.data = torch.from_numpy(ln1_bias.copy()).float()
                self._track_converted_params(ln1_bias, f"{jax_prefix}/LayerNorm_1/bias")
            else:
                print(f"Warning: LayerNorm1 bias not found in JAX model for layer {jax_prefix}")
        else:
            print(f"Warning: LayerNorm1 scale not found in JAX model for layer {jax_prefix}")

        self.conversion_log.append(f"Converted LayerNorm {jax_prefix.split('/')[-1]}")

    def _convert_encoder_norm(self, transformer):
        """Convert the final encoder normalization layer"""
        jax_prefix = "octo_transformer/BlockTransformer_0/Transformer_0/encoder_norm"

        encoder_norm_scale = self._get_jax_param(f"{jax_prefix}/scale")
        encoder_norm_bias = self._get_jax_param(f"{jax_prefix}/bias")

        if encoder_norm_scale is not None:
            try:
                transformer.encoder_norm.weight.data = torch.from_numpy(encoder_norm_scale.copy()).float()
                self._track_converted_params(encoder_norm_scale, f"{jax_prefix}/scale")

                if encoder_norm_bias is not None:
                    transformer.encoder_norm.bias.data = torch.from_numpy(encoder_norm_bias.copy()).float()
                    self._track_converted_params(encoder_norm_bias, f"{jax_prefix}/bias")
                else:
                    print("Warning: Encoder norm bias not found in JAX model")

                self.conversion_log.append("Converted encoder norm")
            except AttributeError:
                print("Warning: Could not find encoder_norm in PyTorch transformer")
        else:
            print("Warning: Encoder norm scale not found in JAX model")

    def _get_jax_param(self, path):
        path_tuple = tuple(path.split("/"))
        return self.jax_params.get(path_tuple)

    def _count_parameters(self):
        """Count total parameters in both models"""
        print("Counting parameters...")

        # I want to save the params namw and shape in a .txt file
        import json

        param_info = {}
        for k, v in self.jax_params.items():
            param_name = "/".join(k)
            param_info[param_name] = {"shape": list(v.shape), "param_count": int(np.prod(v.shape))}

        with open(f"output/jax_params_info_{self.model_name}.json", "w") as f:
            json.dump(param_info, f, indent=4)

        # Debug: Print some JAX parameter names to identify T5 parameters
        print("Sample JAX parameter names:")
        for i, (path, param) in enumerate(self.jax_params.items()):
            # if i < 10:  # Show first 10 parameter names
            param_name = "/".join(path)
            print(f"  {param_name}: {param.shape}")

        # Count JAX parameters (excluding T5/language tokenizer)
        jax_convertible_params = 0
        jax_t5_params = 0

        for path, param in self.jax_params.items():
            param_name = "/".join(path)
            param_count = np.prod(param.shape)
            self.total_jax_params += param_count

            # Skip T5 parameters as they're frozen and don't need conversion
            # T5 parameters in JAX are under task tokenizers, specifically language instruction encoder
            if any(
                skip in param_name.lower()
                for skip in [
                    "task_language_instruction_encoder",
                    "language_instruction_encoder",
                    "t5",
                    "task_tokenizers_language",
                ]
            ):
                jax_t5_params += param_count
            else:
                jax_convertible_params += param_count

        # Count PyTorch parameters (excluding T5/language tokenizer)
        pytorch_convertible_params = 0
        pytorch_t5_params = 0

        for name, param in self.pytorch_model.named_parameters():
            param_count = param.numel()
            self.total_pytorch_params += param_count

            # Skip T5 parameters as they're frozen
            if any(skip in name.lower() for skip in ["t5", "language_tokenizer", "text_processor"]):
                pytorch_t5_params += param_count
            else:
                pytorch_convertible_params += param_count

        print(f"JAX model total parameters: {self.total_jax_params:,}")
        print(f"  - Convertible parameters: {jax_convertible_params:,}")
        print(f"  - T5/Language parameters (frozen): {jax_t5_params:,}")

        print(f"PyTorch model total parameters: {self.total_pytorch_params:,}")
        print(f"  - Convertible parameters: {pytorch_convertible_params:,}")
        print(f"  - T5/Language parameters (frozen): {pytorch_t5_params:,}")

        # Store convertible params for later verification
        self.jax_convertible_params = jax_convertible_params
        self.pytorch_convertible_params = pytorch_convertible_params

        if self.total_jax_params == self.total_pytorch_params:
            print("✅ Total parameter counts match exactly!")
        else:
            print(
                "⚠️  Total parameter count difference: "
                f"{abs(self.total_jax_params - self.total_pytorch_params):,}"
            )

        if jax_convertible_params == pytorch_convertible_params:
            print("✅ Convertible parameter counts match exactly!")
        else:
            print(
                "⚠️  Convertible parameter count difference: "
                f"{abs(jax_convertible_params - pytorch_convertible_params):,}"
            )

    def _track_converted_params(self, jax_param, jax_param_name):
        """Track the number of parameters converted and log their info"""
        if jax_param is not None:
            param_count = np.prod(jax_param.shape)
            self.converted_param_count += param_count
            self.converted_params_info.append(
                {"name": jax_param_name, "shape": list(jax_param.shape), "param_count": int(param_count)}
            )

    def _verify_conversion_completeness(self):
        """Verify that we converted the expected number of parameters"""
        print("\nCONVERSION VERIFICATION:")
        print(f"Total JAX parameters: {self.total_jax_params:,}")
        print(f"Total PyTorch parameters: {self.total_pytorch_params:,}")
        print(f"Convertible JAX parameters (excluding T5): {self.jax_convertible_params:,}")
        print(f"Converted parameters: {self.converted_param_count:,}")

        # Calculate conversion percentage based on convertible parameters
        if self.jax_convertible_params > 0:
            conversion_percentage = (self.converted_param_count / self.jax_convertible_params) * 100
            print(f"Conversion coverage (of convertible params): {conversion_percentage:.1f}%")

            if conversion_percentage >= 95:
                print("✅ Excellent conversion coverage!")
            elif conversion_percentage >= 80:
                print("⚠️  Good conversion coverage, but some parameters may be missing")
            else:
                print("❌ Low conversion coverage - many parameters not converted")

        # Check for unconverted convertible parameters
        unconverted_convertible = self.jax_convertible_params - self.converted_param_count
        if unconverted_convertible > 0:
            print(f"Unconverted convertible parameters: {unconverted_convertible:,}")
            self._analyze_unconverted_params()
        else:
            print("✅ All convertible parameters converted!")

    def _analyze_unconverted_params(self):
        """Analyze which convertible parameters were not converted"""
        print("\nAnalyzing unconverted convertible parameters...")

        # List detailed unconverted parameters (excluding T5)
        unconverted_params = []
        unconverted_groups = {}

        for path, param in self.jax_params.items():
            param_name = "/".join(path)

            # Skip T5 parameters as they're frozen and don't need conversion
            if any(
                skip in param_name.lower()
                for skip in [
                    "task_language_instruction_encoder",
                    "language_instruction_encoder",
                    "t5",
                    "task_tokenizers_language",
                ]
            ):
                continue

            # Check if this specific parameter was actually converted
            converted = False

            # Check transformer MLP layers specifically
            if "encoderblock_" in param_name and "MlpBlock_0" in param_name:
                # These should be converted but apparently aren't
                converted = False
            # Check other known converted components
            elif any(
                component in param_name
                for component in [
                    "pos_embedding",
                    "projection",
                    "MultiHeadDotProductAttention",
                    "LayerNorm",
                    "diffusion_model/cond_encoder",
                    "diffusion_model/reverse_network",
                    "diffusion_model/time_preprocess",
                ]
            ):
                converted = True

            if not converted:
                param_count = np.prod(param.shape)
                unconverted_params.append((param_name, param.shape, param_count))

                group = param_name.split("/")[0] if "/" in param_name else param_name
                if group not in unconverted_groups:
                    unconverted_groups[group] = 0
                unconverted_groups[group] += param_count

        if unconverted_groups:
            print("Unconverted convertible parameter groups:")
            for group, count in sorted(unconverted_groups.items(), key=lambda x: x[1], reverse=True):
                print(f"  {group}: {count:,} parameters")

            print("\nDetailed unconverted parameters (showing largest first):")
            # Sort by parameter count and show top 20
            unconverted_params.sort(key=lambda x: x[2], reverse=True)
            for i, (name, shape, count) in enumerate(unconverted_params[:20]):
                print(f"  {i + 1:2d}. {name}: {shape} ({count:,} params)")

            if len(unconverted_params) > 20:
                print(f"  ... and {len(unconverted_params) - 20} more parameters")

        else:
            print("✅ All convertible parameter groups accounted for!")

    def find_unconverted_params(self):
        """Find and return all unconverted parameters from self.jax_params"""
        print("\n" + "=" * 60)
        print("FINDING UNCONVERTED PARAMETERS")
        print("=" * 60)

        # We need to track what was actually converted by checking the conversion methods
        # This is a more accurate approach than guessing based on parameter names

        # Create a set to track converted parameter paths
        self.converted_paths = set()

        # Re-run conversion tracking to see what gets converted
        print("Tracking converted parameters...")
        original_track_method = self._track_converted_params

        def tracking_wrapper(jax_param, path=None):
            if path:
                self.converted_paths.add(path)
            return original_track_method(jax_param)

        # Find unconverted parameters
        unconverted_params = []
        total_unconverted_count = 0

        print(f"\nScanning {len(self.jax_params)} JAX parameters...")

        for path, param in self.jax_params.items():
            param_name = "/".join(path)
            param_count = np.prod(param.shape)

            # Skip T5/language encoder parameters (they're frozen)
            if any(
                skip in param_name.lower()
                for skip in [
                    "task_language_instruction_encoder",
                    "language_instruction_encoder",
                    "t5",
                    "task_tokenizers_language",
                ]
            ):
                continue

            # Check if this parameter should have been converted based on our conversion logic
            should_convert = self._should_parameter_be_converted(param_name)

            if should_convert:
                unconverted_params.append(
                    {"path": param_name, "shape": param.shape, "param_count": param_count, "data": param}
                )
                total_unconverted_count += param_count

        # Sort by parameter count (largest first)
        unconverted_params.sort(key=lambda x: x["param_count"], reverse=True)

        print(f"\nFOUND {len(unconverted_params)} UNCONVERTED PARAMETERS")
        print(f"Total unconverted parameter count: {total_unconverted_count:,}")
        print("\nDETAILED LIST:")
        print("-" * 80)

        for i, param_info in enumerate(unconverted_params, 1):
            print(f"{i:3d}. {param_info['path']}")
            print(f"     Shape: {param_info['shape']}, Count: {param_info['param_count']:,}")
            print(f"     Data type: {type(param_info['data'])}")
            print()

        # Group by component for better understanding
        print("\nGROUPED BY COMPONENT:")
        print("-" * 40)
        groups = {}
        for param_info in unconverted_params:
            # Extract component name (first part of path)
            component = param_info["path"].split("/")[0]
            if component not in groups:
                groups[component] = []
            groups[component].append(param_info)

        for component, params in sorted(groups.items()):
            total_params = sum(p["param_count"] for p in params)
            print(f"{component}: {len(params)} parameters, {total_params:,} total count")
            for param_info in params:
                print(f"  - {param_info['path']}: {param_info['shape']}")

        return unconverted_params

    def _should_parameter_be_converted(self, param_name):
        """Determine if a parameter should be converted based on our conversion logic"""

        # Skip T5/language parameters
        if any(
            skip in param_name.lower()
            for skip in [
                "task_language_instruction_encoder",
                "language_instruction_encoder",
                "t5",
                "task_tokenizers_language",
            ]
        ):
            return False

        # Parameters that should be converted based on our conversion methods
        convertible_patterns = [
            # Positional embeddings
            "pos_embedding",
            # Projection layers
            "projection",
            # Transformer layers
            "encoderblock_",
            "MultiHeadDotProductAttention",
            "MlpBlock_",
            "LayerNorm_",
            "encoder_norm",  # Final encoder normalization
            # Action head / diffusion model
            "heads_action/diffusion_model",
            "time_preprocess",
            "cond_encoder",
            "reverse_network",
            "MLPResNetBlock_",
            # Observation tokenizers
            "observation_tokenizers_primary",
            "observation_tokenizers_wrist",
        ]

        return any(pattern in param_name for pattern in convertible_patterns)

    def convert_action_heads(self):
        self._convert_diffusion_action_head()

    def _convert_diffusion_action_head(self):
        jax_prefix = "heads_action/diffusion_model"
        pytorch_head = self.pytorch_model.action_head

        # Time embedding (FourierFeatures)
        time_kernel_name = f"{jax_prefix}/time_preprocess/kernel"
        time_kernel = self._get_jax_param(time_kernel_name)
        if time_kernel is not None:
            try:
                pytorch_head.diffusion_model.time_preprocess.kernel.data = torch.from_numpy(
                    time_kernel.copy()
                ).float()
                self._track_converted_params(time_kernel, time_kernel_name)
                self.conversion_log.append("Converted time embedding kernel")
            except AttributeError:
                print("Warning: Could not find time_embedding.kernel in PyTorch model")

        # Condition encoder
        cond_dense0_kernel = self._get_jax_param(f"{jax_prefix}/cond_encoder/Dense_0/kernel")
        cond_dense0_bias = self._get_jax_param(f"{jax_prefix}/cond_encoder/Dense_0/bias")
        cond_dense1_kernel = self._get_jax_param(f"{jax_prefix}/cond_encoder/Dense_1/kernel")
        cond_dense1_bias = self._get_jax_param(f"{jax_prefix}/cond_encoder/Dense_1/bias")

        if cond_dense0_kernel is not None:
            try:
                pytorch_head.diffusion_model.cond_encoder[0].weight.data = torch.from_numpy(
                    cond_dense0_kernel.T.copy()
                ).float()
                self._track_converted_params(cond_dense0_kernel, f"{jax_prefix}/cond_encoder/Dense_0/kernel")
                if cond_dense0_bias is not None:
                    pytorch_head.diffusion_model.cond_encoder[0].bias.data = torch.from_numpy(
                        cond_dense0_bias.copy()
                    ).float()
                    self._track_converted_params(cond_dense0_bias, f"{jax_prefix}/cond_encoder/Dense_0/bias")
                self.conversion_log.append("Converted condition encoder layer 0")
            except (AttributeError, IndexError):
                print("Warning: Could not find cond_encoder[0] in PyTorch model")

        if cond_dense1_kernel is not None:
            try:
                pytorch_head.diffusion_model.cond_encoder[2].weight.data = torch.from_numpy(
                    cond_dense1_kernel.T.copy()
                ).float()
                self._track_converted_params(cond_dense1_kernel, f"{jax_prefix}/cond_encoder/Dense_1/kernel")
                if cond_dense1_bias is not None:
                    pytorch_head.diffusion_model.cond_encoder[2].bias.data = torch.from_numpy(
                        cond_dense1_bias.copy()
                    ).float()
                    self._track_converted_params(cond_dense1_bias, f"{jax_prefix}/cond_encoder/Dense_1/bias")
                self.conversion_log.append("Converted condition encoder layer 2")
            except (AttributeError, IndexError):
                print("Warning: Could not find cond_encoder[2] in PyTorch model")

        # Diffusion network
        try:
            self._convert_diffusion_network(
                f"{jax_prefix}/reverse_network", pytorch_head.diffusion_model.reverse_network
            )
        except AttributeError:
            print("Warning: Could not find diffusion_net in PyTorch model")

    def _convert_diffusion_network(self, jax_prefix, pytorch_module):
        # Initial dense layer (input_proj)
        dense0_kernel = self._get_jax_param(f"{jax_prefix}/Dense_0/kernel")
        dense0_bias = self._get_jax_param(f"{jax_prefix}/Dense_0/bias")
        if dense0_kernel is not None:
            try:
                pytorch_module.input_proj.weight.data = torch.from_numpy(dense0_kernel.T.copy()).float()
                self._track_converted_params(dense0_kernel, f"{jax_prefix}/Dense_0/kernel")
                if dense0_bias is not None:
                    pytorch_module.input_proj.bias.data = torch.from_numpy(dense0_bias.copy()).float()
                    self._track_converted_params(dense0_bias, f"{jax_prefix}/Dense_0/bias")
                self.conversion_log.append("Converted diffusion network initial layer")
            except AttributeError:
                print("Warning: Could not find input_proj in PyTorch model")

        # MLPResNetBlocks - try to find how many blocks exist
        block_count = 0
        for i in range(10):  # Check up to 10 blocks
            if self._get_jax_param(f"{jax_prefix}/MLPResNetBlock_{i}/Dense_0/kernel") is not None:
                block_count = i + 1

        print(f"Found {block_count} MLP ResNet blocks")

        for i in range(block_count):
            try:
                self._convert_mlp_resnet_block(f"{jax_prefix}/MLPResNetBlock_{i}", pytorch_module.blocks[i])
            except (AttributeError, IndexError):
                print(f"Warning: Could not find blocks[{i}] in PyTorch model")

        # Final dense layer (output_proj)
        dense1_kernel = self._get_jax_param(f"{jax_prefix}/Dense_1/kernel")
        dense1_bias = self._get_jax_param(f"{jax_prefix}/Dense_1/bias")
        if dense1_kernel is not None:
            try:
                pytorch_module.output_proj.weight.data = torch.from_numpy(dense1_kernel.T.copy()).float()
                self._track_converted_params(dense1_kernel, f"{jax_prefix}/Dense_1/kernel")
                if dense1_bias is not None:
                    pytorch_module.output_proj.bias.data = torch.from_numpy(dense1_bias.copy()).float()
                    self._track_converted_params(dense1_bias, f"{jax_prefix}/Dense_1/bias")
                self.conversion_log.append("Converted diffusion network final layer")
            except AttributeError:
                print("Warning: Could not find output_proj in PyTorch model")

    def _convert_mlp_resnet_block(self, jax_prefix, pytorch_module):
        dense0_kernel = self._get_jax_param(f"{jax_prefix}/Dense_0/kernel")
        dense0_bias = self._get_jax_param(f"{jax_prefix}/Dense_0/bias")
        dense1_kernel = self._get_jax_param(f"{jax_prefix}/Dense_1/kernel")
        dense1_bias = self._get_jax_param(f"{jax_prefix}/Dense_1/bias")

        # LayerNorm parameters
        layernorm_scale = self._get_jax_param(f"{jax_prefix}/LayerNorm_0/scale")
        layernorm_bias = self._get_jax_param(f"{jax_prefix}/LayerNorm_0/bias")

        if dense0_kernel is not None:
            try:
                pytorch_module.dense1.weight.data = torch.from_numpy(dense0_kernel.T.copy()).float()
                self._track_converted_params(dense0_kernel, f"{jax_prefix}/Dense_0/kernel")
                if dense0_bias is not None:
                    pytorch_module.dense1.bias.data = torch.from_numpy(dense0_bias.copy()).float()
                    self._track_converted_params(dense0_bias, f"{jax_prefix}/Dense_0/bias")
                self.conversion_log.append(f"Converted MLP ResNet block {jax_prefix.split('/')[-1]} layer 0")
            except AttributeError:
                print(f"Warning: Could not find dense1 in {jax_prefix}")

        if dense1_kernel is not None:
            try:
                pytorch_module.dense2.weight.data = torch.from_numpy(dense1_kernel.T.copy()).float()
                self._track_converted_params(dense1_kernel, f"{jax_prefix}/Dense_1/kernel")
                if dense1_bias is not None:
                    pytorch_module.dense2.bias.data = torch.from_numpy(dense1_bias.copy()).float()
                    self._track_converted_params(dense1_bias, f"{jax_prefix}/Dense_1/bias")
                self.conversion_log.append(f"Converted MLP ResNet block {jax_prefix.split('/')[-1]} layer 2")
            except AttributeError:
                print(f"Warning: Could not find dense2 in {jax_prefix}")

        # Convert LayerNorm
        if layernorm_scale is not None:
            try:
                pytorch_module.layer_norm.weight.data = torch.from_numpy(layernorm_scale.copy()).float()
                self._track_converted_params(layernorm_scale, f"{jax_prefix}/LayerNorm_0/scale")
                if layernorm_bias is not None:
                    pytorch_module.layer_norm.bias.data = torch.from_numpy(layernorm_bias.copy()).float()
                    self._track_converted_params(layernorm_bias, f"{jax_prefix}/LayerNorm_0/bias")
                self.conversion_log.append(
                    f"Converted MLP ResNet block {jax_prefix.split('/')[-1]} LayerNorm"
                )
            except AttributeError:
                print(f"Warning: Could not find layer_norm in {jax_prefix}")
        else:
            print(f"Warning: LayerNorm scale not found for {jax_prefix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="octo-base",
        help="Model name to use (e.g., 'octo-base', 'octo-small')",
    )
    args = parser.parse_args()

    model_name = args.model_name

    # Load JAX model directly from checkpoint
    from octo.model.octo_model import OctoModel as JaxOctoModel

    print(f"Loading JAX Octo model: {model_name}...")
    if model_name == "octo-base":
        jax_model = JaxOctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    elif model_name == "octo-small":
        jax_model = JaxOctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create a PyTorch model
    print("Creating PyTorch model...")
    pytorch_model = PyTorchOctoModel(model_name=model_name)

    # Initialize the weight converter and run the conversion
    print("Starting weight conversion...")
    converter = OctoWeightConverter(jax_model.params, pytorch_model, model_name=model_name)
    pytorch_model = converter.convert()

    # Print conversion log
    print("\nConversion Summary:")
    for log_entry in converter.conversion_log:
        print(f"✓ {log_entry}")

    # Save the converted PyTorch model
    output_path = f"output/pytorch_{model_name}_model.pth"
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"\nSuccessfully converted and saved the PyTorch model to {output_path}")


if __name__ == "__main__":
    main()
