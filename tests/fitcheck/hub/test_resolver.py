"""Tests for HuggingFace Hub model resolver.

Tests use a mock config dict to avoid network calls. The nightly
accuracy test (test_meta_device_accuracy) validates against real
models loaded on meta device.
"""

import pytest
from fitcheck.hub.resolver import resolve_from_config, _compute_param_count


# Real config.json values for Llama-3.1-8B
LLAMA_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}

# Real config.json values for Mixtral-8x7B-v0.1
MIXTRAL_CONFIG = {
    "architectures": ["MixtralForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}


class TestResolveFromConfig:
    """Config dict -> ModelProfile conversion."""

    def test_llama_8b_dimensions(self):
        profile = resolve_from_config("meta-llama/Llama-3.1-8B", LLAMA_8B_CONFIG)
        assert profile.family == "llama"
        assert profile.hidden_size == 4096
        assert profile.num_layers == 32
        assert profile.num_kv_heads == 8
        assert profile.vocab_size == 128256
        assert not profile.is_moe

    def test_llama_8b_head_dim(self):
        profile = resolve_from_config("meta-llama/Llama-3.1-8B", LLAMA_8B_CONFIG)
        assert profile.head_dim == 128  # 4096 / 32

    def test_llama_8b_param_count_in_range(self):
        """Computed param count should be within 5% of the known ~8.03B."""
        profile = resolve_from_config("meta-llama/Llama-3.1-8B", LLAMA_8B_CONFIG)
        assert 7.5e9 < profile.total_params < 8.5e9

    def test_mixtral_detected_as_moe(self):
        profile = resolve_from_config("mistralai/Mixtral-8x7B-v0.1", MIXTRAL_CONFIG)
        assert profile.is_moe
        assert profile.num_experts == 8
        assert profile.num_experts_per_token == 2
        assert profile.family == "moe"

    def test_mixtral_param_count_includes_all_experts(self):
        """Mixtral 8x7B total params should be ~46-47B (all 8 experts)."""
        profile = resolve_from_config("mistralai/Mixtral-8x7B-v0.1", MIXTRAL_CONFIG)
        assert 40e9 < profile.total_params < 50e9

    def test_unsupported_architecture_raises(self):
        bad_config = {"architectures": ["T5ForConditionalGeneration"]}
        with pytest.raises(ValueError, match="Unsupported architecture"):
            resolve_from_config("google/t5-base", bad_config)


class TestParamCounting:
    """Parameter count formulas produce correct values."""

    def test_dense_model_formula(self):
        """Simple 1-layer dense model: verify each component."""
        params = _compute_param_count(
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=4,  # MHA (no GQA)
            intermediate_size=256,
            vocab_size=1000,
            num_experts=None,
        )
        # Embedding: 1000 * 64 = 64000
        # LM head: 1000 * 64 = 64000
        # Attention: Q(64*64) + K(64*64) + V(64*64) + O(64*64) = 16384
        # MLP: gate(64*256) + up(64*256) + down(256*64) = 49152
        # Norms: 2 * 64 = 128
        # Final norm: 64
        expected = 64000 + 64000 + 16384 + 49152 + 128 + 64
        assert params == expected

    def test_gqa_reduces_kv_projection_params(self):
        """GQA with fewer KV heads should have fewer params than MHA."""
        mha = _compute_param_count(256, 1, 8, 8, 1024, 1000, None)
        gqa = _compute_param_count(256, 1, 8, 2, 1024, 1000, None)
        assert gqa < mha  # Fewer K,V projection params

    def test_moe_has_more_params_than_dense(self):
        """MoE with 8 experts should have ~8x the MLP params."""
        dense = _compute_param_count(256, 1, 8, 8, 1024, 1000, None)
        moe = _compute_param_count(256, 1, 8, 8, 1024, 1000, num_experts=8)
        # The MLP is replicated 8x, plus router weights
        assert moe > dense * 3  # Much larger due to expert replication
