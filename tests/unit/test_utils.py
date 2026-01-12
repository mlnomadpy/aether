"""Unit tests for utilities module."""

import pytest
import sys
import os
sys.path.insert(0, '/home/runner/work/aether/aether')

import jax.numpy as jnp
import jax


class TestCausalAttentionMask:
    """Tests for causal_attention_mask function."""
    
    def test_causal_mask_shape(self):
        """Test that causal mask has correct shape."""
        from aether.utils.attention import causal_attention_mask
        
        seq_len = 10
        mask = causal_attention_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
    
    def test_causal_mask_is_lower_triangular(self):
        """Test that causal mask is lower triangular."""
        from aether.utils.attention import causal_attention_mask
        
        seq_len = 5
        mask = causal_attention_mask(seq_len)
        
        # Upper triangular part should be zeros (excluding diagonal)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # Above diagonal
                    assert mask[i, j] == 0.0
                else:  # On or below diagonal
                    assert mask[i, j] == 1.0
    
    def test_causal_mask_diagonal_is_ones(self):
        """Test that diagonal elements are ones."""
        from aether.utils.attention import causal_attention_mask
        
        seq_len = 8
        mask = causal_attention_mask(seq_len)
        
        for i in range(seq_len):
            assert mask[i, i] == 1.0
    
    def test_causal_mask_different_sizes(self):
        """Test causal mask with different sequence lengths."""
        from aether.utils.attention import causal_attention_mask
        
        for seq_len in [1, 2, 4, 16, 64, 128]:
            mask = causal_attention_mask(seq_len)
            assert mask.shape == (seq_len, seq_len)
            # Verify it's still lower triangular
            assert jnp.allclose(mask, jnp.tril(jnp.ones((seq_len, seq_len))))
    
    def test_causal_mask_sum(self):
        """Test that the sum of the mask equals expected value."""
        from aether.utils.attention import causal_attention_mask
        
        seq_len = 5
        mask = causal_attention_mask(seq_len)
        
        # Sum should equal n*(n+1)/2 for lower triangular matrix of ones
        expected_sum = seq_len * (seq_len + 1) / 2
        assert jnp.sum(mask) == expected_sum


class TestDeviceSetup:
    """Tests for device setup utilities."""
    
    def test_setup_mesh_import(self):
        """Test that setup_mesh can be imported."""
        from aether.utils.device import setup_mesh
        assert callable(setup_mesh)
    
    def test_setup_mesh_returns_mesh(self):
        """Test that setup_mesh returns a JAX Mesh object."""
        from aether.utils.device import setup_mesh
        from jax.sharding import Mesh
        
        mesh = setup_mesh()
        assert isinstance(mesh, Mesh)
    
    def test_setup_mesh_with_custom_shape(self):
        """Test setup_mesh with custom mesh shape."""
        from aether.utils.device import setup_mesh
        from jax.sharding import Mesh
        
        # Use a shape that matches available devices
        num_devices = len(jax.devices())
        mesh_shape = (num_devices, 1)
        
        mesh = setup_mesh(mesh_shape=mesh_shape)
        assert isinstance(mesh, Mesh)
    
    def test_setup_mesh_axis_names(self):
        """Test that mesh has correct axis names."""
        from aether.utils.device import setup_mesh
        
        mesh = setup_mesh()
        assert 'batch' in mesh.axis_names
        assert 'model' in mesh.axis_names


def _can_access_network():
    """Check if network is available for tokenizer tests."""
    try:
        import tiktoken
        tiktoken.get_encoding("gpt2")
        return True
    except Exception:
        return False


class TestTokenizer:
    """Tests for tokenizer utilities."""
    
    def test_get_tokenizer_import(self):
        """Test that get_tokenizer can be imported."""
        from aether.utils.tokenizer import get_tokenizer
        assert callable(get_tokenizer)
    
    @pytest.mark.skipif(not _can_access_network(), reason="Network not available")
    def test_get_tokenizer_default(self):
        """Test get_tokenizer with default name."""
        from aether.utils.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer()
        assert tokenizer is not None
    
    @pytest.mark.skipif(not _can_access_network(), reason="Network not available")
    def test_get_tokenizer_gpt2(self):
        """Test get_tokenizer with gpt2 name."""
        from aether.utils.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer("gpt2")
        assert tokenizer is not None
    
    @pytest.mark.skipif(not _can_access_network(), reason="Network not available")
    def test_tokenizer_encode(self):
        """Test that tokenizer can encode text."""
        from aether.utils.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer("gpt2")
        tokens = tokenizer.encode("Hello, world!")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    @pytest.mark.skipif(not _can_access_network(), reason="Network not available")
    def test_tokenizer_decode(self):
        """Test that tokenizer can decode tokens."""
        from aether.utils.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer("gpt2")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text


class TestUtilsModuleExports:
    """Tests for utils module exports."""
    
    def test_utils_exports_setup_mesh(self):
        """Test that utils module exports setup_mesh."""
        from aether.utils import setup_mesh
        assert callable(setup_mesh)
    
    def test_utils_exports_get_tokenizer(self):
        """Test that utils module exports get_tokenizer."""
        from aether.utils import get_tokenizer
        assert callable(get_tokenizer)
    
    def test_utils_exports_causal_attention_mask(self):
        """Test that utils module exports causal_attention_mask."""
        from aether.utils import causal_attention_mask
        assert callable(causal_attention_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
