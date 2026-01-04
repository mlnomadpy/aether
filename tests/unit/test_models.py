"""Unit tests for models."""

import pytest
import jax.numpy as jnp
import flax.nnx as nnx
from aether.models import MiniGPT, TransformerBlock, TokenAndPositionEmbedding
from aether.models import RMSNorm, TokenOnlyEmbedding


def test_token_position_embedding():
    """Test token and position embedding layer."""
    rngs = nnx.Rngs(42)
    embedding = TokenAndPositionEmbedding(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        rngs=rngs
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = embedding(inputs)
    
    assert outputs.shape == (1, 5, 256)


def test_rms_norm():
    """Test RMSNorm layer."""
    rngs = nnx.Rngs(42)
    norm = RMSNorm(dim=256, rngs=rngs)
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = norm(inputs)
    
    assert outputs.shape == inputs.shape
    # RMSNorm should normalize values
    # With all ones input and unit weights, output should be normalized


def test_token_only_embedding():
    """Test TokenOnlyEmbedding layer."""
    rngs = nnx.Rngs(42)
    embedding = TokenOnlyEmbedding(
        vocab_size=1000,
        embed_dim=256,
        rngs=rngs
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = embedding(inputs)
    
    assert outputs.shape == (1, 5, 256)


def test_transformer_block_linear():
    """Test transformer block with linear architecture."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear"
    )
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape


def test_transformer_block_linear_without_norm():
    """Test transformer block with linear architecture without layer normalization."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear",
        use_layer_norm=False
    )
    
    # Verify that layer norm layers are None
    assert block.layer_norm1 is None
    assert block.layer_norm2 is None
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape


def test_transformer_block_linear_with_norm():
    """Test transformer block with linear architecture with layer normalization."""
    rngs = nnx.Rngs(42)
    block = TransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        rngs=rngs,
        architecture="linear",
        use_layer_norm=True
    )
    
    # Verify that layer norm layers are created
    assert block.layer_norm1 is not None
    assert block.layer_norm2 is not None
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=True)
    
    assert outputs.shape == inputs.shape


def test_minigpt_linear():
    """Test MiniGPT model with linear architecture."""
    rngs = nnx.Rngs(42)
    model = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear"
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = model(inputs, training=True)
    
    assert outputs.shape == (1, 5, 1000)  # (batch_size, seq_len, vocab_size)


def test_minigpt_config():
    """Test MiniGPT configuration methods."""
    rngs = nnx.Rngs(42)
    config = {
        "maxlen": 128,
        "vocab_size": 1000,
        "embed_dim": 256,
        "num_heads": 4,
        "feed_forward_dim": 512,
        "num_transformer_blocks": 2,
        "architecture": "linear"
    }
    
    model = MiniGPT.from_config(config, rngs)
    retrieved_config = model.get_config()
    
    # Check that configurations match
    for key, value in config.items():
        assert retrieved_config[key] == value


def test_minigpt_attention_block_reuse():
    """Test MiniGPT with attention block reuse functionality."""
    rngs = nnx.Rngs(42)
    
    # Test with reuse = 1 (default, no reuse)
    model1 = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        architecture="linear",
        attention_block_reuse=1
    )
    
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    outputs1 = model1(inputs, training=True)
    assert outputs1.shape == (1, 5, 1000)
    
    # Test with reuse = 3
    rngs2 = nnx.Rngs(42)
    model2 = MiniGPT(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs2,
        architecture="linear",
        attention_block_reuse=3
    )
    
    outputs2 = model2(inputs, training=True)
    assert outputs2.shape == (1, 5, 1000)
    
    # Verify config contains attention_block_reuse
    config = model2.get_config()
    assert "attention_block_reuse" in config
    assert config["attention_block_reuse"] == 3


# --- Me3za Architecture Tests ---

def _nmn_available():
    """Check if nmn package is available."""
    try:
        from nmn.nnx.attention import RotaryYatAttention
        from nmn.nnx.nmn import YatNMN
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_model():
    """Test Me3za model with Rotary YAT Performer attention."""
    from aether.models import Me3za
    
    rngs = nnx.Rngs(42)
    model = Me3za(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        use_performer=True,
        dropout_rate=0.1
    )
    
    # Test forward pass
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
    outputs = model(inputs, training=False)
    
    assert outputs.shape == (1, 5, 1000)  # (batch_size, seq_len, vocab_size)


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_embed():
    """Test Me3za embed method for sentence embeddings."""
    from aether.models import Me3za
    
    rngs = nnx.Rngs(42)
    model = Me3za(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs
    )
    
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    embeddings = model.embed(inputs, training=False)
    
    assert embeddings.shape == (1, 5, 256)  # (batch_size, seq_len, embed_dim)


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_config():
    """Test Me3za configuration methods."""
    from aether.models import Me3za
    
    rngs = nnx.Rngs(42)
    config = {
        "maxlen": 128,
        "vocab_size": 1000,
        "embed_dim": 256,
        "num_heads": 4,
        "feed_forward_dim": 512,
        "num_transformer_blocks": 2,
        "use_performer": True,
        "dropout_rate": 0.1
    }
    
    model = Me3za.from_config(config, rngs)
    retrieved_config = model.get_config()
    
    # Check that configurations match
    assert retrieved_config['maxlen'] == config['maxlen']
    assert retrieved_config['vocab_size'] == config['vocab_size']
    assert retrieved_config['embed_dim'] == config['embed_dim']
    assert retrieved_config['num_heads'] == config['num_heads']
    assert retrieved_config['feed_forward_dim'] == config['feed_forward_dim']
    assert retrieved_config['num_transformer_blocks'] == config['num_transformer_blocks']
    assert retrieved_config['architecture'] == 'me3za'


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_transformer_block():
    """Test Me3zaTransformerBlock with Rotary YAT attention."""
    from aether.models import Me3zaTransformerBlock
    
    rngs = nnx.Rngs(42)
    block = Me3zaTransformerBlock(
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        maxlen=128,
        rngs=rngs,
        use_performer=True
    )
    
    # Test forward pass
    inputs = jnp.ones((2, 10, 256))  # batch_size=2, seq_len=10, embed_dim=256
    outputs = block(inputs, training=False)
    
    assert outputs.shape == inputs.shape


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_without_performer():
    """Test Me3za model without Performer mode (standard attention)."""
    from aether.models import Me3za
    
    rngs = nnx.Rngs(42)
    model = Me3za(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        use_performer=False,  # Standard quadratic attention
        dropout_rate=0.1
    )
    
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(inputs, training=False)
    
    assert outputs.shape == (1, 5, 1000)


@pytest.mark.skipif(not _nmn_available(), reason="nmn package not available")
def test_me3za_training_mode():
    """Test Me3za model in training mode."""
    from aether.models import Me3za
    
    rngs = nnx.Rngs(42)
    model = Me3za(
        maxlen=128,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        feed_forward_dim=512,
        num_transformer_blocks=2,
        rngs=rngs,
        dropout_rate=0.1
    )
    
    inputs = jnp.array([[1, 2, 3, 4, 5]])
    
    # Run in training mode (dropout active)
    outputs_train = model(inputs, training=True)
    assert outputs_train.shape == (1, 5, 1000)
    
    # Run in inference mode (dropout inactive)
    outputs_eval = model(inputs, training=False)
    assert outputs_eval.shape == (1, 5, 1000)


if __name__ == "__main__":
    pytest.main([__file__])