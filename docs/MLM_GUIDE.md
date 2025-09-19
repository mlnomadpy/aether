# MLM (Masked Language Modeling) Support in Aether

Aether now supports both Causal Language Modeling (CLM) and Masked Language Modeling (MLM) training modes.

## Overview

### Causal Language Modeling (CLM)
- **Traditional approach**: Predicts the next token in a sequence
- **Use case**: Generative models like GPT
- **Training**: Model sees tokens 1..n-1 and predicts tokens 2..n

### Masked Language Modeling (MLM)
- **BERT-style approach**: Predicts masked tokens in a sequence
- **Use case**: Bidirectional representations, understanding tasks
- **Training**: Some tokens are masked, model predicts the original tokens

## Configuration

### Enable MLM Mode

```yaml
training:
  training_mode: "mlm"  # Set to "mlm" for masked LM, "clm" for causal LM
  mlm_mask_prob: 0.15   # Probability of masking each token (default: 0.15)
  mlm_replace_prob: 0.8 # Prob of replacing with [MASK] token (default: 0.8)
  mlm_random_prob: 0.1  # Prob of replacing with random token (default: 0.1)
  final_evaluation: true # Run evaluation at end of training
```

### MLM Masking Strategy

For each token selected for masking (with probability `mlm_mask_prob`):
- 80% of the time: Replace with `[MASK]` token
- 10% of the time: Replace with a random token  
- 10% of the time: Keep the original token

This follows the original BERT masking strategy.

## Usage Examples

### Command Line

```bash
# Train with MLM using default settings
python train.py --model minigpt-linear

# Train with MLM using custom config
python train.py --config configs/mlm_example.yaml

# Train with CLM for comparison
python train.py --config configs/clm_example.yaml

# Run the MLM demo
python demo_mlm.py --mode mlm

# Run CLM demo for comparison
python demo_mlm.py --mode clm
```

### Python API

```python
from aether import Config, Trainer

# Create MLM configuration
config = Config()
config.training.training_mode = "mlm"
config.training.mlm_mask_prob = 0.15
config.training.final_evaluation = True

# Create and run trainer
trainer = Trainer(config)
trainer.train()
```

## Training Process

### MLM Training Loop
1. Tokens are randomly masked according to the masking strategy
2. Model receives masked sequence as input
3. Loss is computed only on masked positions
4. Model learns to predict original tokens from context

### Final Evaluation
- When `final_evaluation: true`, comprehensive evaluation runs at training end
- Processes entire validation set (not just `eval_steps` batches)
- Reports final validation loss and metrics

## Differences from CLM

| Aspect | CLM | MLM |
|--------|-----|-----|
| **Input** | [BOS, tok1, tok2, ..., tokN] | [tok1, [MASK], tok3, ..., tokN] |
| **Target** | [tok1, tok2, ..., tokN, EOS] | Original tokens at masked positions |
| **Loss** | All positions except first | Only masked positions |
| **Attention** | Causal (can't see future) | Bidirectional (can see all) |

## Model Architecture

The same transformer models work for both CLM and MLM:
- No architectural changes needed
- Attention masks are handled automatically
- MLM uses bidirectional attention (no causal masking)

## Best Practices

### MLM Training
- Use lower learning rates (e.g., 1e-4 vs 2e-3 for CLM)
- AdamW optimizer often works better than Novograd
- Consider warmup + cosine decay schedule
- Mask probability of 0.15 is standard (BERT default)

### Tokenizer Considerations
- GPT-2 tokenizer doesn't have a `[MASK]` token by default
- Aether automatically handles this by using the last vocabulary token
- For better results, consider using a tokenizer with explicit mask tokens

## Example Results

After training, you should see:
- MLM models learn bidirectional representations
- Good performance on masked token prediction
- Final evaluation provides comprehensive loss metrics

## Troubleshooting

### Common Issues

1. **No masked tokens in batch**: Rare edge case handled gracefully
2. **Memory usage**: MLM uses slightly more memory due to mask labels
3. **Convergence**: MLM may need different hyperparameters than CLM

### Performance Tips

- Start with the provided example configurations
- Monitor both training and validation loss
- Use final evaluation to get comprehensive metrics
- Consider the masking probability based on your use case