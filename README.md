# Automated Gabliteration Optimizer

<table>
  <tr>
    <td align="left" width="30%">
      <img src="gabliteration-logo.jpg" alt="logo" width="100%"/>
    </td>
    <td align="left" width="70%" valign="top">
      <p><strong>Automated hyperparameter search for optimal Gabliteration configurations.</strong></p>
      <p><strong>Paper:</strong> <a href="https://arxiv.org/abs/2512.18901">Gabliteration: Adaptive Multi-Directional Neural Weight Modification</a></p>
      <p><strong>Author:</strong> Gökdeniz Gülmez (2025)</p>
    </td>
  </tr>
</table>

## Overview

This script automates the process of finding optimal Gabliteration parameters by:
1. **Automatically loading datasets** from HuggingFace (mlabonne/harmful_behaviors and mlabonne/harmless_alpaca)
2. Testing multiple random parameter configurations
3. Evaluating each configuration's effectiveness (refusal rate reduction)
4. Measuring model similarity to original (KL divergence)
5. Ranking configurations by combined score
6. Allowing you to select and save the best version

## Quick Start

### 1. Installation

```bash
pip install gabliteration
```

This will install all dependencies and make the `gabliterate.automate` command available system-wide.

The tool automatically downloads these datasets from HuggingFace:
- `mlabonne/harmful_behaviors` - Harmful prompts for training
- `mlabonne/harmless_alpaca` - Harmless prompts for comparison

No local files needed!

### 2. Run with CLI Arguments

Test your favorite model with:

```bash
# Basic usage
gabliterate.automate --model "Nanbeige/Nanbeige4-3B-Thinking-2511"

# With custom parameters
gabliterate.automate --model "meta-llama/Llama-3.2-1B-Instruct" --num-versions 50 --batch-size 4

# Full options
gabliterate.automate --model "Qwen/Qwen3-4B-Instruct-2507" \
  --num-versions 100 \
  --test-samples 200 \
  --max-tokens 150 \
  --batch-size 4 \
  --kl-samples 15
```

**CLI Options:**
- `--model, -m` (required): Hugging Face model name or path
- `--num-versions, -n`: Number of configurations to test (default: 100)
- `--test-samples, -t`: Test samples for refusal evaluation (default: 100)
- `--max-tokens`: Max tokens to generate during evaluation (default: 100)
- `--batch-size, -b`: Batch size for evaluation (default: 2)
- `--kl-samples`: KL divergence samples (default: 10)

Run `gabliterate.automate --help` to see all options.

### 3. Review and Select

The script will:
- Test each configuration
- Print real-time results:
  ```
  Testing Version 5/10
  Config: Samples: 100, Skip: [2, 1], Layer: 0.52, Scale: 0.65, λ: 0.10, k: 2, Adaptive: True, β: 0.45
  KL Divergence: 0.0234
  Refusal Rate: 12.0% (12/100)
  Score: 1.2234
  ```

After all tests, you'll see:

```
TOP 10 BEST CONFIGURATIONS
Rank   Refusal    KL Div     Score      Config
----------------------------------------------------------------------
1      8.0%       0.0189     0.8189     Samples: 150, Skip: [2, 1], ...
2      12.0%      0.0234     1.2234     Samples: 100, Skip: [1, 2], ...
...
```

### 4. Automatic Model Saving

After all tests complete, the script automatically:
- Selects the best configuration (lowest combined score)
- Recreates and saves the gabliterated model
- Saves all configuration details in `gabliteration_config.json`
- Generates a model-specific README.md

## Output Structure

```
Qwen_Qwen3-4B-Instruct-2507-gabliterated-v1-20250102_143022/
├── config.json                      # Model config
├── model.safetensors               # Model weights
├── tokenizer.json                  # Tokenizer
├── tokenizer_config.json           # Tokenizer config
└── gabliteration_config.json       # ⭐ Gabliteration parameters & results
```

## Configuration File Format

The `gabliteration_config.json` contains:

```json
{
  "model_name": "Qwen/Qwen3-4B-Instruct-2507",
  "version_id": 1,
  "timestamp": "20250102_143022",
  "gabliteration_config": {
    "num_prompt_samples": 150,
    "skip_begin_layers": 2,
    "skip_end_layers": 1,
    "layer_fraction": 0.52,
    "base_scale_factor": 0.65,
    "regularization": 0.1,
    "n_directions": 2,
    "adaptive_layer_scale": true,
    "beta": 0.5
  },
  "results": {
    "kl_divergence": 0.0189,
    "refusal_rate": 0.08,
    "score": 0.8189
  },
  "all_results": [...]  // Full results from all tested versions
}
```

## Understanding the Metrics

### Refusal Rate
- **What**: Percentage of test prompts that trigger refusal responses
- **Lower is better**: 0% means no refusals, 100% means all prompts refused
- **Target**: Aim for <10% for effective gabliteration

### KL Divergence
- **What**: Measures how different the modified model is from the original
- **Lower is better**: Smaller values = model behaves more similarly to original
- **Target**: Keep <0.05 to preserve model quality

### Score
- **What**: Combined metric = 10×RefusalRate + KLDivergence
- **Lower is better**: Balances refusal reduction with model preservation
- **Weights refusal rate 10x more than KL**: Primary goal is reducing refusals

## Hyperparameter Ranges

The script randomly samples from these ranges:

| Parameter | Range | Paper Default | Description |
|-----------|-------|---------------|-------------|
| `num_prompt_samples` | [50, 75, 100, 150, 200] | 100 | Training samples for direction extraction |
| `skip_begin_layers` | [1, 2, 3] | 2 | Skip initial layers (preserve embeddings) |
| `skip_end_layers` | [1, 2, 3] | 1 | Skip final layers (preserve output) |
| `layer_fraction` | [0.3, 0.7] | 0.5 | Which layer to extract directions from |
| `base_scale_factor` | [0.2, 0.8] | 0.3 | Modification strength (α_base) |
| `regularization` | [0.05, 0.1, 0.15, 0.2] | 0.1 | Ridge regularization (λ) |
| `n_directions` | [1, 2, 3] | 1 | Number of refusal directions (k) |
| `adaptive_layer_scale` | [True, False] | True | Use adaptive scaling |
| `beta` | [0.3, 0.7] | 0.5 | Adaptive strength (β) |

## Advanced Usage

### Testing More Configurations

Increase the number of versions tested:

```bash
gabliterate.automate --model "Qwen/Qwen3-4B-Instruct-2507" --num-versions 200
```

### Custom Evaluation Parameters

Fine-tune evaluation settings:

```bash
gabliterate.automate --model "meta-llama/Llama-3.2-1B-Instruct" \
  --test-samples 300 \
  --kl-samples 25 \
  --max-tokens 200
```

### Batch Processing for Speed

Adjust batch size for faster evaluation:

```bash
gabliterate.automate --model "Nanbeige/Nanbeige4-3B-Thinking-2511" \
  --batch-size 8 \
  --num-versions 100
```

### For Advanced Configuration Customization

Clone the repository and edit `GabliterationConfig.random()` method in the source code to customize the hyperparameter search space.

## Performance Tips

### Memory Management
- Each version creates a new model copy
- Memory is cleared between versions
- Use smaller models for faster testing
- Reduce `--test-samples` if memory is tight

### Speed Optimization
- Use GPU/CUDA if available (automatically detected)
- Increase `--batch-size` for faster evaluation
- Reduce `--test-samples` for faster evaluation
- Start with fewer `--num-versions` to test the pipeline

### Recommended Workflows

1. **Quick Test** (5 minutes):
   ```bash
   gabliterate.automate --model "your-model" --num-versions 5 --test-samples 50
   ```

2. **Standard Search** (30 minutes):
   ```bash
   gabliterate.automate --model "your-model" --num-versions 20 --test-samples 100
   ```

3. **Thorough Search** (2+ hours):
   ```bash
   gabliterate.automate --model "your-model" --num-versions 50 --test-samples 200
   ```

## Troubleshooting

### Out of Memory
```bash
gabliterate.automate --model "your-model" --num-versions 10 --batch-size 1 --test-samples 50
```
- Reduce `--num-versions`
- Use smaller model
- Reduce `--batch-size`
- Reduce `--test-samples`

### Command Not Found: gabliterate
Ensure the package is installed:
```bash
pip install gabliteration
pip show gabliteration  # Verify installation
```

### All Versions Have High Refusal Rates
- The random configurations may need different ranges
- Try multiple runs with different `--num-versions`
- Check that the model supports the refusal behavior

## Citation

If you use this implementation, please cite:

```bibtex
@article{gulmez2025gabliteration,
  title={Gabliteration: Adaptive Multi-Directional Neural Weight Modification for Selective Behavioral Alteration in Large Language Models},
  author={G{\"u}lmez, G{\"o}kdeniz},
  journal={arXiv preprint arXiv:2512.18901},
  year={2025}
}
```

## License

Same license as the base models being modified (typically Apache 2.0 or similar).

## Support

For issues or questions:
- GitHub: Check the original Gabliteration repository
- Paper: https://arxiv.org/abs/2512.18901
- Email: goekdenizguelmez-ml@gmail.com