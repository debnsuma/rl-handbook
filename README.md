# RL Post-Training Handbook

Code companion for the **RL Post-Training Handbook** — building a reasoning model on a single GPU.

## What This Is

Runnable PyTorch code for training reasoning capabilities into language models using:
- **GRPO** (Group Relative Policy Optimization) — critic-free RL
- **Think tokens** — explicit reasoning traces
- **QLoRA** — memory-efficient fine-tuning for 24GB GPUs

## Structure

```
ch02_rollouts/     # Response sampling with temperature/top-p
ch03_rewards/      # Verifiable reward functions (math, code)
ch05_grpo/         # GRPO algorithm implementation
ch07_training/     # Complete training loop
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with 24GB+ VRAM (3090, 4090, A5000, etc.)

```bash
pip install torch transformers peft bitsandbytes accelerate
```

## Quick Start

```python
from ch02_rollouts.sample import load_model_qlora, sample_batch

model, tokenizer = load_model_qlora("Qwen/Qwen3-8B-Instruct")
responses = sample_batch(model, tokenizer, "What is 15 * 23?", n=4)
```

## Book

Get the full book with explanations: [RL Post-Training Handbook on Gumroad](https://gumroad.com/)

## License

MIT
