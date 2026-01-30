"""Full GRPO training loop for reasoning on RTX 3090."""
import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from ch02_rollouts.sample import load_model_qlora
from ch03_rewards.verifier import math_reward, reward_with_thinking
from ch05_grpo.grpo import grpo_step


def extract_gsm8k_answer(answer_text: str) -> str:
    lines = answer_text.strip().split("\n")
    for line in reversed(lines):
        if "####" in line:
            return line.split("####")[-1].strip()
    return lines[-1].strip() if lines else ""

def train(
    num_steps: int = 50,
    G: int = 4,
    lr: float = 1e-5,
    use_thinking_reward: bool = False,
    log_interval: int = 5,
):
    print("=== GRPO Training on RTX 3090 ===")
    print(f"Steps: {num_steps}, G: {G}, LR: {lr}")
    print(f"Thinking reward: {use_thinking_reward}")
    print()

    model, tokenizer = load_model_qlora()
    optimizer = AdamW(model.parameters(), lr=lr)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=42)

    reward_fn = reward_with_thinking if use_thinking_reward else math_reward

    losses = []
    rewards = []
    start_time = time.time()

    for step in tqdm(range(num_steps), desc="Training"):
        example = dataset[step % len(dataset)]
        prompt = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])

        optimizer.zero_grad()
        loss, reward, responses = grpo_step(
            model, tokenizer, prompt, ground_truth, reward_fn, G=G
        )
        optimizer.step()

        losses.append(loss)
        rewards.append(reward)

        if (step + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            avg_reward = sum(rewards[-log_interval:]) / log_interval
            elapsed = time.time() - start_time
            print(f"\nStep {step+1}: loss={avg_loss:.4f}, reward={avg_reward:.4f}, time={elapsed:.1f}s")
            print(f"  Prompt: {prompt[:60]}...")
            print(f"  Response: {responses[0][:100]}...")

        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/num_steps:.2f}s/step)")
    print(f"Final avg loss: {sum(losses[-10:])/10:.4f}")
    print(f"Final avg reward: {sum(rewards[-10:])/10:.4f}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory: {mem:.2f} GB")

    return model, losses, rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    train(
        num_steps=args.steps,
        G=args.G,
        lr=args.lr,
        use_thinking_reward=args.thinking,
    )
