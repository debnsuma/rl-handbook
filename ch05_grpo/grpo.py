"""GRPO: Group Relative Policy Optimization."""
from collections.abc import Callable

import torch
import torch.nn.functional as F


def compute_log_probs(model, tokenizer, prompt: str, response: str) -> torch.Tensor:
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, prompt_len-1:-1]
    target_ids = inputs.input_ids[0, prompt_len:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum()

def grpo_step(
    model,
    tokenizer,
    prompt: str,
    ground_truth: str,
    reward_fn: Callable[[str, str], float],
    G: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> tuple[float, float, list[str]]:
    model.train()

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(G):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    rewards = torch.tensor(
        [reward_fn(r, ground_truth) for r in responses],
        dtype=torch.float32,
        device=model.device,
    )

    baseline = rewards.mean()
    advantages = rewards - baseline

    loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    for response, adv in zip(responses, advantages):
        if adv.abs() < 1e-8:
            continue
        log_prob = compute_log_probs(model, tokenizer, formatted_prompt, response)
        loss = loss - adv * log_prob / G

    if loss.requires_grad:
        loss.backward()

    return loss.item(), rewards.mean().item(), responses

if __name__ == "__main__":
    print("GRPO module loaded successfully")
    print("Run validate_grpo_theodolos task to test on RTX 3090")
