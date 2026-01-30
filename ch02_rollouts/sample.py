"""Rollout generation for RL post-training."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_qlora(model_name: str = "Qwen/Qwen3-8B-Instruct"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def sample_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def sample_batch(
    model,
    tokenizer,
    prompt: str,
    n: int = 4,
    temperature: float = 0.7,
) -> list[str]:
    return [sample_response(model, tokenizer, prompt, temperature) for _ in range(n)]

if __name__ == "__main__":
    print("Loading model with QLoRA...")
    model, tokenizer = load_model_qlora()

    prompt = "What is 15 * 23?"
    print(f"Prompt: {prompt}")

    responses = sample_batch(model, tokenizer, prompt, n=4)
    for i, r in enumerate(responses):
        print(f"Response {i+1}: {r[:200]}...")
