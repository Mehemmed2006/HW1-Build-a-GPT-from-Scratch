
import argparse
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


# ==================================================
# 1. DEVICE
# ==================================================

def get_device(device_name):
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    return device


# ==================================================
# 2. LOAD MODEL
# ==================================================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ==================================================
# 3. DECODING STRATEGIES
# ==================================================

def sample_next_token(
    logits,
    method="greedy",
    temperature=1.0,
    top_k=50,
    top_p=0.9,
):
    logits = logits[:, -1, :]

    if method == "greedy":
        return torch.argmax(
            logits, dim=-1, keepdim=True
        )

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    logits = logits / temperature

    if method == "top_k":
        k = min(top_k, logits.size(-1))

        if k < 1:
            raise ValueError("top_k must be at least 1.")

        values, _ = torch.topk(logits, k)
        threshold = values[:, -1, None]
        logits = logits.masked_fill(
            logits < threshold, float("-inf")
        )

    elif method == "top_p":
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")

        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True
        )

        sorted_probs = torch.softmax(
            sorted_logits, dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs, dim=-1
        )

        remove = cumulative_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        indices_to_remove = torch.zeros_like(
            logits, dtype=torch.bool
        )

        indices_to_remove.scatter_(
            1, sorted_indices, remove
        )

        logits = logits.masked_fill(
            indices_to_remove, float("-inf")
        )

    elif method != "temperature":
        raise ValueError(f"Unknown method: {method}")

    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(
        probabilities, num_samples=1
    )


# ==================================================
# 4. GENERATE TOKENS
# ==================================================

@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens=120,
    method="greedy",
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    use_cache=True,
):
    model.eval()

    idx = prompt_ids.clone()
    block_size = model.config.block_size

    if idx.size(1) == 0:
        raise ValueError("Prompt cannot be empty.")

    if idx.size(1) + max_new_tokens > block_size:
        raise ValueError(
            f"Prompt ({idx.size(1)}) + new tokens "
            f"({max_new_tokens}) exceeds block_size "
            f"({block_size})."
        )

    past_key_values = None

    if use_cache:
        logits, _, past_key_values = model(
            idx,
            use_cache=True,
        )

    for step in range(max_new_tokens):

        if use_cache:
            if step > 0:
                logits, _, past_key_values = model(
                    idx[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        else:
            logits, _ = model(idx)

        next_token = sample_next_token(
            logits,
            method=method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        idx = torch.cat(
            (idx, next_token), dim=1
        )

    return idx


# ==================================================
# 5. CACHE CORRECTNESS TEST
# ==================================================

def compare_cache(model, prompt_ids, max_new_tokens=120):
    print("\nKV CACHE CORRECTNESS TEST")
    print("-" * 45)

    cached = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        method="greedy",
        use_cache=True,
    )

    uncached = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        method="greedy",
        use_cache=False,
    )

    cached_new = cached[:, prompt_ids.size(1):]
    uncached_new = uncached[:, prompt_ids.size(1):]

    identical = torch.equal(
        cached_new, uncached_new
    )

    print("Cache ON/OFF identical:", identical)
    print("Generated tokens:", max_new_tokens)

    if not identical:
        differences = (
            cached_new != uncached_new
        ).sum().item()

        print("Different tokens:", differences)

    return identical


# ==================================================
# 6. SPEED BENCHMARK
# ==================================================

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, prompt_ids, device, max_new_tokens=120):
    print("\nGENERATION SPEED BENCHMARK")
    print("-" * 45)

    results = {}

    for cache_status in [False, True]:

        # Warm-up
        generate(
            model,
            prompt_ids,
            max_new_tokens=10,
            method="greedy",
            use_cache=cache_status,
        )

        synchronize(device)

        start_time = time.perf_counter()

        generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            method="greedy",
            use_cache=cache_status,
        )

        synchronize(device)

        elapsed = time.perf_counter() - start_time

        speed = max_new_tokens / elapsed

        results[cache_status] = speed

        label = "ON" if cache_status else "OFF"

        print(
            f"Cache {label}: "
            f"{elapsed:.4f} seconds | "
            f"{speed:.2f} tokens/sec"
        )

    speed_without_cache = results[False]
    speed_with_cache = results[True]

    print(
        f"\nKV Cache speedup: "
        f"{speed_with_cache / speed_without_cache:.2f}x"
    )

    return results


# ==================================================
# 7. GENERATE MULTIPLE SAMPLES
# ==================================================

def generate_samples(
    model,
    tokenizer,
    prompt_ids,
    device,
    num_samples=3,
    max_new_tokens=120,
    method="top_p",
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    use_cache=True,
):
    samples = []

    print("\nGENERATING SAMPLES")
    print("-" * 45)

    for sample_number in range(num_samples):
        generated = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            method=method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )

        generated_text = tokenizer.decode(
            generated[0].tolist()
        )

        samples.append(generated_text)

        print(
            f"Sample {sample_number + 1}/{num_samples} completed."
        )

    return samples


def save_samples(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for i, sample in enumerate(samples, start=1):
            file.write(f"SAMPLE {i}\n")
            file.write("=" * 50 + "\n")
            file.write(sample)
            file.write("\n\n")

    print("Samples saved to:", output_path)


# ==================================================
# 8. MAIN
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/M_best.pt",
    )

    parser.add_argument(
        "--tokenizer",
        default="tokenizer.json",
    )

    parser.add_argument(
        "--prompt",
        default="ROMEO:",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--method",
        choices=[
            "greedy",
            "temperature",
            "top_k",
            "top_p",
        ],
        default="greedy",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--cache",
        choices=["on", "off"],
        default="on",
    )

    parser.add_argument(
        "--compare_cache",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        default="samples.txt",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive.")

    if args.num_samples < 1:
        parser.error("--num_samples must be positive.")

    device = get_device(args.device)

    print("=" * 50)
    print("GPT TEXT GENERATION")
    print("=" * 50)

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Decoding method:", args.method)
    print("KV cache:", args.cache.upper())

    tokenizer = BPETokenizer.load(args.tokenizer)

    model = load_model(
        args.checkpoint,
        device,
    )

    prompt_tokens = tokenizer.encode(args.prompt)

    prompt_ids = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    print("Prompt:", repr(args.prompt))
    print("Prompt token count:", prompt_ids.size(1))
    print("Model parameters:", model.num_params())

    if args.compare_cache:
        compare_cache(
            model,
            prompt_ids,
            args.max_new_tokens,
        )

    if args.benchmark:
        benchmark(
            model,
            prompt_ids,
            device,
            args.max_new_tokens,
        )

    samples = generate_samples(
        model,
        tokenizer,
        prompt_ids,
        device,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        method=args.method,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=(args.cache == "on"),
    )

    for i, sample in enumerate(samples, start=1):
        print("\n" + "=" * 50)
        print(f"SAMPLE {i}")
        print("=" * 50)
        print(sample)

    if args.num_samples > 1:
        save_samples(samples, args.output)


if __name__ == "__main__":
    main()
