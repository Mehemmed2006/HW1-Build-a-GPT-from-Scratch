"""Part D: inference with KV cache, greedy, top-k and top-p sampling."""

import argparse
import time

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


# ==================================================
# 1. SAMPLING
# ==================================================

def _sample_next(logits, temperature, top_k, top_p, generator):
    """logits: [1, V] -> next token id tensor [1, 1]."""
    # Greedy.
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # top-k first.
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    # then top-p: smallest set whose total probability >= p (at least one token).
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove a token if the tokens before it already reach top_p.
        remove_sorted = (cumulative - sorted_probs) >= top_p
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ==================================================
# 2. GENERATE
# ==================================================

@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> list[int]:
    """Returns prompt_ids + the new token ids."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    prompt_ids = [int(t) for t in prompt_ids]
    if len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    if len(prompt_ids) + max_new_tokens > model.config.block_size:
        raise ValueError("Prompt + new tokens exceeds block_size.")

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past = None

    try:
        for _ in range(max_new_tokens):
            if use_cache:
                if past is None:
                    # Prefill: whole prompt once.
                    logits, _, past = model(idx, use_cache=True)
                else:
                    # Only the newest token goes through the model.
                    logits, _, past = model(
                        idx[:, -1:], past_key_values=past, use_cache=True
                    )
            else:
                logits, _ = model(idx)

            next_token = _sample_next(
                logits[:, -1, :], temperature, top_k, top_p, generator
            )
            idx = torch.cat((idx, next_token), dim=1)
    finally:
        model.train(was_training)

    return idx[0].tolist()


# ==================================================
# 3. HELPERS (used only by the command line)
# ==================================================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def safe_decode(tokenizer, ids):
    """Decode without failing if sampling cut a UTF-8 character in half."""
    data = b"".join(tokenizer.token_bytes[i] for i in ids)
    return data.decode("utf-8", errors="replace")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare_cache(model, prompt_ids, n):
    cached = generate(model, prompt_ids, n, temperature=0, use_cache=True)
    uncached = generate(model, prompt_ids, n, temperature=0, use_cache=False)
    identical = cached == uncached
    print(f"Greedy cache ON/OFF identical: {identical}")
    return identical


def benchmark(model, prompt_ids, n, device):
    speeds = {}
    for use_cache in (False, True):
        generate(model, prompt_ids, min(10, n), temperature=0, use_cache=use_cache)
        _sync(device)
        start = time.perf_counter()
        generate(model, prompt_ids, n, temperature=0, use_cache=use_cache)
        _sync(device)
        elapsed = time.perf_counter() - start
        speeds[use_cache] = n / elapsed
        label = "ON " if use_cache else "OFF"
        print(f"Cache {label}: {elapsed:.4f} s | {speeds[use_cache]:.2f} tokens/s")
    print(f"KV cache speed-up: {speeds[True] / speeds[False]:.2f}x")


# ==================================================
# 4. COMMAND LINE
# ==================================================

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT.")
    parser.add_argument("--checkpoint", default="checkpoints/M_best.pt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--compare_cache", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, device)
    prompt_ids = tokenizer.encode(args.prompt)

    print(f"Device: {device} | Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)")
    print(f"Parameters: {model.num_params():,}")

    if args.compare_cache:
        compare_cache(model, prompt_ids, args.max_new_tokens)
    if args.benchmark:
        benchmark(model, prompt_ids, args.max_new_tokens, device)

    if args.compare_cache or args.benchmark:
        if args.num_samples == 1 and args.output is None:
            return

    samples = []
    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        ids = generate(
            model,
            prompt_ids,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
            seed=seed,
        )
        samples.append(safe_decode(tokenizer, ids))

    for i, text in enumerate(samples, start=1):
        print("\n" + "=" * 50 + f"\nSAMPLE {i}\n" + "=" * 50)
        print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, text in enumerate(samples, start=1):
                f.write(f"SAMPLE {i}\n" + "=" * 50 + "\n" + text + "\n\n")
        print(f"\nSamples saved to: {args.output}")


if __name__ == "__main__":
    main()
