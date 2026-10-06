"""Day 5: load a trained model from disk, in a fresh Python process, and let it write.

    uv run python generate.py --checkpoint runs/friday/final.pt
    uv run python generate.py --checkpoint runs/friday/final.pt --prompt "The cat" --temperature 1.2 --top-k 0

For each prompt it samples up to --max-tokens tokens: logits are divided by --temperature (lower = safer, higher =
wilder), then only the --top-k most likely tokens are kept (0 keeps all). Sampling stops early at <|endoftext|>.
--seed makes a run repeatable on the same machine.
"""
import argparse, os

import morpheme
import torch
import torch.nn.functional as F

import frozen
from model import GPT, device

PROMPTS = ["Once upon a time", "Lily wanted to", "The big dog", "One day, Tom found", "In the garden, there was"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join("runs", "friday", "final.pt"))
    ap.add_argument("--prompt", action="append", help="repeatable; default: five story openings")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = ap.parse_args()
    frozen.check()
    dev = device() if a.device == "auto" else a.device
    tok = morpheme.Tokenizer.from_file(os.path.join("data", "tok4096.json"))
    eot = tok.token_to_id("<|endoftext|>")
    model = GPT().to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev, weights_only=True))
    model.eval()
    g = torch.Generator().manual_seed(a.seed)
    for prompt in a.prompt or PROMPTS:
        idx = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=dev)
        with torch.no_grad():
            for _ in range(a.max_tokens):
                logits = model(idx[:, -256:])[:, -1, :].float().cpu() / a.temperature
                if a.top_k:
                    kth = torch.topk(logits, a.top_k).values[:, [-1]]
                    logits[logits < kth] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
                if nxt.item() == eot:
                    break
                idx = torch.cat([idx, nxt.to(dev)], dim=1)
        print(f"--- {prompt!r}\n{tok.decode(idx[0].tolist())}\n")


if __name__ == "__main__":
    main()
