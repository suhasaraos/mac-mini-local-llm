"""Benchmark a local MLX model: generation speed, prompt speed, memory, and output.

Usage:
  .venv/bin/python bench.py --model models/<dir> [--draft-model models/<dir>] [--num-draft 3]
                            [--kv-bits 8] [--tag name] [--max-tokens 400]

Results are appended to results/results.jsonl; outputs go to results/<tag>/.
"""
import argparse, json, time, pathlib, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = pathlib.Path(__file__).parent

LONG_DOC = (ROOT / "prompts" / "long_context.txt")

PROMPTS = {
    "chat": "Explain in three short paragraphs why the sky looks blue, and why sunsets are red.",
    "code": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' "
            "into total seconds. Include type hints, a docstring, and three pytest tests.",
    "reasoning": "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 25 minutes, "
                 "then travels 150 km at 100 km/h. What time does it arrive? Show the steps briefly.",
    "long": None,  # filled from LONG_DOC: ~4k-token input, summary output
}


def build_prompt(tokenizer, text, thinking):
    msgs = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def run_one(model, tokenizer, prompt, args, draft):
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)
    text, last, drafted = "", None, 0
    kw = dict(max_tokens=args.max_tokens, sampler=sampler)
    if args.kv_bits:
        kw.update(kv_bits=args.kv_bits, quantized_kv_start=0)
    if draft is not None:
        kw.update(draft_model=draft, num_draft_tokens=args.num_draft)
    for r in stream_generate(model, tokenizer, prompt, **kw):
        text += r.text
        if getattr(args, "show", False):
            print(r.text, end="", flush=True)
        drafted += int(bool(getattr(r, "from_draft", False)))
        last = r
    return dict(prompt_tokens=last.prompt_tokens, prompt_tps=round(last.prompt_tps, 1),
                gen_tokens=last.generation_tokens, gen_tps=round(last.generation_tps, 2),
                peak_gb=round(last.peak_memory, 2),
                draft_accept=round(drafted / max(1, last.generation_tokens), 2) if draft else None,
                text=text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model")
    p.add_argument("--num-draft", type=int, default=3)
    p.add_argument("--kv-bits", type=int)
    p.add_argument("--tag")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--show", action="store_true", help="stream each answer to the screen (for recordings)")
    p.add_argument("--only", help="comma list of prompt names (chat,code,reasoning,long,ctx64k)")
    p.add_argument("--ctx-tokens", type=int, default=62000)
    args = p.parse_args()

    tag = args.tag or pathlib.Path(args.model).name
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tokenizer = load(args.model)
    draft = load(args.draft_model)[0] if args.draft_model else None
    load_s = round(time.time() - t0, 1)

    PROMPTS["long"] = ("Summarize the following text in 5 bullet points:\n\n"
                       + LONG_DOC.read_text())
    # ~64K-token input with a fact buried in the middle: checks memory fits, prompt speed,
    # decode speed at depth, and that the model still reads the whole context.
    base = LONG_DOC.read_text()
    reps = max(1, args.ctx_tokens * 4 // len(base))  # ~4 chars/token
    chunks = [base] * reps
    chunks.insert(reps // 2, "\nNOTE: The secret project codename is BLUE HERON 7.\n")
    PROMPTS["ctx64k"] = ("".join(chunks) +
                         "\n\nQuestion: What is the secret project codename mentioned above? "
                         "Answer in one line, then write two sentences about what the text covers.")
    if not args.only:
        PROMPTS.pop("ctx64k")  # opt-in: slow (minutes of prefill)
    names = args.only.split(",") if args.only else list(PROMPTS)

    # Warm-up so the first measured prompt isn't paying for kernel compilation.
    run_one(model, tokenizer, build_prompt(tokenizer, "Say hi.", False),
            argparse.Namespace(**{**vars(args), "max_tokens": 16, "show": False}), draft)

    rows = []
    for name in names:
        if args.show:
            print(f"\n\033[1m### {name}\033[0m\n", flush=True)
        r = run_one(model, tokenizer, build_prompt(tokenizer, PROMPTS[name], args.thinking), args, draft)
        if args.show:
            print()
        (out_dir / f"{name}.txt").write_text(r.pop("text"))
        r["prompt"] = name
        rows.append(r)
        print(f"{name:10s} gen {r['gen_tps']:6.2f} tok/s  prompt {r['prompt_tps']:7.1f} tok/s "
              f"({r['prompt_tokens']} tok)  peak {r['peak_gb']} GB"
              + (f"  draft-accept {r['draft_accept']}" if draft else ""))

    summary = dict(
        tag=tag, model=args.model, draft=args.draft_model, num_draft=args.num_draft,
        kv_bits=args.kv_bits, load_s=load_s, time=time.strftime("%Y-%m-%d %H:%M"),
        mean_gen_tps=round(statistics.mean(r["gen_tps"] for r in rows), 2),
        peak_gb=max(r["peak_gb"] for r in rows), runs=rows)
    print(f"== {tag}: mean gen {summary['mean_gen_tps']} tok/s, peak {summary['peak_gb']} GB, load {load_s}s")
    with open(ROOT / "results" / "results.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
