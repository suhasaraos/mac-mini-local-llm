"""Terminal chat with the local Qwen (thinking off). Type /reset to clear history, Ctrl-D to quit."""
import sys
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tok = load("models/qwen3.8-27b-dwq3.5")
sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
history = []
while True:
    try:
        user = input("\n> ")
    except EOFError:
        break
    if user.strip() == "/reset":
        history = []; continue
    history.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    reply, r = "", None
    for r in stream_generate(model, tok, prompt, max_tokens=4096, sampler=sampler):
        print(r.text, end="", flush=True); reply += r.text
    print(f"\n[{r.generation_tps:.1f} tok/s, context {r.prompt_tokens + r.generation_tokens} tokens]")
    history.append({"role": "assistant", "content": reply})
