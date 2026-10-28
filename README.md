# A 27B LLM running locally on a base Mac mini

The scripts from my video where I run **Qwen 3.8 27B** fully offline on a base **Mac mini M6 (32 GB)** with Apple's
MLX framework, and test it for speed, memory and quality.

No cloud service and no subscription: just the model, a terminal chat and an OpenAI-compatible local server.

## What's here

| File | What it does |
|---|---|
| `serve.sh` | OpenAI-compatible API at `http://127.0.0.1:8080/v1` (thinking off, Qwen's recommended sampling) |
| `chat.sh` / `chat.py` | Terminal chat. `/reset` clears the history, Ctrl-D quits |
| `bench.py` | Speed and memory benchmark: chat, code, reasoning, a long document, and a ~64K-token context test |
| `quiz.py` | 10 graded maths and logic problems with greedy decoding, so each build is scored on the same questions |
| `ppl.sh` | Perplexity on 64 fixed samples (the same samples for every build) |
| `prompts/long_context.txt` | Filler text for the long-context test (Python's `json` module docs) |

## Setup

Needs an Apple Silicon Mac with 32 GB of memory and Python 3.12.

```sh
python3 -m venv .venv
.venv/bin/pip install mlx-lm==0.32.2 huggingface_hub
.venv/bin/hf download AtomicChat/Qwen3.8-27B-MLX-AD-3.50bpw-DWQ-12.7GB --local-dir models/qwen3.8-27b-dwq3.5
mkdir -p results
```

That's a 12 GB download. Close heavy apps before loading it.

## Use

```sh
./chat.sh                      # chat in the terminal
./serve.sh                     # local API; any OpenAI client can use base URL http://127.0.0.1:8080/v1

curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"models/qwen3.8-27b-dwq3.5","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'

.venv/bin/python bench.py --model models/qwen3.8-27b-dwq3.5 --tag dwq35
.venv/bin/python quiz.py  --model models/qwen3.8-27b-dwq3.5 --tag dwq35
./ppl.sh models/qwen3.8-27b-dwq3.5 dwq35
```

To turn thinking on for one request, add `"chat_template_kwargs": {"enable_thinking": true}` to it. Be warned: it
overthinks and is slow.

## My results (Mac mini M6, 32 GB)

Generation speed is limited by memory bandwidth (~170 GB/s on the base M6), because every token reads all of the
weights once.

| Build | Disk | Tokens/s (short) | Tokens/s @ 61K context | Peak RAM @ 61K | Quiz (out of 10) |
|---|---|---|---|---|---|
| mlx-community 4-bit | 15 GB | 9.7 | 6.4 | 22.1 GB | – |
| mlx-community mxfp4 | 14 GB | 10.2 | – | – | 8 |
| **AtomicChat DWQ 3.5-bit** | **12 GB** | **11.5** | **7.3** | **18.8 GB** | **9** |

- **Prompt reading:** about 300 tokens/s at 4K and 236 tokens/s at 61K. A full 64K prompt takes about 4.5 minutes
  the first time.
- **8-bit KV cache:** it made things worse here (slower and more memory), so leave it off.
- **Long coding sessions** (from the follow-up video): `mlx_lm.server` keeps several prompt caches by default, and in
  a long multi-turn chat they pile up until memory runs out. `./serve.sh --prompt-cache-size 1` fixes it. With it, a
  56K-token session peaked at 30.1 of 32 GB.

## Follow-up

Image to 3D on the same Mac, and the speculative decoding and memory tests from your comments:
[mac-mini-image-to-3d](https://github.com/suhasaraos/mac-mini-image-to-3d).

## License

MIT. The model has its own licence (see its Hugging Face page).
