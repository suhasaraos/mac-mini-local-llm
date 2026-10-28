#!/bin/zsh
# Usage: ./ppl.sh <model-dir> <tag>  — same 64 samples (seed 7) for every quant
cd "${0:A:h}"
.venv/bin/mlx_lm.perplexity --model "$1" --sequence-length 1024 --num-samples 64 --batch-size 4 --seed 7 2>&1 \
  | tr '\r' '\n' | grep -vE "it/s\]|Warning" > "results/ppl-$2.txt"
echo "$2: $(grep -iE 'perplexity' results/ppl-$2.txt | tail -2 | tr '\n' ' ')"
