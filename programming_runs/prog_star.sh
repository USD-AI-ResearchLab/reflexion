#!/bin/bash
# STAR: Reflexion + StepKnowledgeStore (step-level knowledge retrieval)
python main.py \
  --run_name "star" \
  --root_dir "root" \
  --dataset_path ./benchmarks/leetcode-hard-py.jsonl \
  --strategy "star" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 10 \
  --is_leetcode \
  --verbose
