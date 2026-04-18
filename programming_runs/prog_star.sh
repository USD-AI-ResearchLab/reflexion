#!/bin/bash
# STAR: Reflexion + StepKnowledgeStore (step-level knowledge retrieval)
python main.py \
  --run_name "star" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py_hardest50.jsonl \
  --strategy "star" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 10 \
  --verbose
