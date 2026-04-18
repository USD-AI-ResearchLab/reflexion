#!/bin/bash
# STAR: Reflexion + StepKnowledgeStore (step-level knowledge retrieval)
python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "star" \
        --strategy star \
        --knowledge_k 2 \
        --model "gpt-oss"
