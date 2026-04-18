"""
STAR for Programming — Reflexion + StepKnowledgeStore.

STAR maps the HotpotQA step loop onto programming iterations:
  - Each iteration = one step (generate code → run tests → optional reflection)
  - EXPECTED = what tests should return
  - KEY = 3-5 word abstract pattern for the implementation challenge
  - CORRECTION = generalizable rule about the mistake (stored when tests fail)
  - knowledge_k=2 rules retrieved before each implementation attempt

Knowledge retrieval:
  - Iteration 1: classify_problem_type(func_sig) — no LLM call
  - Iteration N: KEY from iteration N-1

Reflection: identical to Reflexion — gen.self_reflection(impl, feedback, model)
The reflection is used in the next implementation attempt unchanged.
"""

import re
import numpy as np
from typing import List, Optional

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory


# ---------------------------------------------------------------------------
# StepKnowledgeStore — copied exactly from hotpotqa_runs/star_agents.py
# ---------------------------------------------------------------------------

class StepKnowledge:
    def __init__(self, action_intent: str, rule: str, positive: bool):
        self.action_intent = action_intent
        self.rule          = rule
        self.positive      = positive
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.action_intent + " " + self.rule)
        return self._embedding


class StepKnowledgeStore:
    """Attention-weighted retrieval over step-level knowledge. Task-agnostic."""

    _st_model = None

    def __init__(self, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
        self.knowledge:   List[StepKnowledge] = []
        self.embed_fn     = self._sentence_transformer_embed
        self.tau          = tau
        self.adaptive_tau = adaptive_tau
        self.mmr_lambda   = mmr_lambda

    def add(self, knowledge: StepKnowledge) -> None:
        if self.knowledge:
            q_emb = self.embed_fn(knowledge.action_intent + " " + knowledge.rule)
            for existing in self.knowledge[-20:]:
                sim = float(np.dot(q_emb, existing.embedding(self.embed_fn)))
                if sim > 0.92:
                    return
        self.knowledge.append(knowledge)

    def retrieve(self, query: str, k: int = 2) -> List[StepKnowledge]:
        if not self.knowledge:
            return []
        q_emb = self.embed_fn(query)
        d     = q_emb.shape[0]
        tau   = (0.05 + 0.25 * min(len(self.knowledge) / 100.0, 1.0)
                 if self.adaptive_tau else self.tau)
        logits = [
            float(np.dot(q_emb, sk.embedding(self.embed_fn))) / np.sqrt(d) / tau
            for sk in self.knowledge
        ]
        logits_arr  = np.array(logits)
        logits_arr -= logits_arr.max()
        alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()
        scored = sorted(zip(alphas.tolist(), self.knowledge),
                        key=lambda x: x[0], reverse=True)
        return self._mmr_select(scored, k)

    def _mmr_select(self, scored, k):
        if not scored:
            return []
        selected, candidates = [], list(scored)
        while len(selected) < k and candidates:
            if not selected:
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score, best = -1e9, None
                sel_embs = [s.embedding(self.embed_fn) for s in selected]
                for attn_score, sk in candidates:
                    max_sim = max(
                        float(np.dot(sk.embedding(self.embed_fn), se))
                        for se in sel_embs
                    )
                    mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
                    if mmr > best_score:
                        best_score, best = mmr, sk
            selected.append(best)
            candidates = [(a, s) for a, s in candidates if s is not best]
        return selected

    @staticmethod
    def _get_st_model():
        if StepKnowledgeStore._st_model is None:
            from sentence_transformers import SentenceTransformer
            StepKnowledgeStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return StepKnowledgeStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        return StepKnowledgeStore._get_st_model().encode(
            text, normalize_embeddings=True).astype(np.float64)


# ---------------------------------------------------------------------------
# Problem type classifier — no LLM call, step-1 retrieval query
# ---------------------------------------------------------------------------

def classify_problem_type(func_sig: str) -> str:
    """
    Classify programming problem into an abstract key for iteration-1 retrieval.
    No LLM call — pure keyword heuristics on the function signature / docstring.
    """
    s = func_sig.lower()
    if any(w in s for w in ['sort', 'order', 'rank']):
        return 'implement-sort-logic'
    if any(w in s for w in ['search', 'find', 'locate', 'index']):
        return 'implement-search-logic'
    if any(w in s for w in ['string', 'str', 'char', 'text', 'parse']):
        return 'implement-string-manipulation'
    if any(w in s for w in ['tree', 'node', 'graph', 'path']):
        return 'implement-tree-graph'
    if any(w in s for w in ['dp', 'dynamic', 'memo', 'cache', 'optimal']):
        return 'implement-dynamic-programming'
    if any(w in s for w in ['list', 'array', 'matrix', 'grid']):
        return 'implement-array-logic'
    if any(w in s for w in ['count', 'sum', 'total', 'number', 'num']):
        return 'implement-counting-logic'
    if any(w in s for w in ['max', 'min', 'largest', 'smallest']):
        return 'implement-extremum-logic'
    return 'implement-general-logic'


# ---------------------------------------------------------------------------
# STAR prompt helpers for programming
# ---------------------------------------------------------------------------

def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
    if not knowledge:
        return ""
    lines = ["[STEP RULES from past experience]"]
    for sk in knowledge:
        icon = "OK" if sk.positive else "FIX"
        lines.append(f"{icon}: {sk.rule[:150]}")
    lines.append("")
    return '\n'.join(lines)


STAR_ITER_INSTRUCTION = """\

After implementing, respond with these additional lines:
EXPECTED: <what test results you expect — pass/fail and why>
KEY: <3-5 word abstract pattern for this problem type, e.g. implement-loop-logic, handle-edge-case, fix-off-by-one, implement-sort-logic>
CORRECTION: <if previous EXPECTED was wrong — one generalizable rule about the mistake. Omit if first iteration or prediction was correct.>
"""


def parse_star_iter_response(raw: str) -> dict:
    """Extract EXPECTED/KEY/CORRECTION from LLM output after the code block."""
    result = {'expected': '', 'key': '', 'correction': ''}
    if not raw:
        return result
    current_key = None
    for line in raw.split('\n'):
        line = line.strip()
        matched = False
        for k in ['EXPECTED', 'KEY', 'CORRECTION']:
            if line.upper().startswith(f'{k}:'):
                current_key = k.lower()
                result[current_key] = line[len(k)+1:].strip()
                matched = True
                break
        if not matched and current_key and line and not line.startswith('```'):
            result[current_key] += ' ' + line
    for k in result:
        result[k] = result[k].strip()
    return result


def prediction_matched_prog(expected: str, is_passing: bool) -> bool:
    """Check if the LLM's EXPECTED prediction matches actual test outcome."""
    exp_lower = expected.lower()
    if is_passing:
        return any(w in exp_lower for w in ['pass', 'correct', 'succeed', 'work'])
    else:
        return any(w in exp_lower for w in ['fail', 'error', 'wrong', 'incorrect'])


# ---------------------------------------------------------------------------
# run_star_reflexion — drop-in replacement for run_reflexion
# ---------------------------------------------------------------------------

def run_star_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    knowledge_k: int = 2,
) -> None:
    """
    STAR Reflexion for programming.

    Shared knowledge_store accumulates rules across all problems and all iterations.
    Per-problem: standard Reflexion loop with STAR knowledge injection.
    """
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    # Shared knowledge store — persists across all problems
    knowledge_store = StepKnowledgeStore()

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        skipped = False

        # Per-problem iteration state
        prev_key      = ''
        prev_expected = ''
        prev_passing  = None   # bool or None

        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            # Retrieval query: problem type on iter 1, prev KEY on subsequent
            retrieval_query = prev_key if prev_key else classify_problem_type(item["prompt"])
            retrieved = knowledge_store.retrieve(retrieval_query, k=knowledge_k)
            knowledge_str = format_step_knowledge(retrieved)
            if retrieved:
                print(f'  [STAR] {len(retrieved)} rules for "{retrieval_query}"')

            # Build knowledge prefix for the prompt
            # We inject knowledge_str by prepending to the func_sig
            star_prefix = knowledge_str + STAR_ITER_INSTRUCTION if knowledge_str else STAR_ITER_INSTRUCTION

            # First attempt with STAR knowledge injected
            # We use "simple" strategy but prepend knowledge to the prompt
            func_sig_with_knowledge = star_prefix + '\n' + item["prompt"] if knowledge_str else item["prompt"]
            cur_func_impl = gen.func_impl(func_sig_with_knowledge, model, "simple")
            if not cur_func_impl:
                print(f"Warning: empty implementation for problem {i}, skipping")
                item["solution"] = ""
                item["is_solved"] = False
                write_jsonl(log_path, [item], append=True)
                skipped = True
                break

            assert isinstance(cur_func_impl, str)
            implementations.append(cur_func_impl)

            # Extract STAR fields from LLM output (appended after code)
            star_fields = parse_star_iter_response(cur_func_impl)
            step_key   = star_fields['key']
            expected   = star_fields['expected']
            correction = star_fields['correction']

            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)
            print(f'  [STAR] Iter 1 | passing={is_passing} | KEY={step_key}')

            # Store correction if prediction was wrong and correction is substantive
            if correction and len(correction) > 15:
                storage_key = step_key if step_key else retrieval_query
                knowledge_store.add(StepKnowledge(
                    action_intent=storage_key,
                    rule=correction,
                    positive=False,
                ))
                print(f'  [STAR] FIX [{storage_key}]: {correction[:80]}')

            # Update per-problem state
            prev_key      = step_key
            prev_expected = expected
            prev_passing  = is_passing

            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                break

            # Reflexion + STAR loop
            cur_iter    = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                # Generate reflection (identical to run_reflexion)
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model)
                reflections.append(reflection)

                # Retrieval for this iteration
                retrieval_query = prev_key if prev_key else classify_problem_type(item["prompt"])
                retrieved = knowledge_store.retrieve(retrieval_query, k=knowledge_k)
                knowledge_str = format_step_knowledge(retrieved)
                if retrieved:
                    print(f'  [STAR] Iter {cur_iter+1} | {len(retrieved)} rules for "{retrieval_query}"')

                # Inject knowledge into func_sig for next attempt
                func_sig_with_knowledge = (
                    (knowledge_str + '\n') if knowledge_str else ''
                ) + item["prompt"]

                cur_func_impl = gen.func_impl(
                    func_sig=func_sig_with_knowledge,
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Extract STAR fields
                star_fields = parse_star_iter_response(cur_func_impl)
                step_key   = star_fields['key']
                expected   = star_fields['expected']
                correction = star_fields['correction']

                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)
                print(f'  [STAR] Iter {cur_iter+1} | passing={is_passing} | KEY={step_key}')

                # Store correction
                if correction and len(correction) > 15:
                    storage_key = step_key if step_key else retrieval_query
                    knowledge_store.add(StepKnowledge(
                        action_intent=storage_key,
                        rule=correction,
                        positive=False,
                    ))
                    print(f'  [STAR] FIX [{storage_key}]: {correction[:80]}')

                prev_key      = step_key
                prev_expected = expected
                prev_passing  = is_passing

                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break

                cur_iter += 1
            cur_pass += 1

        if not skipped:
            item["is_solved"] = is_solved
            item["reflections"] = reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["solution"] = cur_func_impl
            write_jsonl(log_path, [item], append=True)

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
