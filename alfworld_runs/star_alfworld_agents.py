"""
STAR for ALFWorld — StepKnowledgeStore + classify_task_type + prompt helpers.

StepKnowledgeStore is task-agnostic (copied exactly from hotpotqa_runs/star_agents.py).
ALFWorld-specific additions:
  - classify_task_type: regex heuristic for step-1 retrieval (no LLM call)
  - format_step_knowledge: compact [STEP RULES] block injected before action
  - STAR_STEP_INSTRUCTION: ALFWorld action format (text commands, not Search/Lookup)
  - parse_star_response: parses THOUGHT/ACTION/EXPECTED/KEY/CORRECTION from LLM output
"""

import re
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# StepKnowledge / StepKnowledgeStore — copied exactly from star_agents.py
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
# Task type classifier — no LLM call, used for step-1 retrieval query
# ---------------------------------------------------------------------------

def classify_task_type(task_description: str) -> str:
    """
    Classify ALFWorld task into an abstract key for step-1 knowledge retrieval.
    No LLM call — pure keyword heuristics.
    """
    t = task_description.lower()
    if 'examine' in t or 'desklamp' in t or 'under' in t:
        return 'examine-in-light'
    if 'clean' in t or 'sink' in t or 'sinkbasin' in t:
        return 'clean-and-place'
    if 'heat' in t or 'microwave' in t:
        return 'heat-and-place'
    if 'cool' in t or 'fridge' in t or 'refrigerator' in t:
        return 'cool-and-place'
    if 'two' in t or ' 2 ' in t:
        return 'pick-two-and-place'
    return 'pick-and-place'


# ---------------------------------------------------------------------------
# ALFWorld STAR prompt helpers
# ---------------------------------------------------------------------------

def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
    if not knowledge:
        return ""
    lines = ["[STEP RULES]"]
    for sk in knowledge:
        icon = "OK" if sk.positive else "FIX"
        lines.append(f"{icon}: {sk.rule[:150]}")
    lines.append("")
    return '\n'.join(lines)


STAR_STEP_INSTRUCTION = (
    "\nRespond with EXACTLY these labels:\n"
    "THOUGHT: <your reasoning>\n"
    "ACTION: <the exact environment command, e.g. go to fridge 1, take apple 1 from fridge 1>\n"
    "EXPECTED: <brief description of what the environment will say after this action>\n"
    "KEY: <3-5 word abstract pattern for this step type, "
    "e.g. go-to-receptacle, take-object-from, heat-object-microwave, "
    "put-object-in, examine-under-lamp, open-receptacle>\n"
    "CORRECTION: <if prev EXPECTED was wrong — one generalizable rule about this mistake. "
    "Omit entirely if first step or prediction was correct.>\n"
)


def parse_star_response(raw: str) -> dict:
    """Parse THOUGHT/ACTION/EXPECTED/KEY/CORRECTION from LLM output."""
    result = {
        'thought': '', 'action': '', 'expected': '',
        'key': '', 'correction': '',
    }
    if not raw:
        return result
    current_key = None
    for line in raw.split('\n'):
        line = line.strip()
        matched = False
        for k in ['THOUGHT', 'ACTION', 'EXPECTED', 'KEY', 'CORRECTION']:
            if line.upper().startswith(f'{k}:'):
                current_key = k.lower()
                result[current_key] = line[len(k)+1:].strip()
                matched = True
                break
        if not matched and current_key and line:
            result[current_key] += ' ' + line
    for k in result:
        result[k] = result[k].strip()
    return result


def prediction_matched_alfworld(expected: str, actual: str) -> bool:
    """
    ALFWorld observations are short deterministic strings.
    Use near-exact matching: check if ≥60% of expected words appear in actual.
    """
    stopwords = {'you', 'the', 'a', 'an', 'is', 'was', 'to', 'of', 'in',
                 'and', 'or', 'it', 'this', 'that', 'with', 'on', 'at'}
    exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
    act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
    if not exp_words:
        return False
    return len(exp_words & act_words) / len(exp_words) > 0.6
