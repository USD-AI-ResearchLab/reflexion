# """
# STAR — Step-level Trajectory-Agnostic Retrieval

# STAR = RAR + StepKnowledgeStore

# RAR component (unchanged):
#   - TrajectoryStore: stores full episodes with reflections
#   - After failure: retrieve top-k similar trajectories + their reflections
#   - Use retrieved trajectories as context to generate new reflection
#   - Agent sees reflections from retrieved trajectories only (not all)

# STAR addition (on top of RAR):
#   - StepKnowledgeStore: stores atomic step-level rules
#   - Before every action: retrieve top-k relevant rules via attention
#   - LLM generates EXPECTED + NEXT_INTENT + optional CORRECTION per step
#   - Zero extra LLM calls vs RAR
# """

# import re
# import string
# from typing import List, Tuple, Optional
# from enum import Enum
# import numpy as np
# import tiktoken
# from langchain import Wikipedia
# from langchain.agents.react.base import DocstoreExplorer
# from langchain.prompts import PromptTemplate
# from llm import AnyOpenAILLM
# from prompts import (
#     REFLECTION_SYSTEM_PROMPT,
#     REACT_SYSTEM_PROMPT,
#     reflect_prompt,
#     react_reflect_agent_prompt,
#     REFLECTION_HEADER,
#     LAST_TRIAL_HEADER,
#     REFLECTION_AFTER_LAST_TRIAL_HEADER,
#     CLASSIFY_ERROR_SYSTEM_PROMPT,
# )
# from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, WEBTHINK_SIMPLE2
# from sentence_transformers import SentenceTransformer

# # ── Import RAR components — exact names from retrieval_agents.py ─────────────
# from retrieval_agents import (
#     TrajectoryRecord,
#     TrajectoryStore,
#     classify_error,
#     format_retrieved_trajectories,
#     RETRIEVAL_REFLECTION_HEADER,
#     format_last_attempt,
#     format_reflections,
# )


# class ReflexionStrategy(Enum):
#     NONE                           = 'base'
#     REFLEXION                      = 'reflexion'
#     RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'
#     STAR                           = 'star'


# # ---------------------------------------------------------------------------
# # Step Knowledge — STAR addition
# # ---------------------------------------------------------------------------

# class StepKnowledge:
#     """
#     One generalizable rule extracted from a single step.
#     positive=True  → prediction matched actual (confirmed rule)
#     positive=False → prediction differed from actual (corrective rule)
#     """
#     def __init__(self,
#                  action_intent: str,
#                  rule: str,
#                  positive: bool):
#         self.action_intent = action_intent
#         self.rule          = rule
#         self.positive      = positive
#         self._embedding: Optional[np.ndarray] = None

#     def embedding(self, embed_fn) -> np.ndarray:
#         if self._embedding is None:
#             self._embedding = embed_fn(self.action_intent + " " + self.rule)
#         return self._embedding


# class StepKnowledgeStore:
#     """Attention-weighted retrieval over step-level knowledge."""

#     _st_model = None

#     def __init__(self, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
#         self.knowledge:   List[StepKnowledge] = []
#         self.embed_fn     = self._sentence_transformer_embed
#         self.tau          = tau
#         self.adaptive_tau = adaptive_tau
#         self.mmr_lambda   = mmr_lambda

#     def add(self, knowledge: StepKnowledge) -> None:
#         # Deduplication: skip if very similar rule already exists
#         if self.knowledge:
#             q_emb = self.embed_fn(knowledge.action_intent + " " + knowledge.rule)
#             for existing in self.knowledge[-20:]:
#                 sim = float(np.dot(q_emb, existing.embedding(self.embed_fn)))
#                 if sim > 0.92:
#                     return
#         self.knowledge.append(knowledge)

#     def retrieve(self, action_intent: str, k: int = 2) -> List[StepKnowledge]:
#         if not self.knowledge:
#             return []
#         q_emb = self.embed_fn(action_intent)
#         d     = q_emb.shape[0]
#         tau   = (0.05 + 0.25 * min(len(self.knowledge) / 100.0, 1.0)
#                  if self.adaptive_tau else self.tau)
#         logits = [
#             float(np.dot(q_emb, sk.embedding(self.embed_fn))) / np.sqrt(d) / tau
#             for sk in self.knowledge
#         ]
#         logits_arr  = np.array(logits)
#         logits_arr -= logits_arr.max()
#         alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()
#         scored = sorted(zip(alphas.tolist(), self.knowledge),
#                         key=lambda x: x[0], reverse=True)
#         return self._mmr_select(scored, k)

#     def _mmr_select(self, scored, k):
#         if not scored:
#             return []
#         selected, candidates = [], list(scored)
#         while len(selected) < k and candidates:
#             if not selected:
#                 _, best = max(candidates, key=lambda x: x[0])
#             else:
#                 best_score, best = -1e9, None
#                 sel_embs = [s.embedding(self.embed_fn) for s in selected]
#                 for attn_score, sk in candidates:
#                     max_sim = max(
#                         float(np.dot(sk.embedding(self.embed_fn), se))
#                         for se in sel_embs
#                     )
#                     mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
#                     if mmr > best_score:
#                         best_score, best = mmr, sk
#             selected.append(best)
#             candidates = [(a, s) for a, s in candidates if s is not best]
#         return selected

#     @staticmethod
#     def _get_st_model():
#         if StepKnowledgeStore._st_model is None:
#             StepKnowledgeStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
#         return StepKnowledgeStore._st_model

#     @staticmethod
#     def _sentence_transformer_embed(text: str) -> np.ndarray:
#         return StepKnowledgeStore._get_st_model().encode(
#             text, normalize_embeddings=True).astype(np.float64)


# # ---------------------------------------------------------------------------
# # Structured output parser
# # ---------------------------------------------------------------------------

# def parse_structured_response(raw: str) -> dict:
#     result = {
#         'thought': '', 'action': '', 'expected': '',
#         'next_intent': '', 'correction': '',
#     }
#     if not raw:
#         return result
#     current_key = None
#     for line in raw.split('\n'):
#         line = line.strip()
#         matched = False
#         for key in ['THOUGHT', 'ACTION', 'EXPECTED', 'NEXT_INTENT', 'CORRECTION']:
#             if line.upper().startswith(f'{key}:'):
#                 current_key = key.lower()
#                 result[current_key] = line[len(key)+1:].strip()
#                 matched = True
#                 break
#         if not matched and current_key and line:
#             result[current_key] += ' ' + line
#     for k in result:
#         result[k] = result[k].strip()
#     return result


# def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
#     if not knowledge:
#         return ""
#     lines = ["=== STEP KNOWLEDGE FROM PAST EXPERIENCE ==="]
#     for sk in knowledge:
#         icon = "CONFIRMED" if sk.positive else "CORRECTION"
#         lines.append(f"[{icon}] {sk.rule}")
#     lines.append("=== END STEP KNOWLEDGE ===\n")
#     return '\n'.join(lines)


# STAR_STEP_INSTRUCTION = (
#     "\n\nNow respond using EXACTLY these labels (as shown in examples above):\n"
#     "THOUGHT: <your reasoning>\n"
#     "ACTION: <Search[X] or Lookup[X] or Finish[X]>\n"
#     "EXPECTED: <what you expect this action to return>\n"
#     "NEXT_INTENT: <what you plan to do after this, e.g. 'lookup birth year'>\n"
#     "CORRECTION: <only if previous EXPECTED was wrong — one generalizable rule. "
#     "Skip this line entirely if first step or prediction was accurate.>\n"
# )


# # ---------------------------------------------------------------------------
# # STAR ReactAgent — RAR + StepKnowledgeStore
# # ---------------------------------------------------------------------------

# class STARReactAgent:
#     """
#     STAR = RAR trajectory retrieval + step-level knowledge retrieval.

#     RAR component:
#       - TrajectoryStore shared across all agents
#       - After failure: retrieve top-k trajectories by attention
#       - Use retrieved trajectories + their reflections to generate new reflection
#       - Agent sees reflections from retrieved trajectories only (not all)

#     STAR addition:
#       - StepKnowledgeStore shared across all agents
#       - Before every action: retrieve top-k relevant rules
#       - One LLM call per step: THOUGHT + ACTION + EXPECTED + NEXT_INTENT + CORRECTION
#       - Mismatch injection: NOTE added when prev EXPECTED != actual
#     """

#     def __init__(self,
#                  question: str,
#                  key: str,
#                  max_steps: int = 6,
#                  react_llm: AnyOpenAILLM = None,
#                  reflect_llm: AnyOpenAILLM = None,
#                  trajectory_store: TrajectoryStore = None,   # ← RAR
#                  knowledge_store: StepKnowledgeStore = None,  # ← STAR
#                  retrieval_k: int = 3,
#                  use_reflection: bool = True):

#         self.question          = question
#         self.key               = key
#         self.max_steps         = max_steps
#         self.llm               = react_llm  or AnyOpenAILLM()
#         self.reflect_llm       = reflect_llm or AnyOpenAILLM()
#         self.trajectory_store  = trajectory_store if trajectory_store is not None \
#                                  else TrajectoryStore()
#         self.knowledge_store   = knowledge_store if knowledge_store is not None \
#                                  else StepKnowledgeStore()
#         self.retrieval_k       = retrieval_k
#         self.use_reflection    = use_reflection
#         self.react_examples    = WEBTHINK_SIMPLE2
#         self.reflect_examples  = REFLECTIONS
#         self.enc               = tiktoken.encoding_for_model("text-davinci-003")
#         # reflections_str shows only reflections from retrieved trajectories
#         self.reflections_str   = ''
#         self.docstore          = DocstoreExplorer(Wikipedia())
#         self.__reset_agent()

#     # ------------------------------------------------------------------
#     # Run
#     # ------------------------------------------------------------------

#     def run(self, reset: bool = True,
#             reflect_strategy: ReflexionStrategy = ReflexionStrategy.STAR) -> None:
#         if (self.is_finished() or self.is_halted()) and not self.is_correct():
#             if self.use_reflection:
#                 self._reflect()
#         if reset:
#             self.__reset_agent()
#         while not self.is_halted() and not self.is_finished():
#             self.step()

#     # ------------------------------------------------------------------
#     # Step — one LLM call does everything
#     # ------------------------------------------------------------------

#     def step(self) -> None:
#         # 1. Use pre-fetched step knowledge from previous step's NEXT_INTENT
#         knowledge_str = format_step_knowledge(self._prefetched_knowledge)

#         # 2. One LLM call
#         prompt = self._build_agent_prompt(knowledge_str)
#         raw    = self.llm(prompt, REACT_SYSTEM_PROMPT) or ''
#         parsed = parse_structured_response(raw)

#         thought     = parsed['thought']
#         action_str  = parsed['action']
#         expected    = parsed['expected']
#         next_intent = parsed['next_intent']
#         correction  = parsed['correction']

#         # Fallback if structured parsing failed entirely
#         if not thought and not action_str:
#             print('  [STAR] Structured parse failed — falling back to ReAct format')
#             for line in raw.split('\n'):
#                 line = line.strip()
#                 if not thought and re.match(r'Thought\s*\d*\s*:', line, re.IGNORECASE):
#                     thought = line.split(':', 1)[-1].strip()
#                 if not action_str and any(a in line for a in ['Search[', 'Lookup[', 'Finish[']):
#                     action_str = line.strip()

#         # 3. Update scratchpad
#         self.scratchpad += f'\nThought {self.step_n}: {thought}'
#         self.scratchpad += f'\nAction {self.step_n}: {action_str}'
#         print(f'Thought {self.step_n}: {thought[:80]}')
#         print(f'ACTION=> {action_str}')
#         print("EXPECTED:", expected)
#         print("NEXT_INTENT:", next_intent)
#         print('CORRECTION:', correction)

#         action_type, argument = '', ''
#         try:
#             action_type, argument = parse_action(action_str)
#         except Exception:
#             print("Invalid Action")

#         self.scratchpad += f'\nObservation {self.step_n}: '

#         # 4. Execute action
#         if action_type == 'Finish':
#             self.answer = argument
#             observation = 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
#             self.scratchpad += observation
#             self.finished   = True
#             print(observation)
#             # Store successful trajectory in RAR store
#             self.trajectory_store.add(TrajectoryRecord(
#                 question    = self.question,
#                 scratchpad  = self.scratchpad,
#                 reflection  = '',
#                 success     = True,
#                 error_class = 'SUCCESS',
#             ))
#             self.step_n += 1
#             return

#         if action_type == 'Search':
#             try:
#                 observation = format_step(self.docstore.search(argument))
#             except Exception as e:
#                 print(e)
#                 observation = 'Could not find that page, please try again.'
#         elif action_type == 'Lookup':
#             try:
#                 observation = format_step(self.docstore.lookup(argument))
#             except ValueError:
#                 observation = ('The last page Searched was not found, '
#                                'so you cannot Lookup a keyword in it. '
#                                'Please try one of the similar pages given.')
#         else:
#             observation = ('Invalid Action. '
#                            'Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')

#         self.scratchpad += observation
#         print(f'Observation {self.step_n}: {observation[:100]}')

#         # 5. Store step knowledge — no extra LLM call
#         if correction and len(correction) > 15:
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = next_intent or action_str,
#                 rule          = correction,
#                 positive      = False,
#             ))
#             print(f'  [STAR] Correction stored: {correction[:80]}')
#         elif expected and observation and self._prediction_matched(expected, observation):
#             rule = f"{action_type}[{argument}] returns {expected[:100]}"
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = next_intent or action_str,
#                 rule          = rule,
#                 positive      = True,
#             ))

#         # Store prev expected/observation for mismatch injection next step
#         self._prev_expected    = expected
#         self._prev_observation = observation

#         # 6. Pre-fetch step knowledge for next step using NEXT_INTENT
#         if next_intent:
#             self._prefetched_knowledge = self.knowledge_store.retrieve(
#                 next_intent, k=self.retrieval_k)
#             if self._prefetched_knowledge:
#                 print(f'  [STAR] Pre-fetched {len(self._prefetched_knowledge)} rules '
#                       f'for: {next_intent[:60]}')
#         else:
#             self._prefetched_knowledge = []

#         self.step_n += 1

#     # ------------------------------------------------------------------
#     # Prediction match
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _prediction_matched(expected: str, actual: str) -> bool:
#         stopwords = {'the','a','an','is','was','will','to','of','in','and','or','it','this','that'}
#         exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
#         act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
#         if not exp_words:
#             return False
#         return len(exp_words & act_words) / len(exp_words) > 0.4

#     # ------------------------------------------------------------------
#     # Reflection — RAR style: retrieve trajectories, use their reflections
#     # as context to generate new reflection, store trajectory
#     # ------------------------------------------------------------------

#     def _reflect(self) -> None:
#         print('  [STAR] Reflecting...')

#         # ── RAR: classify error (same signature as retrieval_agents.py) ───────
#         error_class = classify_error(
#             self.question, self.scratchpad, self.reflect_llm)
#         print(f'  Error class: {error_class}')

#         # ── RAR: retrieve similar trajectories ───────────────────────────────
#         retrieved = self.trajectory_store.retrieve(
#             question    = self.question,
#             error_class = error_class,
#             k           = self.retrieval_k,
#         )
#         print(f'  [STAR] Retrieved {len(retrieved)} trajectories '
#               f'({sum(1 for r in retrieved if r.success)} successes, '
#               f'{sum(1 for r in retrieved if not r.success)} failures)')

#         # ── STAR: step knowledge as additional context ────────────────────────
#         recent_knowledge = self.knowledge_store.retrieve(self.question, k=2)
#         knowledge_ctx    = format_step_knowledge(recent_knowledge)

#         # ── Build reflection prompt: same as RAR + step knowledge ─────────────
#         retrieved_context  = format_retrieved_trajectories(retrieved)
#         current_block = (
#             "\n=== CURRENT FAILED TRAJECTORY ===\n"
#             f"Question: {self.question}\n"
#             f"Error class: {error_class}\n\n"
#             f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
#         )
#         instruction = (
#             "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
#             "FAILED_STEP: <the step where reasoning went wrong>\n"
#             "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
#             "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial>\n"
#             "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
#         )
#         reflection_prompt = (
#             RETRIEVAL_REFLECTION_HEADER + retrieved_context +
#             current_block + instruction
#         )
#         if knowledge_ctx:
#             reflection_prompt = knowledge_ctx + "\n" + reflection_prompt

#         reflection = format_step(
#             self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
#         print(f'  Reflection: {reflection[:120]}...')

#         # ── Store trajectory (same as RAR) ────────────────────────────────────
#         self.trajectory_store.add(TrajectoryRecord(
#             question    = self.question,
#             scratchpad  = self.scratchpad,
#             reflection  = reflection,
#             success     = False,
#             error_class = error_class,
#         ))

#         # ── reflections_str: same format as RAR ───────────────────────────────
#         # last attempt scratchpad + current reflection only
#         # NOT all accumulated reflections — token efficient
#         self.reflections     = [reflection]
#         self.reflections_str = (
#             format_last_attempt(self.question, self.scratchpad) +
#             format_reflections(self.reflections,
#                                header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
#         )
#         print(self.reflections_str)

#     # ------------------------------------------------------------------
#     # Agent prompt
#     # ------------------------------------------------------------------

#     def _build_agent_prompt(self, knowledge_str: str = '') -> str:
#         parts = []
#         # Step knowledge (STAR) — injected before examples
#         if knowledge_str:
#             parts.append(knowledge_str)
#         parts.append(self.react_examples)
#         parts.append('(END OF EXAMPLES)')
#         # Reflections from retrieved trajectories (RAR) — not all reflections
#         if self.reflections_str:
#             parts.append(self.reflections_str)
#         parts.append(f"Question: {self.question}")
#         parts.append(f"Scratchpad:\n{self.scratchpad}")
#         # Mismatch injection — tells LLM to write CORRECTION
#         if (self._prev_expected and self._prev_observation
#                 and not self._prediction_matched(self._prev_expected,
#                                                   self._prev_observation)):
#             parts.append(
#                 f"NOTE: Your previous prediction was WRONG.\n"
#                 f"You expected: {self._prev_expected[:100]}\n"
#                 f"You got:      {self._prev_observation[:100]}\n"
#                 f"You MUST include a CORRECTION rule in your response."
#             )
#         parts.append(STAR_STEP_INSTRUCTION)
#         return '\n\n'.join(parts)

#     # ------------------------------------------------------------------
#     # Standard interface
#     # ------------------------------------------------------------------

#     def is_finished(self) -> bool: return self.finished
#     def is_correct(self)  -> bool: return EM(self.answer, self.key)

#     def is_halted(self) -> bool:
#         return (
#             (self.step_n > self.max_steps) or
#             (len(self.enc.encode(self._build_agent_prompt())) > 3896)
#         ) and not self.finished

#     def __reset_agent(self) -> None:
#         self.step_n                = 1
#         self.finished              = False
#         self.answer                = ''
#         self.scratchpad            = ''
#         self._prefetched_knowledge = []
#         self._prev_expected        = ''
#         self._prev_observation     = ''
#         # reflections_str persists across resets — updated in _reflect()

#     def set_qa(self, question: str, key: str) -> None:
#         self.question = question
#         self.key      = key


# # ---------------------------------------------------------------------------
# # String utilities
# # ---------------------------------------------------------------------------

# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

# def parse_action(string: str):
#     for action_type in ['Finish', 'Search', 'Lookup']:
#         match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
#         if match:
#             return action_type, match.group(1)
#     return None, None

# def format_step(step: str) -> str:
#     return step.strip('\n').strip().replace('\n', '') if step else ''

# def format_reflections(reflections: List[str],
#                        header: str = REFLECTION_HEADER) -> str:
#     if not reflections:
#         return ''
#     return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# def format_last_attempt(question: str, scratchpad: str,
#                         header: str = LAST_TRIAL_HEADER) -> str:
#     return (header + f'Question: {question}\n' +
#             truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() +
#             '\n(END PREVIOUS TRIAL)\n')

# def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600,
#                         tokenizer=gpt2_enc) -> str:
#     lines = scratchpad.split('\n')
#     observations = list(filter(lambda x: x.startswith('Observation'), lines))
#     observations_by_tokens = sorted(observations,
#                                     key=lambda x: len(tokenizer.encode(x)))
#     while (len(gpt2_enc.encode('\n'.join(lines))) > n_tokens
#            and observations_by_tokens):
#         largest = observations_by_tokens.pop(-1)
#         ind     = lines.index(largest)
#         lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
#     return '\n'.join(lines)

# def normalize_answer(s: str) -> str:
#     def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
#     def white_space_fix(text):  return " ".join(text.split())
#     def remove_punc(text):
#         return "".join(ch for ch in text if ch not in set(string.punctuation))
#     def lower(text): return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def EM(answer: str, key: str) -> bool:
#     return normalize_answer(answer) == normalize_answer(key)


# """
# STAR — Step-level Trajectory-Agnostic Retrieval

# STAR = RAR + StepKnowledgeStore

# RAR component (unchanged):
#   - TrajectoryStore: stores full episodes with reflections
#   - After failure: retrieve top-k similar trajectories + their reflections
#   - Use retrieved trajectories as context to generate new reflection
#   - Agent sees reflections from retrieved trajectories only (not all)

# STAR addition (on top of RAR):
#   - StepKnowledgeStore: stores atomic step-level rules
#   - Before every action: retrieve top-k relevant rules via attention
#   - LLM generates EXPECTED + NEXT_INTENT + optional CORRECTION per step
#   - Zero extra LLM calls vs RAR

# Retrieval counts (separated):
#   - knowledge_k: rules retrieved from StepKnowledgeStore per step (default 2)
#   - trajectory_k: trajectories retrieved at reflection time (default 5, max_failures=3, max_successes=2)
#   - step_reflection_k: trajectory reflections injected per step prompt (default 3)
# """

# import re
# import string
# from typing import List, Tuple, Optional
# from enum import Enum
# import numpy as np
# import tiktoken
# from langchain import Wikipedia
# from langchain.agents.react.base import DocstoreExplorer
# from langchain.prompts import PromptTemplate
# from llm import AnyOpenAILLM
# from prompts import (
#     REFLECTION_SYSTEM_PROMPT,
#     REACT_SYSTEM_PROMPT,
#     reflect_prompt,
#     react_reflect_agent_prompt,
#     REFLECTION_HEADER,
#     LAST_TRIAL_HEADER,
#     REFLECTION_AFTER_LAST_TRIAL_HEADER,
#     CLASSIFY_ERROR_SYSTEM_PROMPT,
# )
# from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, WEBTHINK_SIMPLE2
# from sentence_transformers import SentenceTransformer

# # ── Import RAR components — exact names from retrieval_agents.py ─────────────
# from retrieval_agents import (
#     TrajectoryRecord,
#     TrajectoryStore,
#     classify_error,
#     format_retrieved_trajectories,
#     RETRIEVAL_REFLECTION_HEADER,
#     format_last_attempt,
#     format_reflections,
# )


# class ReflexionStrategy(Enum):
#     NONE                           = 'base'
#     REFLEXION                      = 'reflexion'
#     RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'
#     STAR                           = 'star'


# # ---------------------------------------------------------------------------
# # Step Knowledge — STAR addition
# # ---------------------------------------------------------------------------

# class StepKnowledge:
#     """
#     One generalizable rule extracted from a single step.
#     positive=True  → prediction matched actual (confirmed rule)
#     positive=False → prediction differed from actual (corrective rule)
#     """
#     def __init__(self,
#                  action_intent: str,
#                  rule: str,
#                  positive: bool):
#         self.action_intent = action_intent
#         self.rule          = rule
#         self.positive      = positive
#         self._embedding: Optional[np.ndarray] = None

#     def embedding(self, embed_fn) -> np.ndarray:
#         if self._embedding is None:
#             self._embedding = embed_fn(self.action_intent + " " + self.rule)
#         return self._embedding


# class StepKnowledgeStore:
#     """Attention-weighted retrieval over step-level knowledge."""

#     _st_model = None

#     def __init__(self, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
#         self.knowledge:   List[StepKnowledge] = []
#         self.embed_fn     = self._sentence_transformer_embed
#         self.tau          = tau
#         self.adaptive_tau = adaptive_tau
#         self.mmr_lambda   = mmr_lambda

#     def add(self, knowledge: StepKnowledge) -> None:
#         # Deduplication: skip if very similar rule already exists
#         if self.knowledge:
#             q_emb = self.embed_fn(knowledge.action_intent + " " + knowledge.rule)
#             for existing in self.knowledge[-20:]:
#                 sim = float(np.dot(q_emb, existing.embedding(self.embed_fn)))
#                 if sim > 0.92:
#                     return
#         self.knowledge.append(knowledge)

#     def retrieve(self, action_intent: str, k: int = 2) -> List[StepKnowledge]:
#         if not self.knowledge:
#             return []
#         q_emb = self.embed_fn(action_intent)
#         d     = q_emb.shape[0]
#         tau   = (0.05 + 0.25 * min(len(self.knowledge) / 100.0, 1.0)
#                  if self.adaptive_tau else self.tau)
#         logits = [
#             float(np.dot(q_emb, sk.embedding(self.embed_fn))) / np.sqrt(d) / tau
#             for sk in self.knowledge
#         ]
#         logits_arr  = np.array(logits)
#         logits_arr -= logits_arr.max()
#         alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()
#         scored = sorted(zip(alphas.tolist(), self.knowledge),
#                         key=lambda x: x[0], reverse=True)
#         return self._mmr_select(scored, k)

#     def _mmr_select(self, scored, k):
#         if not scored:
#             return []
#         selected, candidates = [], list(scored)
#         while len(selected) < k and candidates:
#             if not selected:
#                 _, best = max(candidates, key=lambda x: x[0])
#             else:
#                 best_score, best = -1e9, None
#                 sel_embs = [s.embedding(self.embed_fn) for s in selected]
#                 for attn_score, sk in candidates:
#                     max_sim = max(
#                         float(np.dot(sk.embedding(self.embed_fn), se))
#                         for se in sel_embs
#                     )
#                     mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
#                     if mmr > best_score:
#                         best_score, best = mmr, sk
#             selected.append(best)
#             candidates = [(a, s) for a, s in candidates if s is not best]
#         return selected

#     @staticmethod
#     def _get_st_model():
#         if StepKnowledgeStore._st_model is None:
#             StepKnowledgeStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
#         return StepKnowledgeStore._st_model

#     @staticmethod
#     def _sentence_transformer_embed(text: str) -> np.ndarray:
#         return StepKnowledgeStore._get_st_model().encode(
#             text, normalize_embeddings=True).astype(np.float64)


# # ---------------------------------------------------------------------------
# # Structured output parser
# # ---------------------------------------------------------------------------

# def parse_structured_response(raw: str) -> dict:
#     result = {
#         'thought': '', 'action': '', 'expected': '',
#         'next_intent': '', 'correction': '',
#     }
#     if not raw:
#         return result
#     current_key = None
#     for line in raw.split('\n'):
#         line = line.strip()
#         matched = False
#         for key in ['THOUGHT', 'ACTION', 'EXPECTED', 'NEXT_INTENT', 'CORRECTION']:
#             if line.upper().startswith(f'{key}:'):
#                 current_key = key.lower()
#                 result[current_key] = line[len(key)+1:].strip()
#                 matched = True
#                 break
#         if not matched and current_key and line:
#             result[current_key] += ' ' + line
#     for k in result:
#         result[k] = result[k].strip()
#     return result


# def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
#     if not knowledge:
#         return ""
#     lines = ["=== STEP KNOWLEDGE FROM PAST EXPERIENCE ==="]
#     for sk in knowledge:
#         icon = "CONFIRMED" if sk.positive else "CORRECTION"
#         lines.append(f"[{icon}] {sk.rule}")
#     lines.append("=== END STEP KNOWLEDGE ===\n")
#     return '\n'.join(lines)


# STAR_STEP_INSTRUCTION = (
#     "\n\nNow respond using EXACTLY these labels (as shown in examples above):\n"
#     "THOUGHT: <your reasoning>\n"
#     "ACTION: <Search[X] or Lookup[X] or Finish[X]>\n"
#     "EXPECTED: <what you expect this action to return>\n"
#     "NEXT_INTENT: <what you plan to do after this, e.g. 'lookup birth year'>\n"
#     "CORRECTION: <only if previous EXPECTED was wrong — one generalizable rule. "
#     "Skip this line entirely if first step or prediction was accurate.>\n"
# )


# # ---------------------------------------------------------------------------
# # STAR ReactAgent — RAR + StepKnowledgeStore
# # ---------------------------------------------------------------------------

# class STARReactAgent:
#     """
#     STAR = RAR trajectory retrieval + step-level knowledge retrieval.

#     RAR component:
#       - TrajectoryStore shared across all agents
#       - After failure: retrieve trajectory_k trajectories (max_failures=3, max_successes=2)
#       - Full scratchpad + reflection stored in reflections_str (same as RAR)

#     STAR addition:
#       - StepKnowledgeStore shared across all agents
#       - Per step: inject knowledge_k rules + step_reflection_k trajectory reflections
#       - One LLM call per step: THOUGHT + ACTION + EXPECTED + NEXT_INTENT + CORRECTION
#       - Mismatch injection: NOTE added when prev EXPECTED != actual
#     """

#     def __init__(self,
#                  question: str,
#                  key: str,
#                  max_steps: int = 6,
#                  react_llm: AnyOpenAILLM = None,
#                  reflect_llm: AnyOpenAILLM = None,
#                  trajectory_store: TrajectoryStore = None,
#                  knowledge_store: StepKnowledgeStore = None,
#                  knowledge_k: int = 2,          # ← rules per step from StepKnowledgeStore
#                  trajectory_k: int = 5,          # ← trajectories at reflection time
#                  trajectory_max_failures: int = 3,
#                  trajectory_max_successes: int = 2,
#                  step_reflection_k: int = 3,     # ← trajectory reflections per step prompt
#                  use_reflection: bool = True):

#         self.question                = question
#         self.key                     = key
#         self.max_steps               = max_steps
#         self.llm                     = react_llm  or AnyOpenAILLM()
#         self.reflect_llm             = reflect_llm or AnyOpenAILLM()
#         self.trajectory_store        = trajectory_store if trajectory_store is not None \
#                                        else TrajectoryStore()
#         self.knowledge_store         = knowledge_store if knowledge_store is not None \
#                                        else StepKnowledgeStore()
#         self.knowledge_k             = knowledge_k
#         self.trajectory_k            = trajectory_k
#         self.trajectory_max_failures = trajectory_max_failures
#         self.trajectory_max_successes= trajectory_max_successes
#         self.step_reflection_k       = step_reflection_k
#         self.use_reflection          = use_reflection
#         self.react_examples          = WEBTHINK_SIMPLE2
#         self.reflect_examples        = REFLECTIONS
#         self.enc                     = tiktoken.encoding_for_model("text-davinci-003")
#         # reflections_str: last attempt scratchpad + reflection (same as RAR)
#         # persists across resets, updated in _reflect()
#         self.reflections_str         = ''
#         self.reflections             = []
#         self.docstore                = DocstoreExplorer(Wikipedia())
#         self.__reset_agent()

#     # ------------------------------------------------------------------
#     # Run
#     # ------------------------------------------------------------------

#     def run(self, reset: bool = True,
#             reflect_strategy: ReflexionStrategy = ReflexionStrategy.STAR) -> None:
#         if (self.is_finished() or self.is_halted()) and not self.is_correct():
#             if self.use_reflection:
#                 self._reflect()
#         if reset:
#             self.__reset_agent()
#         while not self.is_halted() and not self.is_finished():
#             self.step()

#     # ------------------------------------------------------------------
#     # Step — one LLM call does everything
#     # ------------------------------------------------------------------

#     def step(self) -> None:
#         # 1. Use pre-fetched step knowledge from previous step's NEXT_INTENT
#         #    knowledge_k rules from StepKnowledgeStore
#         knowledge_str = format_step_knowledge(self._prefetched_knowledge)

#         # 2. One LLM call
#         prompt = self._build_agent_prompt(knowledge_str)
#         raw    = self.llm(prompt, REACT_SYSTEM_PROMPT) or ''
#         parsed = parse_structured_response(raw)

#         thought     = parsed['thought']
#         action_str  = parsed['action']
#         expected    = parsed['expected']
#         next_intent = parsed['next_intent']
#         correction  = parsed['correction']

#         # Fallback if structured parsing failed entirely
#         if not thought and not action_str:
#             print('  [STAR] Structured parse failed — falling back to ReAct format')
#             for line in raw.split('\n'):
#                 line = line.strip()
#                 if not thought and re.match(r'Thought\s*\d*\s*:', line, re.IGNORECASE):
#                     thought = line.split(':', 1)[-1].strip()
#                 if not action_str and any(a in line for a in ['Search[', 'Lookup[', 'Finish[']):
#                     action_str = line.strip()

#         # 3. Update scratchpad
#         self.scratchpad += f'\nThought {self.step_n}: {thought}'
#         self.scratchpad += f'\nAction {self.step_n}: {action_str}'
#         print(f'Thought {self.step_n}: {thought[:80]}')
#         print(f'ACTION=> {action_str}')
#         print("EXPECTED:", expected)
#         print("NEXT_INTENT:", next_intent)
#         print('CORRECTION:', correction)

#         action_type, argument = '', ''
#         try:
#             action_type, argument = parse_action(action_str)
#         except Exception:
#             print("Invalid Action")

#         self.scratchpad += f'\nObservation {self.step_n}: '

#         # 4. Execute action
#         if action_type == 'Finish':
#             self.answer = argument
#             observation = 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
#             self.scratchpad += observation
#             self.finished   = True
#             print(observation)
#             self.trajectory_store.add(TrajectoryRecord(
#                 question    = self.question,
#                 scratchpad  = self.scratchpad,
#                 reflection  = '',
#                 success     = True,
#                 error_class = 'SUCCESS',
#             ))
#             self.step_n += 1
#             return

#         if action_type == 'Search':
#             try:
#                 observation = format_step(self.docstore.search(argument))
#             except Exception as e:
#                 print(e)
#                 observation = 'Could not find that page, please try again.'
#         elif action_type == 'Lookup':
#             try:
#                 observation = format_step(self.docstore.lookup(argument))
#             except ValueError:
#                 observation = ('The last page Searched was not found, '
#                                'so you cannot Lookup a keyword in it. '
#                                'Please try one of the similar pages given.')
#         else:
#             observation = ('Invalid Action. '
#                            'Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')

#         self.scratchpad += observation
#         print(f'Observation {self.step_n}: {observation[:100]}')

#         # 5. Store step knowledge — no extra LLM call
#         if correction and len(correction) > 15:
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = next_intent or action_str,
#                 rule          = correction,
#                 positive      = False,
#             ))
#             print(f'  [STAR] Correction stored: {correction[:80]}')
#         elif expected and observation and self._prediction_matched(expected, observation):
#             rule = f"{action_type}[{argument}] returns {expected[:100]}"
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = next_intent or action_str,
#                 rule          = rule,
#                 positive      = True,
#             ))

#         # Store prev expected/observation for mismatch injection next step
#         self._prev_expected    = expected
#         self._prev_observation = observation

#         # 6. Pre-fetch knowledge_k rules for next step using NEXT_INTENT
#         if next_intent:
#             self._prefetched_knowledge = self.knowledge_store.retrieve(
#                 next_intent, k=self.knowledge_k)
#             if self._prefetched_knowledge:
#                 print(f'  [STAR] Pre-fetched {len(self._prefetched_knowledge)} rules '
#                       f'for: {next_intent[:60]}')
#         else:
#             self._prefetched_knowledge = []

#         self.step_n += 1

#     # ------------------------------------------------------------------
#     # Prediction match
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _prediction_matched(expected: str, actual: str) -> bool:
#         stopwords = {'the','a','an','is','was','will','to','of','in','and','or','it','this','that'}
#         exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
#         act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
#         if not exp_words:
#             return False
#         return len(exp_words & act_words) / len(exp_words) > 0.4

#     # ------------------------------------------------------------------
#     # Reflection — RAR style with separated trajectory_k
#     # ------------------------------------------------------------------

#     def _reflect(self) -> None:
#         print('  [STAR] Reflecting...')

#         # ── RAR: classify error ───────────────────────────────────────────────
#         error_class = classify_error(
#             self.question, self.scratchpad, self.reflect_llm)
#         print(f'  Error class: {error_class}')

#         # ── RAR: retrieve trajectory_k trajectories (max_failures/max_successes)
#         retrieved = self.trajectory_store.retrieve(
#             question      = self.question,
#             error_class   = error_class,
#             k             = self.trajectory_k,
#             max_failures  = self.trajectory_max_failures,
#             max_successes = self.trajectory_max_successes,
#         )
#         print(f'  [STAR] Retrieved {len(retrieved)} trajectories '
#               f'({sum(1 for r in retrieved if r.success)} successes, '
#               f'{sum(1 for r in retrieved if not r.success)} failures)')

#         # ── STAR: step knowledge as additional context ────────────────────────
#         recent_knowledge = self.knowledge_store.retrieve(self.question, k=2)
#         knowledge_ctx    = format_step_knowledge(recent_knowledge)

#         # ── Build reflection prompt ───────────────────────────────────────────
#         retrieved_context = format_retrieved_trajectories(retrieved)
#         current_block = (
#             "\n=== CURRENT FAILED TRAJECTORY ===\n"
#             f"Question: {self.question}\n"
#             f"Error class: {error_class}\n\n"
#             f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
#         )
#         instruction = (
#             "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
#             "FAILED_STEP: <the step where reasoning went wrong>\n"
#             "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
#             "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial>\n"
#             "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
#         )
#         reflection_prompt = (
#             RETRIEVAL_REFLECTION_HEADER + retrieved_context +
#             current_block + instruction
#         )
#         if knowledge_ctx:
#             reflection_prompt = knowledge_ctx + "\n" + reflection_prompt

#         reflection = format_step(
#             self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
#         print(f'  Reflection: {reflection[:120]}...')

#         # ── Store trajectory ──────────────────────────────────────────────────
#         self.trajectory_store.add(TrajectoryRecord(
#             question    = self.question,
#             scratchpad  = self.scratchpad,
#             reflection  = reflection,
#             success     = False,
#             error_class = error_class,
#         ))

#         # ── reflections_str: last attempt scratchpad + reflection (same as RAR)
#         # Full scratchpad preserved here — used in agent prompt next trial
#         self.reflections = [reflection]
#         self.reflections_str = (
#             format_last_attempt(self.question, self.scratchpad) +
#             format_reflections(self.reflections,
#                                header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
#         )
#         print(self.reflections_str)

#     # ------------------------------------------------------------------
#     # Agent prompt — knowledge_k rules + step_reflection_k reflections
#     # ------------------------------------------------------------------

#     def _build_agent_prompt(self, knowledge_str: str = '') -> str:
#         parts = []

#         # 1. Step knowledge (knowledge_k rules) — injected before examples
#         if knowledge_str:
#             parts.append(knowledge_str)

#         parts.append(self.react_examples)
#         parts.append('(END OF EXAMPLES)')

#         # 2. Trajectory reflections (step_reflection_k, reflection text only)
#         #    Retrieved fresh each step by question similarity
#         #    If store has trajectories: retrieve and show only reflection text
#         #    If store empty: fall back to reflections_str (last attempt + reflection)
#         if self.trajectory_store.records:
#             step_retrieved = self.trajectory_store.retrieve(
#                 question      = self.question,
#                 error_class   = 'UNKNOWN',   # no error class at step time
#                 k             = self.step_reflection_k,
#                 max_failures  = self.step_reflection_k - 1,
#                 max_successes = 1,
#             )
#             step_refs = [r.reflection for r in step_retrieved if r.reflection]
#             if step_refs:
#                 ref_block = (
#                     REFLECTION_HEADER + 'Reflections:\n- ' +
#                     '\n- '.join(r.strip()[:200] for r in step_refs)
#                 )
#                 parts.append(ref_block)
#         elif self.reflections_str:
#             # Early trials — no trajectories yet, use stored reflections_str
#             parts.append(self.reflections_str)

#         parts.append(f"Question: {self.question}")
#         parts.append(f"Scratchpad:\n{self.scratchpad}")

#         # 3. Mismatch injection
#         if (self._prev_expected and self._prev_observation
#                 and not self._prediction_matched(self._prev_expected,
#                                                   self._prev_observation)):
#             parts.append(
#                 f"NOTE: Your previous prediction was WRONG.\n"
#                 f"You expected: {self._prev_expected[:100]}\n"
#                 f"You got:      {self._prev_observation[:100]}\n"
#                 f"You MUST include a CORRECTION rule in your response."
#             )

#         parts.append(STAR_STEP_INSTRUCTION)
#         return '\n\n'.join(parts)

#     # ------------------------------------------------------------------
#     # Standard interface
#     # ------------------------------------------------------------------

#     def is_finished(self) -> bool: return self.finished
#     def is_correct(self)  -> bool: return EM(self.answer, self.key)

#     def is_halted(self) -> bool:
#         return (
#             (self.step_n > self.max_steps) or
#             (len(self.enc.encode(self._build_agent_prompt())) > 3896)
#         ) and not self.finished

#     def __reset_agent(self) -> None:
#         self.step_n                = 1
#         self.finished              = False
#         self.answer                = ''
#         self.scratchpad            = ''
#         self._prefetched_knowledge = []
#         self._prev_expected        = ''
#         self._prev_observation     = ''
#         # reflections_str persists across resets — updated in _reflect()

#     def set_qa(self, question: str, key: str) -> None:
#         self.question = question
#         self.key      = key


# # ---------------------------------------------------------------------------
# # String utilities
# # ---------------------------------------------------------------------------

# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

# def parse_action(string: str):
#     for action_type in ['Finish', 'Search', 'Lookup']:
#         match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
#         if match:
#             return action_type, match.group(1)
#     return None, None

# def format_step(step: str) -> str:
#     return step.strip('\n').strip().replace('\n', '') if step else ''

# def format_reflections(reflections: List[str],
#                        header: str = REFLECTION_HEADER) -> str:
#     if not reflections:
#         return ''
#     return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# def format_last_attempt(question: str, scratchpad: str,
#                         header: str = LAST_TRIAL_HEADER) -> str:
#     return (header + f'Question: {question}\n' +
#             truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() +
#             '\n(END PREVIOUS TRIAL)\n')

# def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600,
#                         tokenizer=gpt2_enc) -> str:
#     lines = scratchpad.split('\n')
#     observations = list(filter(lambda x: x.startswith('Observation'), lines))
#     observations_by_tokens = sorted(observations,
#                                     key=lambda x: len(tokenizer.encode(x)))
#     while (len(gpt2_enc.encode('\n'.join(lines))) > n_tokens
#            and observations_by_tokens):
#         largest = observations_by_tokens.pop(-1)
#         ind     = lines.index(largest)
#         lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
#     return '\n'.join(lines)

# def normalize_answer(s: str) -> str:
#     def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
#     def white_space_fix(text):  return " ".join(text.split())
#     def remove_punc(text):
#         return "".join(ch for ch in text if ch not in set(string.punctuation))
#     def lower(text): return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def EM(answer: str, key: str) -> bool:
#     return normalize_answer(answer) == normalize_answer(key)


# """
# STAR — Step-level Trajectory-Agnostic Retrieval

# STAR = RAR + StepKnowledgeStore

# RAR component (unchanged):
#   - TrajectoryStore: stores full episodes with reflections
#   - After failure: retrieve trajectory_k trajectories (max_failures=3, max_successes=2)
#   - Full scratchpad + reflection in reflections_str (same as RAR)

# STAR addition (on top of RAR):
#   - StepKnowledgeStore: stores atomic step-level rules under abstract KEYs
#   - LLM generates KEY (3-5 word abstract pattern) instead of NEXT_INTENT
#   - Step 1: classify_question_type(question) used as retrieval query (no LLM call)
#   - Step N: KEY from step N-1 used as retrieval query for step N
#   - Storage: rule stored under LLM-generated KEY (abstract, reusable across questions)
#   - Per step prompt: knowledge_k rules + step_reflection_k trajectory reflections
#   - At reflection: trajectory_k trajectories with full scratchpad as context

# Retrieval counts (all separated):
#   - knowledge_k:              rules per step from StepKnowledgeStore (default 2)
#   - step_reflection_k:        trajectory reflections per step prompt (default 3)
#   - trajectory_k:             trajectories at reflection time (default 5)
#   - trajectory_max_failures:  max failure trajectories at reflection (default 3)
#   - trajectory_max_successes: max success trajectories at reflection (default 2)
# """

# import re
# import string
# from typing import List, Optional
# from enum import Enum
# import numpy as np
# import tiktoken
# from langchain import Wikipedia
# from langchain.agents.react.base import DocstoreExplorer
# from llm import AnyOpenAILLM
# from prompts import (
#     REFLECTION_SYSTEM_PROMPT,
#     REACT_SYSTEM_PROMPT,
#     REFLECTION_HEADER,
#     LAST_TRIAL_HEADER,
#     REFLECTION_AFTER_LAST_TRIAL_HEADER,
#     CLASSIFY_ERROR_SYSTEM_PROMPT,
# )
# from fewshots import REFLECTIONS, WEBTHINK_SIMPLE2
# from sentence_transformers import SentenceTransformer

# from retrieval_agents import (
#     TrajectoryRecord,
#     TrajectoryStore,
#     classify_error,
#     format_retrieved_trajectories,
#     RETRIEVAL_REFLECTION_HEADER,
#     format_last_attempt,
#     format_reflections,
# )


# class ReflexionStrategy(Enum):
#     NONE                           = 'base'
#     REFLEXION                      = 'reflexion'
#     RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'
#     STAR                           = 'star'


# # ---------------------------------------------------------------------------
# # Question type classifier — no LLM call, used for step 1 retrieval
# # ---------------------------------------------------------------------------

# def classify_question_type(question: str) -> str:
#     """
#     Classify question into an abstract key for step-1 knowledge retrieval.
#     No LLM call — pure keyword heuristics.
#     """
#     q = question.lower()
#     if any(w in q for w in ['both', 'same', 'common', 'share', 'also']):
#         return 'search-comparison'
#     if any(w in q for w in ['how many', 'number of', 'count']):
#         return 'search-count'
#     if any(w in q for w in ['where', 'located', 'headquartered', 'based']):
#         return 'search-location'
#     if any(w in q for w in ['when', 'year', 'born', 'died', 'founded', 'date']):
#         return 'search-date'
#     if any(w in q for w in ['which', 'what team', 'what country', 'what city']):
#         return 'search-entity-lookup'
#     if any(w in q for w in ['who', 'directed', 'created', 'wrote', 'invented',
#                              'played', 'founded', 'authored']):
#         return 'search-person-role'
#     if any(w in q for w in ['what type', 'what kind', 'what genre', 'what sport']):
#         return 'search-category'
#     return 'search-multi-hop'


# # ---------------------------------------------------------------------------
# # Step Knowledge
# # ---------------------------------------------------------------------------

# class StepKnowledge:
#     """
#     One generalizable rule extracted from a single step.
#     Stored under an abstract KEY generated by the LLM.
#     positive=True  → prediction matched (confirmed rule)
#     positive=False → prediction differed (corrective rule)
#     """
#     def __init__(self, action_intent: str, rule: str, positive: bool):
#         self.action_intent = action_intent
#         self.rule          = rule
#         self.positive      = positive
#         self._embedding: Optional[np.ndarray] = None

#     def embedding(self, embed_fn) -> np.ndarray:
#         if self._embedding is None:
#             self._embedding = embed_fn(self.action_intent + " " + self.rule)
#         return self._embedding


# class StepKnowledgeStore:
#     """Attention-weighted retrieval over step-level knowledge."""

#     _st_model = None

#     def __init__(self, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
#         self.knowledge:   List[StepKnowledge] = []
#         self.embed_fn     = self._sentence_transformer_embed
#         self.tau          = tau
#         self.adaptive_tau = adaptive_tau
#         self.mmr_lambda   = mmr_lambda

#     def add(self, knowledge: StepKnowledge) -> None:
#         if self.knowledge:
#             q_emb = self.embed_fn(knowledge.action_intent + " " + knowledge.rule)
#             for existing in self.knowledge[-20:]:
#                 sim = float(np.dot(q_emb, existing.embedding(self.embed_fn)))
#                 if sim > 0.92:
#                     return
#         self.knowledge.append(knowledge)

#     def retrieve(self, query: str, k: int = 2) -> List[StepKnowledge]:
#         if not self.knowledge:
#             return []
#         q_emb = self.embed_fn(query)
#         d     = q_emb.shape[0]
#         tau   = (0.05 + 0.25 * min(len(self.knowledge) / 100.0, 1.0)
#                  if self.adaptive_tau else self.tau)
#         logits = [
#             float(np.dot(q_emb, sk.embedding(self.embed_fn))) / np.sqrt(d) / tau
#             for sk in self.knowledge
#         ]
#         logits_arr  = np.array(logits)
#         logits_arr -= logits_arr.max()
#         alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()
#         scored = sorted(zip(alphas.tolist(), self.knowledge),
#                         key=lambda x: x[0], reverse=True)
#         return self._mmr_select(scored, k)

#     def _mmr_select(self, scored, k):
#         if not scored:
#             return []
#         selected, candidates = [], list(scored)
#         while len(selected) < k and candidates:
#             if not selected:
#                 _, best = max(candidates, key=lambda x: x[0])
#             else:
#                 best_score, best = -1e9, None
#                 sel_embs = [s.embedding(self.embed_fn) for s in selected]
#                 for attn_score, sk in candidates:
#                     max_sim = max(
#                         float(np.dot(sk.embedding(self.embed_fn), se))
#                         for se in sel_embs
#                     )
#                     mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
#                     if mmr > best_score:
#                         best_score, best = mmr, sk
#             selected.append(best)
#             candidates = [(a, s) for a, s in candidates if s is not best]
#         return selected

#     @staticmethod
#     def _get_st_model():
#         if StepKnowledgeStore._st_model is None:
#             StepKnowledgeStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
#         return StepKnowledgeStore._st_model

#     @staticmethod
#     def _sentence_transformer_embed(text: str) -> np.ndarray:
#         return StepKnowledgeStore._get_st_model().encode(
#             text, normalize_embeddings=True).astype(np.float64)


# # ---------------------------------------------------------------------------
# # Structured output parser — KEY replaces NEXT_INTENT
# # ---------------------------------------------------------------------------

# def parse_structured_response(raw: str) -> dict:
#     """
#     Parses LLM output with fields:
#       THOUGHT, ACTION, EXPECTED, KEY, CORRECTION
#     KEY is a 3-5 word abstract pattern e.g. 'search-person-role', 'lookup-birth-year'
#     used as storage key and next-step retrieval query.
#     """
#     result = {
#         'thought': '', 'action': '', 'expected': '',
#         'key': '', 'correction': '',
#     }
#     if not raw:
#         return result
#     current_key = None
#     for line in raw.split('\n'):
#         line = line.strip()
#         matched = False
#         for k in ['THOUGHT', 'ACTION', 'EXPECTED', 'KEY', 'CORRECTION']:
#             if line.upper().startswith(f'{k}:'):
#                 current_key = k.lower()
#                 result[current_key] = line[len(k)+1:].strip()
#                 matched = True
#                 break
#         if not matched and current_key and line:
#             result[current_key] += ' ' + line
#     for k in result:
#         result[k] = result[k].strip()
#     return result


# def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
#     """Compact format to minimise token cost."""
#     if not knowledge:
#         return ""
#     lines = ["[STEP RULES]"]
#     for sk in knowledge:
#         icon = "OK" if sk.positive else "FIX"
#         lines.append(f"{icon}: {sk.rule[:120]}")
#     lines.append("")
#     return '\n'.join(lines)


# # Compact instruction — KEY replaces NEXT_INTENT, shorter overall
# STAR_STEP_INSTRUCTION = (
#     "\nRespond with EXACTLY these labels:\n"
#     "THOUGHT: <reasoning>\n"
#     "ACTION: <Search[X] or Lookup[X] or Finish[X]>\n"
#     "EXPECTED: <brief expected result>\n"
#     "KEY: <3-5 word abstract pattern for this step, e.g. search-person-role, "
#     "lookup-birth-year, search-film-director>\n"
#     "CORRECTION: <if prev EXPECTED was wrong — one generalizable rule. "
#     "Omit entirely if first step or prediction was correct.>\n"
# )


# # ---------------------------------------------------------------------------
# # STAR ReactAgent
# # ---------------------------------------------------------------------------

# class STARReactAgent:
#     """
#     STAR = RAR + StepKnowledgeStore with KEY-based abstract retrieval.

#     Knowledge flow per step:
#       Step 1: classify_question_type(question) → retrieval query (no LLM call)
#       Step N: _prev_key (KEY from step N-1) → retrieval query
#       Storage: rule stored under LLM-generated KEY (abstract, reusable)
#       Mismatch: NOTE injected when prev EXPECTED != actual observation

#     Reflection flow:
#       Retrieve trajectory_k trajectories with full scratchpad
#       Generate new reflection using retrieved context + step knowledge
#       Store reflection text only in reflections_str (not full scratchpad)
#       Per-step: inject step_reflection_k reflection texts from trajectory store
#     """

#     def __init__(self,
#                  question: str,
#                  key: str,
#                  max_steps: int = 6,
#                  react_llm: AnyOpenAILLM = None,
#                  reflect_llm: AnyOpenAILLM = None,
#                  trajectory_store: TrajectoryStore = None,
#                  knowledge_store: StepKnowledgeStore = None,
#                  knowledge_k: int = 2,
#                  trajectory_k: int = 5,
#                  trajectory_max_failures: int = 3,
#                  trajectory_max_successes: int = 2,
#                  step_reflection_k: int = 3,
#                  use_reflection: bool = True):

#         self.question                = question
#         self.key                     = key
#         self.max_steps               = max_steps
#         self.llm                     = react_llm  or AnyOpenAILLM()
#         self.reflect_llm             = reflect_llm or AnyOpenAILLM()
#         self.trajectory_store        = trajectory_store if trajectory_store is not None \
#                                        else TrajectoryStore()
#         self.knowledge_store         = knowledge_store if knowledge_store is not None \
#                                        else StepKnowledgeStore()
#         self.knowledge_k             = knowledge_k
#         self.trajectory_k            = trajectory_k
#         self.trajectory_max_failures = trajectory_max_failures
#         self.trajectory_max_successes= trajectory_max_successes
#         self.step_reflection_k       = step_reflection_k
#         self.use_reflection          = use_reflection
#         self.react_examples          = WEBTHINK_SIMPLE2
#         self.reflect_examples        = REFLECTIONS
#         self.enc                     = tiktoken.encoding_for_model("text-davinci-003")
#         self.reflections             = []
#         self.reflections_str         = ''   # persists across resets, set in _reflect()
#         self.docstore                = DocstoreExplorer(Wikipedia())
#         self.__reset_agent()

#     # ------------------------------------------------------------------
#     # Run
#     # ------------------------------------------------------------------

#     def run(self, reset: bool = True,
#             reflect_strategy: ReflexionStrategy = ReflexionStrategy.STAR) -> None:
#         if (self.is_finished() or self.is_halted()) and not self.is_correct():
#             if self.use_reflection:
#                 self._reflect()
#         if reset:
#             self.__reset_agent()
#         while not self.is_halted() and not self.is_finished():
#             self.step()

#     # ------------------------------------------------------------------
#     # Step
#     # ------------------------------------------------------------------

#     def step(self) -> None:
#         # ── 1. Retrieval query ────────────────────────────────────────────────
#         # Step 1: heuristic question type (no LLM call, no prior key)
#         # Step N: KEY generated by LLM at step N-1
#         retrieval_query = self._prev_key if self._prev_key \
#                           else classify_question_type(self.question)

#         # Retrieve knowledge_k rules from StepKnowledgeStore
#         retrieved_knowledge = self.knowledge_store.retrieve(
#             retrieval_query, k=self.knowledge_k)
#         knowledge_str = format_step_knowledge(retrieved_knowledge)
#         if retrieved_knowledge:
#             print(f'  [STAR] Retrieved {len(retrieved_knowledge)} rules '
#                   f'for key: "{retrieval_query}"')

#         # ── 2. One LLM call ───────────────────────────────────────────────────
#         prompt = self._build_agent_prompt(knowledge_str)
#         raw    = self.llm(prompt, REACT_SYSTEM_PROMPT) or ''
#         parsed = parse_structured_response(raw)

#         thought    = parsed['thought']
#         action_str = parsed['action']
#         expected   = parsed['expected']
#         step_key   = parsed['key']        # abstract key for storage + next retrieval
#         correction = parsed['correction']

#         # Fallback
#         if not thought and not action_str:
#             print('  [STAR] Parse failed — falling back to ReAct format')
#             for line in raw.split('\n'):
#                 line = line.strip()
#                 if not thought and re.match(r'Thought\s*\d*\s*:', line, re.IGNORECASE):
#                     thought = line.split(':', 1)[-1].strip()
#                 if not action_str and any(a in line for a in ['Search[', 'Lookup[', 'Finish[']):
#                     action_str = line.strip()

#         # ── 3. Update scratchpad (Thought + Action only, same as ReAct) ───────
#         self.scratchpad += f'\nThought {self.step_n}: {thought}'
#         self.scratchpad += f'\nAction {self.step_n}: {action_str}'
#         print(f'Thought {self.step_n}: {thought[:80]}')
#         print(f'ACTION=> {action_str}')
#         print(f'EXPECTED: {expected}')
#         print(f'KEY: {step_key}')
#         print(f'CORRECTION: {correction}')

#         action_type, argument = '', ''
#         try:
#             action_type, argument = parse_action(action_str)
#         except Exception:
#             print("Invalid Action")

#         self.scratchpad += f'\nObservation {self.step_n}: '

#         # ── 4. Execute action ─────────────────────────────────────────────────
#         if action_type == 'Finish':
#             self.answer = argument
#             observation = 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
#             self.scratchpad += observation
#             self.finished   = True
#             print(observation)
#             self.trajectory_store.add(TrajectoryRecord(
#                 question=self.question, scratchpad=self.scratchpad,
#                 reflection='', success=True, error_class='SUCCESS',
#             ))
#             self.step_n += 1
#             return

#         if action_type == 'Search':
#             try:
#                 observation = format_step(self.docstore.search(argument))
#             except Exception as e:
#                 print(e)
#                 observation = 'Could not find that page, please try again.'
#         elif action_type == 'Lookup':
#             try:
#                 observation = format_step(self.docstore.lookup(argument))
#             except ValueError:
#                 observation = ('The last page Searched was not found, '
#                                'so you cannot Lookup a keyword in it. '
#                                'Please try one of the similar pages given.')
#         else:
#             observation = ('Invalid Action. '
#                            'Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')

#         self.scratchpad += observation
#         print(f'Observation {self.step_n}: {observation[:100]}')

#         # ── 5. Store knowledge under abstract KEY — no LLM call ───────────────
#         # Storage key: LLM-generated KEY (abstract, reusable across questions)
#         # Falls back to abstract action type if KEY not generated
#         storage_key = step_key if step_key \
#                       else re.sub(r'\[.*?\]', '[X]', action_str)[:60]

#         if correction and len(correction) > 15:
#             # Corrective rule — prediction was wrong
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = storage_key,
#                 rule          = correction,
#                 positive      = False,
#             ))
#             print(f'  [STAR] Correction stored under [{storage_key}]: {correction[:80]}')
#         elif expected and observation and self._prediction_matched(expected, observation):
#             # Positive rule — prediction matched, stored without LLM call
#             rule = f"{action_type}[X]: {expected[:100]}"
#             self.knowledge_store.add(StepKnowledge(
#                 action_intent = storage_key,
#                 rule          = rule,
#                 positive      = True,
#             ))
#             print(f'  [STAR] Confirmed stored under [{storage_key}]')

#         # ── 6. Store state for next step ──────────────────────────────────────
#         self._prev_expected    = expected
#         self._prev_observation = observation
#         self._prev_key         = step_key   # KEY used as retrieval query at step N+1

#         self.step_n += 1

#     # ------------------------------------------------------------------
#     # Prediction match
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _prediction_matched(expected: str, actual: str) -> bool:
#         stopwords = {'the','a','an','is','was','will','to','of','in','and','or','it','this','that'}
#         exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
#         act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
#         if not exp_words:
#             return False
#         return len(exp_words & act_words) / len(exp_words) > 0.4

#     # ------------------------------------------------------------------
#     # Reflection
#     # ------------------------------------------------------------------

#     def _reflect(self) -> None:
#         print('  [STAR] Reflecting...')

#         error_class = classify_error(
#             self.question, self.scratchpad, self.reflect_llm)
#         print(f'  Error class: {error_class}')

#         # trajectory_k trajectories with full scratchpad for reflection context
#         retrieved = self.trajectory_store.retrieve(
#             question      = self.question,
#             error_class   = error_class,
#             k             = self.trajectory_k,
#             max_failures  = self.trajectory_max_failures,
#             max_successes = self.trajectory_max_successes,
#         )
#         print(f'  [STAR] Retrieved {len(retrieved)} trajectories '
#               f'({sum(1 for r in retrieved if r.success)} successes, '
#               f'{sum(1 for r in retrieved if not r.success)} failures)')

#         # Step knowledge as extra context at reflection time
#         recent_knowledge = self.knowledge_store.retrieve(
#             classify_question_type(self.question), k=2)
#         knowledge_ctx = format_step_knowledge(recent_knowledge)

#         retrieved_context = format_retrieved_trajectories(retrieved)
#         current_block = (
#             "\n=== CURRENT FAILED TRAJECTORY ===\n"
#             f"Question: {self.question}\n"
#             f"Error class: {error_class}\n\n"
#             f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
#         )
#         instruction = (
#             "\nWrite reflection in EXACTLY this format:\n"
#             "FAILED_STEP: <step number>\n"
#             "WHAT_WENT_WRONG: <one sentence>\n"
#             "WHAT_TO_DO_DIFFERENTLY: <exact first action next trial>\n"
#             "GENERALISATION: <one sentence>\n"
#         )
#         reflection_prompt = (
#             RETRIEVAL_REFLECTION_HEADER + retrieved_context +
#             current_block + instruction
#         )
#         if knowledge_ctx:
#             reflection_prompt = knowledge_ctx + "\n" + reflection_prompt

#         reflection = format_step(
#             self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
#         print(f'  Reflection: {reflection[:120]}...')

#         self.trajectory_store.add(TrajectoryRecord(
#             question=self.question, scratchpad=self.scratchpad,
#             reflection=reflection, success=False, error_class=error_class,
#         ))

#         # reflections_str: reflection texts from retrieved trajectories + current
#         # Capped at trajectory_k (how many were retrieved at reflection time)
#         # step_reflection_k is used separately in _build_agent_prompt per step
#         retrieved_refs = [r.reflection for r in retrieved if r.reflection]
#         all_refs = (retrieved_refs + [reflection])[-self.trajectory_k:]
#         self.reflections = all_refs
#         self.reflections_str = (
#             REFLECTION_HEADER + 'Reflections:\n- ' +
#             '\n- '.join(r.strip() for r in all_refs)
#         )
#         print(self.reflections_str)

#     # ------------------------------------------------------------------
#     # Agent prompt
#     # ------------------------------------------------------------------

#     def _build_agent_prompt(self, knowledge_str: str = '') -> str:
#         parts = []

#         # Step knowledge (knowledge_k rules, compact format)
#         if knowledge_str:
#             parts.append(knowledge_str)

#         parts.append(self.react_examples)
#         parts.append('(END OF EXAMPLES)')

#         # Trajectory reflections per step (step_reflection_k, text only)
#         # Retrieved fresh each step by question similarity
#         # Falls back to reflections_str when trajectory store is empty
#         if self.trajectory_store.records:
#             step_retrieved = self.trajectory_store.retrieve(
#                 question      = self.question,
#                 error_class   = 'UNKNOWN',
#                 k             = self.step_reflection_k,
#                 max_failures  = self.step_reflection_k - 1,
#                 max_successes = 1,
#             )
#             step_refs = [r.reflection for r in step_retrieved if r.reflection]
#             if step_refs:
#                 parts.append(
#                     REFLECTION_HEADER + 'Reflections:\n- ' +
#                     '\n- '.join(r.strip()[:200] for r in step_refs)
#                 )
#         elif self.reflections_str:
#             parts.append(self.reflections_str)

#         parts.append(f"Question: {self.question}")
#         parts.append(f"Scratchpad:\n{self.scratchpad}")

#         # Mismatch injection — short, only when prediction was wrong
#         if (self._prev_expected and self._prev_observation
#                 and not self._prediction_matched(self._prev_expected,
#                                                   self._prev_observation)):
#             parts.append(
#                 f"NOTE: Prediction wrong. "
#                 f"Expected: {self._prev_expected[:80]} | "
#                 f"Got: {self._prev_observation[:80]} | "
#                 f"Write CORRECTION rule."
#             )

#         parts.append(STAR_STEP_INSTRUCTION)
#         return '\n\n'.join(parts)

#     # ------------------------------------------------------------------
#     # Standard interface
#     # ------------------------------------------------------------------

#     def is_finished(self) -> bool: return self.finished
#     def is_correct(self)  -> bool: return EM(self.answer, self.key)

#     def is_halted(self) -> bool:
#         return (
#             (self.step_n > self.max_steps) or
#             (len(self.enc.encode(self._build_agent_prompt())) > 3896)
#         ) and not self.finished

#     def __reset_agent(self) -> None:
#         self.step_n            = 1
#         self.finished          = False
#         self.answer            = ''
#         self.scratchpad        = ''
#         self._prev_expected    = ''
#         self._prev_observation = ''
#         self._prev_key         = ''   # KEY from previous step, '' on first step

#     def set_qa(self, question: str, key: str) -> None:
#         self.question = question
#         self.key      = key


# # ---------------------------------------------------------------------------
# # String utilities
# # ---------------------------------------------------------------------------

# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

# def parse_action(string: str):
#     for action_type in ['Finish', 'Search', 'Lookup']:
#         match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
#         if match:
#             return action_type, match.group(1)
#     return None, None

# def format_step(step: str) -> str:
#     return step.strip('\n').strip().replace('\n', '') if step else ''

# def format_reflections(reflections: List[str],
#                        header: str = REFLECTION_HEADER) -> str:
#     if not reflections:
#         return ''
#     return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# def format_last_attempt(question: str, scratchpad: str,
#                         header: str = LAST_TRIAL_HEADER) -> str:
#     return (header + f'Question: {question}\n' +
#             truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() +
#             '\n(END PREVIOUS TRIAL)\n')

# def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600,
#                         tokenizer=gpt2_enc) -> str:
#     lines = scratchpad.split('\n')
#     observations = list(filter(lambda x: x.startswith('Observation'), lines))
#     observations_by_tokens = sorted(observations,
#                                     key=lambda x: len(tokenizer.encode(x)))
#     while (len(gpt2_enc.encode('\n'.join(lines))) > n_tokens
#            and observations_by_tokens):
#         largest = observations_by_tokens.pop(-1)
#         ind     = lines.index(largest)
#         lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
#     return '\n'.join(lines)

# def normalize_answer(s: str) -> str:
#     def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
#     def white_space_fix(text):  return " ".join(text.split())
#     def remove_punc(text):
#         return "".join(ch for ch in text if ch not in set(string.punctuation))
#     def lower(text): return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def EM(answer: str, key: str) -> bool:
#     return normalize_answer(answer) == normalize_answer(key)

"""
STAR — Step-level Knowledge Retrieval on top of Reflexion

Reflection mechanism: identical to Reflexion LAST_ATTEMPT_AND_REFLEXION
  - After failure: reflect using last attempt scratchpad as context
  - Agent sees: last attempt trajectory + reflection at every step next trial
  - Only last reflection kept — no accumulation

STAR addition on top of Reflexion:
  - StepKnowledgeStore: stores atomic rules extracted per step
  - KEY (3-5 word abstract): storage key + next-step retrieval query
  - Step 1: classify_question_type(question) as retrieval query (no LLM call)
  - Step N: KEY from step N-1 as retrieval query
  - Correct prediction → positive rule stored (no LLM call)
  - Wrong prediction → NOTE injected, LLM writes CORRECTION rule
"""

import re
import string
from typing import List, Optional
from enum import Enum
import numpy as np
import tiktoken
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from llm import AnyOpenAILLM
from prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    reflect_prompt,
    REFLECTION_HEADER,
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
)
from fewshots import REFLECTIONS, WEBTHINK_SIMPLE2


class ReflexionStrategy(Enum):
    NONE = 'base'
    STAR = 'star'


# ---------------------------------------------------------------------------
# Question type classifier — no LLM call, step 1 retrieval query
# ---------------------------------------------------------------------------

def classify_question_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ['both', 'same', 'common', 'share', 'also']):
        return 'search-comparison'
    if any(w in q for w in ['how many', 'number of', 'count']):
        return 'search-count'
    if any(w in q for w in ['where', 'located', 'headquartered', 'based']):
        return 'search-location'
    if any(w in q for w in ['when', 'year', 'born', 'died', 'founded', 'date']):
        return 'search-date'
    if any(w in q for w in ['which', 'what team', 'what country', 'what city']):
        return 'search-entity-lookup'
    if any(w in q for w in ['who', 'directed', 'created', 'wrote',
                             'invented', 'played', 'authored']):
        return 'search-person-role'
    if any(w in q for w in ['what type', 'what kind', 'what genre', 'what sport']):
        return 'search-category'
    return 'search-multi-hop'

# def classify_question_type_llm(question: str, llm_fn) -> str:
#     """
#     Classify question into abstract retrieval key using LLM.
#     Single short call — no fewshots needed.
#     """
#     prompt = (
#         f"Classify this question into a 3-5 word abstract search pattern.\n"
#         f"Question: {question}\n\n"
#         f"Reply with ONLY a short hyphenated key, e.g.:\n"
#         f"search-person-role, search-location, search-date, "
#         f"search-comparison, search-multi-hop, search-film-director, "
#         f"search-entity-category\n\n"
#         f"Key:"
#     )
#     raw = llm_fn(prompt).strip().lower()
#     # Clean up — take first word/phrase, strip punctuation
#     key = re.sub(r'[^a-z0-9\-]', '', raw.split('\n')[0].strip())
#     return key if key else 'search-multi-hop'

def classify_question_type_llm(question: str, llm_fn) -> str:
    prompt = (
        f"Classify this question into a 3-5 word abstract search pattern.\n"
        f"Question: {question}\n\n"
        f"Reply with ONLY a short hyphenated key, e.g.:\n"
        f"search-person-role, search-location, search-date, "
        f"search-comparison, search-multi-hop, search-film-director, "
        f"search-entity-category\n\n"
        f"Key:"
    )
    raw = llm_fn(prompt, REFLECTION_SYSTEM_PROMPT).strip().lower()
    key = re.sub(r'[^a-z0-9\-]', '', raw.split('\n')[0].strip())
    return key if key else 'search-multi-hop'
# ---------------------------------------------------------------------------
# Step Knowledge
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
    """Attention-weighted retrieval over step-level knowledge."""

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
# Structured output parser — KEY field
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> dict:
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


def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
    if not knowledge:
        return ""
    lines = ["[STEP RULES]"]
    for sk in knowledge:
        icon = "OK" if sk.positive else "FIX"
        lines.append(f"{icon}: {sk.rule[:120]}")
    lines.append("")
    return '\n'.join(lines)


STAR_STEP_INSTRUCTION = (
    "\nRespond with EXACTLY these labels:\n"
    "THOUGHT: <reasoning>\n"
    "ACTION: <Search[X] or Lookup[X] or Finish[X]>\n"
    "EXPECTED: <brief expected result>\n"
    "KEY: <3-5 word abstract pattern for this step, "
    "e.g. search-person-role, lookup-birth-year, search-film-director>\n"
    "CORRECTION: <if prev EXPECTED was wrong — one generalizable rule. "
    "Omit entirely if first step or prediction was correct.>\n"
)


# ---------------------------------------------------------------------------
# STAR ReactAgent
# ---------------------------------------------------------------------------

class STARReactAgent:
    """
    STAR = Reflexion (LAST_ATTEMPT_AND_REFLEXION) + StepKnowledgeStore.

    Reflection (identical to Reflexion best variant):
      - reflect_prompt uses last attempt scratchpad as context
      - Agent sees: last attempt trajectory + reflection at every step
      - last_scratchpad and last_reflection persist across resets
      - Only last trial's reflection kept — no accumulation

    Step-level addition:
      - Retrieve knowledge_k rules before each action using _prev_key
      - Step 1: classify_question_type(question) as retrieval query
      - Store rule under LLM-generated KEY after each step
      - Inject NOTE when prediction was wrong → LLM writes CORRECTION
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 react_llm: AnyOpenAILLM = None,
                 reflect_llm: AnyOpenAILLM = None,
                 knowledge_store: StepKnowledgeStore = None,
                 knowledge_k: int = 2,
                 use_reflection: bool = True):

        self.question        = question
        self.key             = key
        self.max_steps       = max_steps
        self.llm             = react_llm  or AnyOpenAILLM()
        self.reflect_llm     = reflect_llm or AnyOpenAILLM()
        self.knowledge_store = knowledge_store if knowledge_store is not None \
                               else StepKnowledgeStore()
        self.knowledge_k     = knowledge_k
        self.use_reflection  = use_reflection
        self.react_examples  = WEBTHINK_SIMPLE2
        self.enc             = tiktoken.encoding_for_model("text-davinci-003")
        self.docstore        = DocstoreExplorer(Wikipedia())

        # Persist across resets — set in _reflect(), never reset
        self.last_reflection = ''
        self.last_scratchpad = ''

        self.__reset_agent()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, reset: bool = True,
            reflect_strategy: ReflexionStrategy = ReflexionStrategy.STAR) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            if self.use_reflection:
                self._reflect()   # uses self.scratchpad before reset
        if reset:
            self.__reset_agent()
        while not self.is_halted() and not self.is_finished():
            self.step()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        # 1. Retrieve knowledge_k rules
        # retrieval_query = self._prev_key if self._prev_key \
        #                   else classify_question_type(self.question)
        if self._prev_key:
            retrieval_query = self._prev_key
        elif self.step_n == 1:
            retrieval_query = classify_question_type_llm(
                self.question, self.reflect_llm)
            print(f'  [STAR] LLM question type: "{retrieval_query}"')
        else:
            retrieval_query = 'search-multi-hop'   # fallback if key missing
        retrieved_knowledge = self.knowledge_store.retrieve(
            retrieval_query, k=self.knowledge_k)
        knowledge_str = format_step_knowledge(retrieved_knowledge)
        if retrieved_knowledge:
            print(f'  [STAR] {len(retrieved_knowledge)} rules for "{retrieval_query}"')

        # 2. One LLM call
        prompt = self._build_agent_prompt(knowledge_str)
        raw    = self.llm(prompt, REACT_SYSTEM_PROMPT) or ''
        parsed = parse_structured_response(raw)

        thought    = parsed['thought']
        action_str = parsed['action']
        expected   = parsed['expected']
        step_key   = parsed['key']
        correction = parsed['correction']

        # Fallback
        if not thought and not action_str:
            print('  [STAR] Parse failed — falling back to ReAct format')
            for line in raw.split('\n'):
                line = line.strip()
                if not thought and re.match(r'Thought\s*\d*\s*:', line, re.IGNORECASE):
                    thought = line.split(':', 1)[-1].strip()
                if not action_str and any(a in line for a in ['Search[', 'Lookup[', 'Finish[']):
                    action_str = line.strip()

        # 3. Update scratchpad
        self.scratchpad += f'\nThought {self.step_n}: {thought}'
        self.scratchpad += f'\nAction {self.step_n}: {action_str}'
        print(f'Thought {self.step_n}: {thought[:80]}')
        print(f'ACTION=> {action_str}')
        print(f'EXPECTED: {expected} | KEY: {step_key}')
        print(f'CORRECTION: {correction}')

        action_type, argument = '', ''
        try:
            action_type, argument = parse_action(action_str)
        except Exception:
            print("Invalid Action")

        self.scratchpad += f'\nObservation {self.step_n}: '

        # 4. Execute action
        if action_type == 'Finish':
            self.answer = argument
            observation = 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.scratchpad += observation
            self.finished   = True
            print(observation)
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                observation = format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                observation = 'Could not find that page, please try again.'
        elif action_type == 'Lookup':
            try:
                observation = format_step(self.docstore.lookup(argument))
            except ValueError:
                observation = ('The last page Searched was not found, '
                               'so you cannot Lookup a keyword in it. '
                               'Please try one of the similar pages given.')
        else:
            observation = ('Invalid Action. Valid Actions are '
                           'Lookup[<topic>] Search[<topic>] and Finish[<answer>].')

        self.scratchpad += observation
        print(f'Observation {self.step_n}: {observation[:100]}')

        # 5. Store knowledge under KEY — no LLM call
        storage_key = step_key if step_key \
                      else re.sub(r'\[.*?\]', '[X]', action_str)[:60]

        if correction and len(correction) > 15:
            self.knowledge_store.add(StepKnowledge(
                action_intent = storage_key,
                rule          = correction,
                positive      = False,
            ))
            print(f'  [STAR] FIX [{storage_key}]: {correction[:80]}')
        # elif expected and observation and self._prediction_matched(expected, observation):
        #     rule = f"{action_type}[X]: {expected[:100]}"
        #     self.knowledge_store.add(StepKnowledge(
        #         action_intent = storage_key,
        #         rule          = rule,
        #         positive      = True,
        #     ))
        #     print(f'  [STAR] OK [{storage_key}]')

        # 6. Store state for next step
        self._prev_expected    = expected
        self._prev_observation = observation
        self._prev_key         = step_key

        self.step_n += 1

    # ------------------------------------------------------------------
    # Prediction match
    # ------------------------------------------------------------------

    @staticmethod
    def _prediction_matched(expected: str, actual: str) -> bool:
        stopwords = {'the','a','an','is','was','will','to','of','in','and','or','it','this','that'}
        exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
        act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
        if not exp_words:
            return False
        return len(exp_words & act_words) / len(exp_words) > 0.4

    # ------------------------------------------------------------------
    # Reflection — identical to Reflexion LAST_ATTEMPT_AND_REFLEXION
    # ------------------------------------------------------------------

    def _reflect(self) -> None:
        print('  [STAR] Reflecting...')

        # Save scratchpad before reset so prompt can show last attempt trajectory
        self.last_scratchpad = self.scratchpad

        # Generate reflection using last attempt scratchpad as context
        # — identical to Reflexion's reflect_prompt usage
        prompt = reflect_prompt.format(
            examples   = REFLECTIONS,
            question   = self.question,
            scratchpad = truncate_scratchpad(self.scratchpad, tokenizer=self.enc),
        )
        self.last_reflection = format_step(
            self.reflect_llm(prompt, REFLECTION_SYSTEM_PROMPT))
        print(f'  Reflection: {self.last_reflection[:120]}')

    # ------------------------------------------------------------------
    # Agent prompt
    # ------------------------------------------------------------------

    def _build_agent_prompt(self, knowledge_str: str = '') -> str:
        parts = []

        # Step knowledge (STAR addition)
        if knowledge_str:
            parts.append(knowledge_str)

        parts.append(self.react_examples)
        parts.append('(END OF EXAMPLES)')

        # Last attempt trajectory + reflection — identical to Reflexion
        # LAST_ATTEMPT_AND_REFLEXION format: scratchpad then reflection
        if self.last_reflection:
            last_attempt_block = (
                LAST_TRIAL_HEADER +
                f'Question: {self.question}\n' +
                truncate_scratchpad(self.last_scratchpad, tokenizer=self.enc).strip() +
                '\n(END PREVIOUS TRIAL)\n' +
                REFLECTION_AFTER_LAST_TRIAL_HEADER +
                f'Reflection:\n- {self.last_reflection.strip()}'
            )
            parts.append(last_attempt_block)

        parts.append(f"Question: {self.question}")
        parts.append(f"Scratchpad:\n{self.scratchpad}")

        # Mismatch NOTE
        if (self._prev_expected and self._prev_observation
                and not self._prediction_matched(self._prev_expected,
                                                  self._prev_observation)):
            parts.append(
                f"NOTE: Prediction wrong. "
                f"Expected: {self._prev_expected[:80]} | "
                f"Got: {self._prev_observation[:80]} | "
                f"Write CORRECTION rule."
            )

        parts.append(STAR_STEP_INSTRUCTION)
        return '\n\n'.join(parts)

    # ------------------------------------------------------------------
    # Standard interface
    # ------------------------------------------------------------------

    def is_finished(self) -> bool: return self.finished
    def is_correct(self)  -> bool: return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps) or
            (len(self.enc.encode(self._build_agent_prompt())) > 3896)
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n            = 1
        self.finished          = False
        self.answer            = ''
        self.scratchpad        = ''
        self._prev_expected    = ''
        self._prev_observation = ''
        self._prev_key         = ''
        # last_reflection and last_scratchpad persist — set in _reflect()

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key      = key


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string: str):
    for action_type in ['Finish', 'Search', 'Lookup']:
        match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
        if match:
            return action_type, match.group(1)
    return None, None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') if step else ''

def truncate_scratchpad(scratchpad: str, n_tokens: int = 800,
                        tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = list(filter(lambda x: x.startswith('Observation'), lines))
    observations_by_tokens = sorted(observations,
                                    key=lambda x: len(tokenizer.encode(x)))
    while (len(gpt2_enc.encode('\n'.join(lines))) > n_tokens
           and observations_by_tokens):
        largest = observations_by_tokens.pop(-1)
        ind     = lines.index(largest)
        lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):  return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer: str, key: str) -> bool:
    return normalize_answer(answer) == normalize_answer(key)