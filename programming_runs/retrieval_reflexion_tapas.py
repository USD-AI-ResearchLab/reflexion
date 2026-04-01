"""
Retrieval-augmented Reflexion + TAPAS policy learning.
Changes from original retrieval_reflexion.py:
  1. Added policy_store=None param to run_retrieval_reflexion
  2. Policy injection into reflection prompt (Step 3)
  3. Policy update after each failure (Step 4b)
"""

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from generators.model import ModelBase, Message
from programming_agents_tapas import (
    TrajectoryRecord,
    TrajectoryStore,
    format_retrieved_trajectories,
    build_retrieval_reflection_prompt,
    PROGRAMMING_ERROR_TAXONOMY,
)

import sys
sys.path.append('..')
from policy_store import PolicyStore   # ← new

from typing import List, Optional


CLASSIFY_ERROR_SYSTEM_PROMPT = (
    "You are an error classification agent. You will be given a failed Python "
    "implementation and its test feedback. Classify the failure into exactly ONE "
    "error type from the provided list. Respond with only the error type label, "
    "nothing else. No explanation, no punctuation, just the label."
)

RETRIEVAL_REFLECTION_SYSTEM_PROMPT = (
    "You are a Python programming assistant analyzing why a function implementation failed. "
    "You will be given similar past trajectories as context and the current failed implementation. "
    "Write a concise, actionable reflection explaining what went wrong and what to do differently. "
    "Only provide the reflection text, not the implementation."
)


def classify_error_with_model(func_sig, implementation, feedback, model):
    prompt = (
        f"Classify the following failed Python implementation into exactly ONE of these error types:\n"
        f"{', '.join(PROGRAMMING_ERROR_TAXONOMY)}\n\n"
        f"Function signature:\n{func_sig[:300]}\n\n"
        f"Failed implementation:\n{implementation[:500]}\n\n"
        f"Test feedback:\n{feedback[:300]}\n\n"
        "Reply with only the error type label, nothing else."
    )
    messages = [
        Message(role="system", content=CLASSIFY_ERROR_SYSTEM_PROMPT),
        Message(role="user",   content=prompt),
    ]
    raw = (model.generate_chat(messages=messages) or "").strip().upper()
    for label in PROGRAMMING_ERROR_TAXONOMY:
        if label in raw: return label
    return "UNKNOWN"


def generate_retrieval_reflection(func_sig, implementation, feedback,
                                   error_class, retrieved, model,
                                   policy_str: str = ""):   # ← new param
    user_content = build_retrieval_reflection_prompt(
        func_sig=func_sig, implementation=implementation,
        feedback=feedback, error_class=error_class,
        retrieved=retrieved, policy_str=policy_str,   # ← pass policy
    )
    messages = [
        Message(role="system", content=RETRIEVAL_REFLECTION_SYSTEM_PROMPT),
        Message(role="user",   content=user_content),
    ]
    return (model.generate_chat(messages=messages) or "").strip()


def run_retrieval_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    trajectory_store: Optional[TrajectoryStore] = None,
    policy_store: Optional[PolicyStore] = None,   # ← new for TAPAS
) -> None:

    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    if trajectory_store is None:
        trajectory_store = TrajectoryStore()

    is_tapas    = policy_store is not None
    num_items   = len(dataset)
    num_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        cur_pass        = 0
        is_solved       = False
        reflections     = []
        implementations = []
        test_feedback   = []
        cur_func_impl   = ""
        func_sig        = item["prompt"]
        skipped         = False

        while cur_pass < pass_at_k and not is_solved:
            tests_i = item['visible_tests'] if is_leetcode else gen.internal_tests(func_sig, model, 1)

            cur_func_impl = gen.func_impl(func_sig, model, "simple")
            if not cur_func_impl:
                print(f"Warning: empty implementation for problem {i}, skipping")
                item["solution"] = ""; item["is_solved"] = False
                write_jsonl(log_path, [item], append=True)
                skipped = True; break

            assert isinstance(cur_func_impl, str)
            implementations.append(cur_func_impl)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            if is_passing:
                is_passing = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                if is_passing:
                    is_solved = True; num_success += 1
                    trajectory_store.add(TrajectoryRecord(
                        func_sig=func_sig, implementation=cur_func_impl,
                        feedback=feedback, reflection='', success=True, error_class='SUCCESS',
                    ))
                break

            cur_iter     = 1
            cur_feedback = feedback

            while cur_iter < max_iters:
                # Step 1 — classify error
                error_class = classify_error_with_model(func_sig, cur_func_impl, cur_feedback, model)
                print_v(f"  Error class: {error_class}")

                # Step 2 — retrieve trajectories
                retrieved = trajectory_store.retrieve(
                    func_sig=func_sig, error_class=error_class, k=3,
                    max_failures=2, max_successes=1,
                )
                print_v(f"  Retrieved {len(retrieved)} trajectories")

                # Step 3 — get policy (TAPAS only) ← new
                policy_str = ""
                if is_tapas:
                    policy = policy_store.get(error_class)
                    policy_str = policy.to_prompt_str()
                    if policy_str:
                        print_v(f"  Policy v{policy.version} injected for {error_class}")

                # Step 4 — generate reflection (with policy for TAPAS)
                reflection = generate_retrieval_reflection(
                    func_sig=func_sig, implementation=cur_func_impl,
                    feedback=cur_feedback, error_class=error_class,
                    retrieved=retrieved, model=model,
                    policy_str=policy_str,   # ← new
                )
                reflections.append(reflection)
                print_v(f"  Reflection: {reflection[:120]}...")

                # Step 4b — update policy from failure + reflection (TAPAS only) ← new
                if is_tapas:
                    llm_fn = lambda p: model.generate_chat([Message(role="user", content=p)])
                    policy_store.update(
                        key=error_class,
                        trajectory=(f"Function:\n{func_sig[:400]}\n\n"
                                    f"Implementation:\n{cur_func_impl[:400]}\n\n"
                                    f"Feedback:\n{cur_feedback[:300]}"),
                        reflection=reflection,
                        llm_fn=llm_fn,
                    )

                # Step 5 — store failed trajectory
                trajectory_store.add(TrajectoryRecord(
                    func_sig=func_sig, implementation=cur_func_impl,
                    feedback=cur_feedback, reflection=reflection,
                    success=False, error_class=error_class,
                ))

                # Step 6 — generate improved implementation
                cur_func_impl = gen.func_impl(
                    func_sig=func_sig, model=model, strategy="reflexion",
                    prev_func_impl=cur_func_impl, feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True; num_success += 1
                        trajectory_store.add(TrajectoryRecord(
                            func_sig=func_sig, implementation=cur_func_impl,
                            feedback=cur_feedback, reflection='',
                            success=True, error_class='SUCCESS',
                        ))
                    break
                cur_iter += 1
            cur_pass += 1

        if not skipped:
            item["is_solved"]       = is_solved
            item["reflections"]     = reflections
            item["implementations"] = implementations
            item["test_feedback"]   = test_feedback
            item["solution"]        = cur_func_impl
            write_jsonl(log_path, [item], append=True)

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')