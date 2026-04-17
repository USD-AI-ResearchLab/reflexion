"""
STAR trial runner for ALFWorld.

STAR = Reflexion (LAST_ATTEMPT_AND_REFLEXION) + StepKnowledgeStore.

Follows the same loop structure as alfworld_trial.py but:
  - Per step: retrieve knowledge_k rules → inject as [STEP RULES] block
  - LLM outputs THOUGHT/ACTION/EXPECTED/KEY/CORRECTION per step
  - CORRECTION stored under KEY when prediction was wrong
  - last_scratchpad + last_reflection persist across trials (per env)
  - Step 1 retrieval: classify_task_type(task_desc) — no LLM call
  - Step N retrieval: KEY from step N-1
"""

import os
import re
import sys
import yaml
import importlib
import alfworld
import alfworld.agents.environment

from utils import Model, get_chat, get_completion
from alfword_agents import ALFWORLD_TASK_TYPES
from star_alfworld_agents import (
    StepKnowledge,
    StepKnowledgeStore,
    classify_task_type,
    format_step_knowledge,
    parse_star_response,
    prediction_matched_alfworld,
    STAR_STEP_INSTRUCTION,
)
from star_alfworld_fewshots import STAR_FEWSHOTS

from typing import List, Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# LLM helpers (same as alfworld_trial.py)
# ---------------------------------------------------------------------------

def llm(prompt: str, model: Model, stop: List[str] = ["\n"]) -> str:
    try:
        cur_try = 0
        while cur_try < 6:
            if model == "text-davinci-003":
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            else:
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            if len((text or "").strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(e)
        sys.exit(1)


def llm_no_stop(prompt: str, model: Model) -> str:
    try:
        if model == "text-davinci-003":
            return get_completion(prompt=prompt, temperature=0.0, stop_strs=[])
        else:
            return get_chat(prompt=prompt, model=model, temperature=0.0, stop_strs=[])
    except Exception as e:
        print(e)
        return ""


def process_ob(ob: str) -> str:
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


# ---------------------------------------------------------------------------
# STAR reflection headers (mirrors hotpotqa LAST_TRIAL / REFLECTION_AFTER headers)
# ---------------------------------------------------------------------------

LAST_TRIAL_HEADER = (
    "\n---------------\n"
    "PREVIOUS TRIAL (for reference):\n"
)

REFLECTION_AFTER_LAST_TRIAL_HEADER = (
    "\n---------------\n"
    "REFLECTION on previous trial:\n"
)

REFLECT_PROMPT_TEMPLATE = """\
You failed the following household task. Study your trajectory and write a concise reflection.

Task: {task_desc}

Your trajectory:
{scratchpad}

Write a reflection explaining:
1. Where you went wrong
2. What you should do differently next time
Reflection:"""


# ---------------------------------------------------------------------------
# Build STAR step prompt
# ---------------------------------------------------------------------------

def build_star_step_prompt(
    base_prompt: str,
    ob: str,
    scratchpad: str,
    knowledge_str: str,
    last_scratchpad: str,
    last_reflection: str,
    task_desc: str,
    prev_expected: str,
    prev_observation: str,
) -> str:
    parts = []

    # 1. Step knowledge (STAR addition)
    if knowledge_str:
        parts.append(knowledge_str)

    # 2. Base prompt (fewshot examples from alfworld_3prompts.json)
    parts.append(base_prompt)

    # 3. Last attempt + reflection (Reflexion LAST_ATTEMPT_AND_REFLEXION)
    if last_reflection:
        last_attempt_block = (
            LAST_TRIAL_HEADER +
            f'Task: {task_desc}\n' +
            _truncate_scratchpad(last_scratchpad, max_chars=2000) +
            '\n(END PREVIOUS TRIAL)\n' +
            REFLECTION_AFTER_LAST_TRIAL_HEADER +
            f'Reflection:\n- {last_reflection.strip()}'
        )
        parts.append(last_attempt_block)

    # 4. Current state
    parts.append(f"\nHere is the task.\n{ob}")
    if scratchpad:
        parts.append(scratchpad)

    # 5. Mismatch NOTE
    if prev_expected and prev_observation:
        if not prediction_matched_alfworld(prev_expected, prev_observation):
            parts.append(
                f"NOTE: Prediction wrong. "
                f"Expected: {prev_expected[:100]} | "
                f"Got: {prev_observation[:100]} | "
                f"Write CORRECTION rule."
            )

    parts.append(STAR_STEP_INSTRUCTION)
    return '\n'.join(parts)


def _truncate_scratchpad(scratchpad: str, max_chars: int = 2000) -> str:
    if len(scratchpad) <= max_chars:
        return scratchpad
    half = max_chars // 2
    return scratchpad[:half] + '\n...[truncated]...\n' + scratchpad[-half:]


# ---------------------------------------------------------------------------
# STAR ALFWorld run — one episode
# ---------------------------------------------------------------------------

def alfworld_run_star(
    env,
    base_prompt: str,
    knowledge_store: StepKnowledgeStore,
    task_desc: str,
    last_scratchpad: str,
    last_reflection: str,
    model: Model,
    to_print: bool = True,
    ob: str = '',
    knowledge_k: int = 2,
    max_steps: int = 49,
) -> Tuple[str, bool, str, str]:
    """
    Run one ALFWorld episode with STAR step-level knowledge injection.

    Returns: (scratchpad_str, is_success, new_last_scratchpad, new_last_reflection)
    new_last_scratchpad / new_last_reflection are only set on failure (for next trial).
    On success they are '' (no reflection needed).
    """
    scratchpad   = ''
    prev_key     = ''
    prev_expected  = ''
    prev_observation = ''

    if to_print:
        print(ob)
        sys.stdout.flush()

    for step_n in range(1, max_steps + 1):
        print(f"STEP: {step_n}")

        # Retrieval query: task type on step 1, prev KEY on subsequent steps
        retrieval_query = prev_key if prev_key else classify_task_type(task_desc)
        retrieved = knowledge_store.retrieve(retrieval_query, k=knowledge_k)
        knowledge_str = format_step_knowledge(retrieved)
        if retrieved:
            print(f'  [STAR] {len(retrieved)} rules for "{retrieval_query}"')

        # Build full prompt
        prompt = build_star_step_prompt(
            base_prompt     = base_prompt,
            ob              = ob,
            scratchpad      = scratchpad,
            knowledge_str   = knowledge_str,
            last_scratchpad = last_scratchpad,
            last_reflection = last_reflection,
            task_desc       = task_desc,
            prev_expected   = prev_expected,
            prev_observation= prev_observation,
        )

        # LLM call — no stop token so we get all STAR fields
        raw = llm_no_stop(prompt, model) or ''
        parsed = parse_star_response(raw)

        thought    = parsed['thought']
        action     = parsed['action']
        expected   = parsed['expected']
        step_key   = parsed['key']
        correction = parsed['correction']

        # Fallback: if structured parsing failed, try to extract just an action
        if not action:
            for line in raw.split('\n'):
                line = line.strip()
                if line and not line.upper().startswith(('THOUGHT', 'EXPECTED', 'KEY', 'CORRECTION')):
                    action = line.lstrip('> ').strip()
                    if action:
                        break

        # Update scratchpad (Thought + Action only, identical to ReAct)
        scratchpad += f'\n> think: {thought}' if thought else ''
        print(f'Thought {step_n}: {thought[:80]}')
        print(f'ACTION=> {action}')
        print(f'EXPECTED: {expected} | KEY: {step_key}')
        print(f'CORRECTION: {correction}')

        # Execute action in environment
        observation, reward, done, info = env.step([action])
        observation = process_ob(observation[0])
        is_won      = info['won'][0]
        done        = done[0]

        if action.startswith('think:'):
            observation = 'OK.'

        scratchpad += f'\n{action}\n{observation}'

        if to_print:
            print(f'> {action}')
            print(observation)
            print('OK')
            sys.stdout.flush()

        # Store knowledge under KEY — only when the previous prediction was wrong
        storage_key = step_key if step_key else re.sub(r'\s+', '-', action.split()[0])[:40]

        prev_mismatch = (
            prev_expected and prev_observation
            and not prediction_matched_alfworld(prev_expected, prev_observation)
        )
        if correction and len(correction) > 15 and prev_mismatch:
            knowledge_store.add(StepKnowledge(
                action_intent=storage_key,
                rule=correction,
                positive=False,
            ))
            print(f'  [STAR] FIX [{storage_key}]: {correction[:80]}')

        # Update state for next step
        prev_expected   = expected
        prev_observation = observation
        prev_key        = step_key

        if is_won:
            return scratchpad, True, '', ''

        # Check exhausted (repeated action loop) — same heuristic as EnvironmentHistory
        if _is_exhausted(scratchpad):
            break

    # Episode failed — generate reflection
    reflection = _generate_reflection(task_desc, scratchpad, model)
    return scratchpad, False, scratchpad, reflection


def _is_exhausted(scratchpad: str, window: int = 6) -> bool:
    """Detect repeated action loop: same action 3+ times in last window lines."""
    lines = [l.strip() for l in scratchpad.strip().split('\n') if l.strip()]
    skip_prefixes = ('OK', 'You', 'Nothing', 'I')
    recent_actions = [l for l in lines[-window:]
                      if not any(l.startswith(p) for p in skip_prefixes)]
    if len(recent_actions) >= 3:
        if len(set(recent_actions[-3:])) == 1:
            return True
    return False


def _generate_reflection(task_desc: str, scratchpad: str, model: Model) -> str:
    prompt = REFLECT_PROMPT_TEMPLATE.format(
        task_desc=task_desc,
        scratchpad=_truncate_scratchpad(scratchpad, max_chars=2000),
    )
    return (llm_no_stop(prompt, model) or '').strip()


# ---------------------------------------------------------------------------
# run_trial_star — drop-in replacement for run_trial with strategy='star'
# ---------------------------------------------------------------------------

def run_trial_star(
    trial_log_path: str,
    world_log_path: str,
    trial_idx: int,
    env_configs: List[Dict[str, Any]],
    model: Model,
    knowledge_store: StepKnowledgeStore,
    knowledge_k: int = 2,
    use_reflection: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run one STAR trial across all environments.

    env_config fields used/set by STAR:
      - is_success: bool
      - skip: bool
      - steps: int
      - star_last_scratchpad: str (persists across trials)
      - star_last_reflection: str (persists across trials)

    knowledge_store is shared across all environments and all trials.
    """
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    num_successes = 0
    num_additional_successes = 0
    num_envs = len(env_configs)
    env_steps: List[int] = []

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(f"using {name}")

        if env_config.get("is_success", False):
            num_successes += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            env_steps.append(0)
            continue

        task_desc = (ob or '').strip().split('\n')[-1] if (ob or '').strip() else name
        task_type = next(
            (k for k in ALFWORLD_TASK_TYPES if name.startswith(k)), 'pick_and_place')
        v = ALFWORLD_TASK_TYPES[task_type]

        fewshot_1, fewshot_0 = STAR_FEWSHOTS[v]
        base_prompt = (
            'Interact with a household to solve a task. Here are two examples.\n'
            + fewshot_1 + '\n\n' + fewshot_0
        )

        # Per-env persistent state (Reflexion LAST_ATTEMPT_AND_REFLEXION)
        last_scratchpad = env_config.get('star_last_scratchpad', '') if use_reflection else ''
        last_reflection = env_config.get('star_last_reflection', '') if use_reflection else ''

        scratchpad, is_success, new_last_scratchpad, new_last_reflection = alfworld_run_star(
            env            = env,
            base_prompt    = base_prompt,
            knowledge_store= knowledge_store,
            task_desc      = task_desc,
            last_scratchpad= last_scratchpad,
            last_reflection= last_reflection,
            model          = model,
            ob             = ob,
            knowledge_k    = knowledge_k,
        )

        # Count steps from scratchpad (each action line)
        steps_taken = sum(1 for l in scratchpad.split('\n')
                          if l.strip() and not l.startswith(('>', 'You', 'Nothing', 'OK')))
        env_steps.append(steps_taken)
        env_configs[z]['steps'] = steps_taken

        if is_success:
            status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'
            # Persist last attempt + reflection for next trial
            if use_reflection and new_last_scratchpad:
                env_configs[z]['star_last_scratchpad'] = new_last_scratchpad
                env_configs[z]['star_last_reflection'] = new_last_reflection

        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')
        with open(trial_log_path, 'a') as wf:
            wf.write(
                f'\n#####\n\nEnvironment #{z}:\n'
                f'Task type: {task_type}\n'
                f'Steps: {steps_taken}\n'
                f'{scratchpad}\n\n'
                f'STATUS: {"OK" if is_success else "FAIL"}\n\n#####\n'
            )

    print("CLOSE")
    env.close()

    active_steps = [s for s in env_steps if s > 0]
    avg_steps = round(sum(active_steps) / len(active_steps), 2) if active_steps else 0.0

    log_str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
AVG_STEPS: {avg_steps}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
