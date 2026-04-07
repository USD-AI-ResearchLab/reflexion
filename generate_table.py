# """
# Summary Table Generator
# ========================
# Reads the same CONFIG as generate_plots.py and produces a CSV summary table:
#   - For all standard methods: final trial SuccessRate
#   - For ExpeL: eval CSV result (not gather)

# Output: ./plots/results_table.csv
# """

# import os
# import glob
# import pandas as pd

# DATA_ROOT = os.path.expanduser('~/Downloads/reflexion-res')

# # ── Same CONFIG as generate_plots.py ────────────────────────────────────────

# HOTPOT = {
#     'ReAct':          f'{DATA_ROOT}/hotpot/react/',
#     'CoT+GT':         f'{DATA_ROOT}/hotpot/cot/',
#     'Reflexion':      f'{DATA_ROOT}/hotpot/reflexion/',
#     'RAR (Ours)':     f'{DATA_ROOT}/hotpot/retrieval/',
#     'ExpeL (Gather)': f'{DATA_ROOT}/hotpot/expel/100_questions_gather_metrics.csv',
#     'ExpeL (Eval)':   f'{DATA_ROOT}/hotpot/expel/100_questions_eval_metrics.csv',
# }

# ALFWORLD = {
#     'ReAct':          f'{DATA_ROOT}/alf/react/',
#     'Reflexion':      f'{DATA_ROOT}/alf/reflexion/',
#     'RAR (Ours)':     f'{DATA_ROOT}/alf/retrieval/',
#     'ExpeL (Gather)': f'{DATA_ROOT}/alf/expel/134_envs_gather_metrics.csv',
#     'ExpeL (Eval)':   f'{DATA_ROOT}/alf/expel/134_envs_eval_metrics.csv',
# }

# HUMANEVAL = {
#     'Simple':         f'{DATA_ROOT}/prog/simple/simple_humaneval_hard/',
#     'CoT+GT':         f'{DATA_ROOT}/prog/cot_gt/cot_gt_humaneval_hard/',
#     'Reflexion':      f'{DATA_ROOT}/prog/reflexion/reflexion_humaneval_hard/',
#     'RAR (Ours)':     f'{DATA_ROOT}/prog/retrieval/retrieval_humaneval_hard',
#     'ExpeL (Gather)': f'{DATA_ROOT}/prog/expel/50_problems_metrics_gather_metrics.csv',
#     'ExpeL (Eval)':   f'{DATA_ROOT}/prog/expel/50_problems_metrics_eval_metrics.csv',
# }

# OUTPUT_DIR   = './plots'
# OUTPUT_CSV   = f'{OUTPUT_DIR}/results_table.csv'

# # All methods in display order for the table
# ALL_METHODS = [
#     'Simple',
#     'CoT+GT',
#     'ReAct',
#     'Reflexion',
#     'ExpeL',
#     'RAR (Ours)',
# ]

# # ── Helpers ──────────────────────────────────────────────────────────────────

# def resolve_path(path):
#     """Resolve folder → single CSV, or return file path if it exists."""
#     if os.path.isdir(path):
#         csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
#         if not csvs:
#             return None
#         if len(csvs) > 1:
#             print(f"  WARNING: multiple CSVs in '{path}', using: {csvs[0]}")
#         return csvs[0]
#     if os.path.exists(path):
#         return path
#     return None


# def get_final_success(csv_path):
#     """Return last row SuccessRate from a CSV."""
#     try:
#         df = pd.read_csv(csv_path)
#         return float(df['SuccessRate'].iloc[-1])
#     except Exception as e:
#         print(f"  ERROR reading {csv_path}: {e}")
#         return None


# def extract_score(config, method_key):
#     """
#     Extract final success rate for a method.
#     For ExpeL: use Eval CSV (single row result).
#     For all others: use last row of their CSV (final trial).
#     """
#     if method_key == 'ExpeL':
#         # Find the Eval entry
#         eval_key = 'ExpeL (Eval)'
#         if eval_key not in config:
#             return None
#         path = resolve_path(config[eval_key])
#         if path is None:
#             return None
#         return get_final_success(path)
#     else:
#         if method_key not in config:
#             return None
#         path = resolve_path(config[method_key])
#         if path is None:
#             return None
#         return get_final_success(path)


# # ── Build table ───────────────────────────────────────────────────────────────

# def build_table():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     task_configs = {
#         'HotPotQA':         HOTPOT,
#         'ALFWorld':         ALFWORLD,
#         'HumanEval Hard 50': HUMANEVAL,
#     }

#     rows = []
#     for method in ALL_METHODS:
#         row = {'Method': method}
#         for task_name, config in task_configs.items():
#             score = extract_score(config, method)
#             if score is not None:
#                 row[task_name] = f'{score * 100:.1f}%'
#             else:
#                 row[task_name] = '—'
#         rows.append(row)

#     df = pd.DataFrame(rows, columns=['Method', 'HotPotQA',
#                                       'ALFWorld', 'HumanEval Hard 50'])
#     df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\nResults table saved to: {OUTPUT_CSV}")
#     print()
#     print(df.to_string(index=False))
#     return df


# if __name__ == '__main__':
#     build_table()


# """
# Table Generator
# ================
# Produces 4 CSV files:
#   1. hotpot_table.csv     — HotPotQA per-trial success + final metrics
#   2. alfworld_table.csv   — ALFWorld per-trial success + final metrics
#   3. humaneval_table.csv  — HumanEval per-iteration success + final metrics
#   4. combined_table.csv   — Final success rate only, all tasks side by side

# ExpeL rule: use Eval CSV for score, not Gather.
# Standard methods: final row = final trial.
# """

# import os
# import glob
# import pandas as pd

# DATA_ROOT = os.path.expanduser('~/Downloads/reflexion-res')
# OUTPUT_DIR = './plots'

# # ── CONFIG (same as generate_plots.py) ───────────────────────────────────────

# HOTPOT = {
#     'ReAct':          f'{DATA_ROOT}/hotpot/react/',
#     'CoT+GT':         f'{DATA_ROOT}/hotpot/cot/',
#     'Reflexion':      f'{DATA_ROOT}/hotpot/reflexion/',
#     'ExpeL':          f'{DATA_ROOT}/hotpot/expel/100_questions_eval_metrics.csv',
#     'RAR (Ours)':     f'{DATA_ROOT}/hotpot/retrieval/',
# }

# ALFWORLD = {
#     'ReAct':          f'{DATA_ROOT}/alf/react/',
#     'Reflexion':      f'{DATA_ROOT}/alf/reflexion/',
#     'ExpeL':          f'{DATA_ROOT}/alf/expel/134_envs_eval_metrics.csv',
#     'RAR (Ours)':     f'{DATA_ROOT}/alf/retrieval/',
# }

# HUMANEVAL = {
#     'Simple':         f'{DATA_ROOT}/prog/simple/simple_humaneval_hard/',
#     'CoT+GT':         f'{DATA_ROOT}/prog/cot_gt/cot_gt_humaneval_hard/',
#     'Reflexion':      f'{DATA_ROOT}/prog/reflexion/reflexion_humaneval_hard/',
#     'ExpeL':          f'{DATA_ROOT}/prog/expel/50_problems_metrics_eval_metrics.csv',
#     'RAR (Ours)':     f'{DATA_ROOT}/prog/retrieval/retrieval_humaneval_hard',
# }

# # Which trial indices to show in the per-task tables (0-indexed rows in CSV)
# # These correspond to trial 0, midpoint, final
# HOTPOT_TRIALS   = [0, 2, 4]   # trials 1, 3, 5
# ALFWORLD_TRIALS = [0, 4, 9]   # trials 1, 5, 10
# HUMANEVAL_ITERS = [0, 4, 9]   # iterations 1, 5, 10


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def resolve_path(path):
#     if os.path.isdir(path):
#         csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
#         if not csvs:
#             print(f"  WARNING: no CSV in '{path}'")
#             return None
#         if len(csvs) > 1:
#             print(f"  WARNING: multiple CSVs in '{path}', using: {csvs[0]}")
#         return csvs[0]
#     if os.path.exists(path):
#         return path
#     print(f"  WARNING: not found: '{path}'")
#     return None


# def load_df(path):
#     p = resolve_path(path)
#     if p is None:
#         return None
#     try:
#         df = pd.read_csv(p)
#         print(f"  Loaded: {p}  ({len(df)} rows)")
#         return df
#     except Exception as e:
#         print(f"  ERROR: {e}")
#         return None


# def fmt(val, decimals=4):
#     """Format float to fixed decimals, or '—' if None."""
#     if val is None:
#         return '—'
#     return f'{val:.{decimals}f}'


# # ── Per-task table builder ────────────────────────────────────────────────────

# def build_task_table(config, trial_indices, task_name):
#     """
#     Returns a DataFrame with columns:
#       Strategy | Trial_X | Trial_Y | Trial_Z | FinalFail | FinalHalt | AvgSteps
#     One row per method.
#     """
#     rows = []
#     for method, path in config.items():
#         df = load_df(path)
#         if df is None:
#             row = {'Strategy': method}
#             for i in trial_indices:
#                 row[f'Trial_{i}'] = '—'
#             row['FinalFail']  = '—'
#             row['FinalHalt']  = '—'
#             row['AvgSteps']   = '—'
#             rows.append(row)
#             continue

#         # Clamp indices to available rows
#         max_idx = len(df) - 1
#         row = {'Strategy': method}

#         for i in trial_indices:
#             idx = min(i, max_idx)
#             val = df['SuccessRate'].iloc[idx]
#             row[f'Trial_{i}'] = fmt(val)

#         final = df.iloc[max_idx]
#         row['FinalFail']  = fmt(final['FailRate'])
#         row['FinalHalt']  = fmt(final['HaltedRate'])
#         row['AvgSteps']   = fmt(final['AvgSteps'])
#         rows.append(row)

#     col_names = (['Strategy']
#                  + [f'Trial_{i}' for i in trial_indices]
#                  + ['FinalFail', 'FinalHalt', 'AvgSteps'])
#     result = pd.DataFrame(rows, columns=col_names)

#     # Rename trial columns to human-readable
#     trial_labels = {f'Trial_{i}': f'Trial {i+1}' for i in trial_indices}
#     # For HumanEval rename to Iteration
#     if task_name == 'HumanEval':
#         trial_labels = {f'Trial_{i}': f'Iter {i+1}' for i in trial_indices}
#     result = result.rename(columns=trial_labels)

#     out_path = os.path.join(OUTPUT_DIR, f'{task_name.lower()}_table.csv')
#     result.to_csv(out_path, index=False)
#     print(f"\n{task_name} table saved to: {out_path}")
#     print(result.to_string(index=False))
#     return result


# # ── Combined table builder ────────────────────────────────────────────────────

# def build_combined_table():
#     """
#     Final success, fail, halt, avg steps for all methods × all tasks.
#     ExpeL uses eval CSV. All others use final row.
#     """
#     all_methods = ['Simple', 'CoT+GT', 'ReAct', 'Reflexion', 'ExpeL', 'RAR (Ours)']
#     task_configs = [
#         ('HotPotQA',          HOTPOT),
#         ('ALFWorld',          ALFWORLD),
#         ('HumanEval Hard 50', HUMANEVAL),
#     ]

#     # Build columns: for each task → Success, Fail, Halt, Steps
#     cols = ['Method']
#     for task_name, _ in task_configs:
#         cols += [
#             f'{task_name} Success',
#             f'{task_name} Fail',
#             f'{task_name} Halt',
#             f'{task_name} AvgSteps',
#         ]

#     rows = []
#     for method in all_methods:
#         row = {'Method': method}
#         for task_name, config in task_configs:
#             if method not in config:
#                 row[f'{task_name} Success']  = '—'
#                 row[f'{task_name} Fail']     = '—'
#                 row[f'{task_name} Halt']     = '—'
#                 row[f'{task_name} AvgSteps'] = '—'
#                 continue
#             df = load_df(config[method])
#             if df is None:
#                 row[f'{task_name} Success']  = '—'
#                 row[f'{task_name} Fail']     = '—'
#                 row[f'{task_name} Halt']     = '—'
#                 row[f'{task_name} AvgSteps'] = '—'
#             else:
#                 final = df.iloc[-1]
#                 row[f'{task_name} Success']  = fmt(final['SuccessRate'])
#                 row[f'{task_name} Fail']     = fmt(final['FailRate'])
#                 row[f'{task_name} Halt']     = fmt(final['HaltedRate'])
#                 row[f'{task_name} AvgSteps'] = fmt(final['AvgSteps'])
#         rows.append(row)

#     result = pd.DataFrame(rows, columns=cols)
#     out_path = os.path.join(OUTPUT_DIR, 'combined_table.csv')
#     result.to_csv(out_path, index=False)
#     print(f"\nCombined table saved to: {out_path}")
#     print(result.to_string(index=False))
#     return result


# # ── Main ─────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print("=" * 60)
#     print("HotPotQA table")
#     print("=" * 60)
#     build_task_table(HOTPOT, HOTPOT_TRIALS, 'hotpot')

#     print("\n" + "=" * 60)
#     print("ALFWorld table")
#     print("=" * 60)
#     build_task_table(ALFWORLD, ALFWORLD_TRIALS, 'alfworld')

#     print("\n" + "=" * 60)
#     print("HumanEval table")
#     print("=" * 60)
#     build_task_table(HUMANEVAL, HUMANEVAL_ITERS, 'humaneval')

#     print("\n" + "=" * 60)
#     print("Combined table")
#     print("=" * 60)
#     build_combined_table()

#     print("\nDone. All tables saved to ./plots/")




"""
Table Generator
================
Produces 4 CSV files:
  1. hotpot_table.csv     — HotPotQA per-trial success + final metrics
  2. alfworld_table.csv   — ALFWorld per-trial success + final metrics
  3. humaneval_table.csv  — HumanEval per-iteration success + final metrics
  4. combined_table.csv   — Final success rate only, all tasks side by side

ExpeL rule: use Eval CSV for score, not Gather.
Standard methods: final row = final trial.
"""

import os
import glob
import pandas as pd

DATA_ROOT = os.path.expanduser('~/Downloads/reflexion-res')
OUTPUT_DIR = './plots'

# ── CONFIG (same as generate_plots.py) ───────────────────────────────────────

HOTPOT = {
    'ReAct':          f'{DATA_ROOT}/hotpot/react/',
    'CoT+GT':         f'{DATA_ROOT}/hotpot/cot/',
    'Reflexion':      f'{DATA_ROOT}/hotpot/reflexion/',
    'ExpeL':          f'{DATA_ROOT}/hotpot/expel/100_questions_eval_metrics.csv',
    'RAR (Ours)':     f'{DATA_ROOT}/hotpot/retrieval/',
}

ALFWORLD = {
    'ReAct':          f'{DATA_ROOT}/alf/react/',
    'Reflexion':      f'{DATA_ROOT}/alf/reflexion/',
    'ExpeL':          f'{DATA_ROOT}/alf/expel/134_envs_eval_metrics.csv',
    'RAR (Ours)':     f'{DATA_ROOT}/alf/retrieval/',
}

HUMANEVAL = {
    'Simple':         f'{DATA_ROOT}/prog/simple/simple_humaneval_hard/',
    'CoT+GT':         f'{DATA_ROOT}/prog/cot_gt/cot_gt_humaneval_hard/',
    'Reflexion':      f'{DATA_ROOT}/prog/reflexion/reflexion_humaneval_hard/',
    'ExpeL':          f'{DATA_ROOT}/prog/expel/50_problems_metrics_eval_metrics.csv',
    'RAR (Ours)':     f'{DATA_ROOT}/prog/retrieval/retrieval_humaneval_hard',
}

# Which trial indices to show in the per-task tables (0-indexed rows in CSV)
# These correspond to trial 0, midpoint, final
HOTPOT_TRIALS   = [0, 2, 4]   # trials 1, 3, 5
ALFWORLD_TRIALS = [0, 4, 9]   # trials 1, 5, 10
HUMANEVAL_ITERS = [0, 4, 9]   # iterations 1, 5, 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_path(path):
    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
        if not csvs:
            print(f"  WARNING: no CSV in '{path}'")
            return None
        if len(csvs) > 1:
            print(f"  WARNING: multiple CSVs in '{path}', using: {csvs[0]}")
        return csvs[0]
    if os.path.exists(path):
        return path
    print(f"  WARNING: not found: '{path}'")
    return None


def load_df(path):
    p = resolve_path(path)
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
        print(f"  Loaded: {p}  ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def fmt(val, decimals=4):
    """Format float to fixed decimals, or '—' if None."""
    if val is None:
        return '—'
    return f'{val:.{decimals}f}'


# ── Per-task table builder ────────────────────────────────────────────────────

def build_task_table(config, trial_indices, task_name):
    """
    Returns a DataFrame with columns:
      Strategy | Trial_X | Trial_Y | Trial_Z | FinalFail | FinalHalt | AvgSteps
    One row per method.
    """
    rows = []
    for method, path in config.items():
        df = load_df(path)
        if df is None:
            row = {'Strategy': method}
            for i in trial_indices:
                row[f'Trial_{i}'] = '—'
            row['FinalFail']  = '—'
            row['FinalHalt']  = '—'
            row['AvgSteps']   = '—'
            rows.append(row)
            continue

        # Clamp indices to available rows
        max_idx = len(df) - 1
        row = {'Strategy': method}

        for i in trial_indices:
            idx = min(i, max_idx)
            val = df['SuccessRate'].iloc[idx]
            row[f'Trial_{i}'] = fmt(val)

        final = df.iloc[max_idx]
        row['FinalFail']  = fmt(final['FailRate'])
        row['FinalHalt']  = fmt(final['HaltedRate'])
        row['AvgSteps']   = fmt(final['AvgSteps'])
        rows.append(row)

    col_names = (['Strategy']
                 + [f'Trial_{i}' for i in trial_indices]
                 + ['FinalFail', 'FinalHalt', 'AvgSteps'])
    result = pd.DataFrame(rows, columns=col_names)

    # Rename trial columns to human-readable
    trial_labels = {f'Trial_{i}': f'Trial {i+1}' for i in trial_indices}
    # For HumanEval rename to Iteration
    if task_name == 'HumanEval':
        trial_labels = {f'Trial_{i}': f'Iter {i+1}' for i in trial_indices}
    result = result.rename(columns=trial_labels)

    out_path = os.path.join(OUTPUT_DIR, f'{task_name.lower()}_table.csv')
    result.to_csv(out_path, index=False)
    print(f"\n{task_name} table saved to: {out_path}")
    print(result.to_string(index=False))
    return result


# ── Combined table builder ────────────────────────────────────────────────────

def build_combined_table():
    """
    Final success, fail, halt, avg steps for all methods × all tasks.
    ExpeL uses eval CSV. All others use final row.
    """
    all_methods = ['Simple', 'CoT+GT', 'ReAct', 'Reflexion', 'ExpeL', 'RAR (Ours)']
    task_configs = [
        ('HotPotQA',          HOTPOT),
        ('ALFWorld',          ALFWORLD),
        ('HumanEval Hard 50', HUMANEVAL),
    ]

    # Build columns: for each task → Success, Fail, Halt, Steps
    cols = ['Method']
    for task_name, _ in task_configs:
        cols += [
            f'{task_name} Success',
            f'{task_name} Fail',
            f'{task_name} Halt',
            f'{task_name} AvgSteps',
        ]

    rows = []
    for method in all_methods:
        row = {'Method': method}
        for task_name, config in task_configs:
            if method not in config:
                row[f'{task_name} Success']  = '—'
                row[f'{task_name} Fail']     = '—'
                row[f'{task_name} Halt']     = '—'
                row[f'{task_name} AvgSteps'] = '—'
                continue
            df = load_df(config[method])
            if df is None:
                row[f'{task_name} Success']  = '—'
                row[f'{task_name} Fail']     = '—'
                row[f'{task_name} Halt']     = '—'
                row[f'{task_name} AvgSteps'] = '—'
            else:
                final = df.iloc[-1]
                row[f'{task_name} Success']  = fmt(final['SuccessRate'])
                row[f'{task_name} Fail']     = fmt(final['FailRate'])
                row[f'{task_name} Halt']     = fmt(final['HaltedRate'])
                row[f'{task_name} AvgSteps'] = fmt(final['AvgSteps'])
        rows.append(row)

    result = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(OUTPUT_DIR, 'combined_table.csv')
    result.to_csv(out_path, index=False)
    print(f"\nCombined table saved to: {out_path}")
    print(result.to_string(index=False))
    return result


# ── Per-method full trial table ───────────────────────────────────────────────

def build_per_method_trial_tables():
    """
    For each task, produce one CSV per method with all trials:
      Trial | Success | Fail | Halt | AvgSteps | DeltaSuccess
    Saved to ./plots/trials/{task}_{method}.csv
    """
    trials_dir = os.path.join(OUTPUT_DIR, 'trials')
    os.makedirs(trials_dir, exist_ok=True)

    task_configs = [
        ('hotpot',    HOTPOT),
        ('alfworld',  ALFWORLD),
        ('humaneval', HUMANEVAL),
    ]

    for task_name, config in task_configs:
        print(f"\n{'='*60}")
        print(f"Per-trial tables for {task_name}")
        print('='*60)

        for method, path in config.items():
            # Skip gather entries — use eval for ExpeL
            if '(Gather)' in method:
                continue
            clean_method = method.replace('(Eval)', '').strip()

            df = load_df(path)
            if df is None:
                continue

            rows = []
            prev_success = None
            for i, row in df.iterrows():
                trial_num  = int(row.iloc[0])   # first col = trial/iter number
                success    = float(row['SuccessRate'])
                fail       = float(row['FailRate'])
                halt       = float(row['HaltedRate'])
                steps      = float(row['AvgSteps'])
                delta      = f'+{success - prev_success:.4f}' \
                             if prev_success is not None else '---'
                prev_success = success
                rows.append({
                    'Trial':        trial_num,
                    'Success':      fmt(success),
                    'Fail':         fmt(fail),
                    'Halt':         fmt(halt),
                    'AvgSteps':     fmt(steps),
                    'DeltaSuccess': delta,
                })

            result = pd.DataFrame(rows)
            safe_method = clean_method.replace(' ', '_').replace('(', '').replace(')', '')
            out_path = os.path.join(trials_dir,
                                    f'{task_name}_{safe_method}.csv')
            result.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            print(result.to_string(index=False))

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HotPotQA table")
    print("=" * 60)
    build_task_table(HOTPOT, HOTPOT_TRIALS, 'hotpot')

    print("\n" + "=" * 60)
    print("ALFWorld table")
    print("=" * 60)
    build_task_table(ALFWORLD, ALFWORLD_TRIALS, 'alfworld')

    print("\n" + "=" * 60)
    print("HumanEval table")
    print("=" * 60)
    build_task_table(HUMANEVAL, HUMANEVAL_ITERS, 'humaneval')

    print("\n" + "=" * 60)
    print("Combined table")
    print("=" * 60)
    build_combined_table()

    print("\n" + "=" * 60)
    print("Per-method full trial tables")
    print("=" * 60)
    build_per_method_trial_tables()

    print("\nDone. All tables saved to ./plots/")