"""
Table Generator — Multi-Run (Mean ± SD)
========================================
Set RUNS to your 3 result root directories (same folder structure in each).
Produces 4 CSV files in ./plots/:
  combined_table.csv   — final success mean ± SD, all methods × all tasks
  hotpot_table.csv     — per-trial success mean ± SD, HotPotQA
  alfworld_table.csv   — per-trial success mean ± SD, ALFWorld
  humaneval_table.csv  — per-iter  success mean ± SD, HumanEval

Values reported as percentages: "65.0 ± 2.5"
LaTeX column: "$65.0 \\pm 2.5$"
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Set these to your 3 run root directories ─────────────────────────────────
RUNS = [
    os.path.expanduser('~/Downloads/reflexion-res/run1'),
    os.path.expanduser('~/Downloads/reflexion-res/run2'),
    os.path.expanduser('~/Downloads/reflexion-res/run3'),
]
OUTPUT_DIR = './plots'

# ── Path templates (relative to each run root) ───────────────────────────────
HOTPOT_TPL = {
    'ReAct':      'hotpot/react/',
    'CoT+GT':     'hotpot/cot/',
    'Reflexion':  'hotpot/reflexion/',
    'ExpeL':      'hotpot/expel/100_questions_eval_metrics.csv',
    'RAR (Ours)': 'hotpot/retrieval/',
}
ALFWORLD_TPL = {
    'ReAct':      'alf/react/',
    'Reflexion':  'alf/reflexion/',
    'ExpeL':      'alf/expel/134_envs_metrics_eval_metrics.csv',
    'RAR (Ours)': 'alf/retrieval/',
}
HUMANEVAL_TPL = {
    'Simple':     'prog/simple/',
    'CoT+GT':     'prog/cot_gt/',
    'Reflexion':  'prog/reflexion/',
    'ExpeL':      'prog/expel/50_problems_metrics_eval_metrics.csv',
    'RAR (Ours)': 'prog/retrieval/',
}

# Trial/iteration indices to show in per-task tables (0-indexed)
HOTPOT_TRIALS   = [0, 2, 4]   # → Trial 1, 3, 5
ALFWORLD_TRIALS = [0, 4, 9]   # → Trial 1, 5, 10
HUMANEVAL_ITERS = [0, 4, 9]   # → Iter 1, 5, 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_csv(path):
    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
        if not csvs:
            return None
        if len(csvs) > 1:
            print(f"  WARNING: multiple CSVs in '{path}', using: {csvs[0]}")
        return csvs[0]
    return path if os.path.exists(path) else None


def load_df(root, rel_path):
    p = resolve_csv(os.path.join(root, rel_path))
    if p is None:
        print(f"  WARNING: not found: {os.path.join(root, rel_path)}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"  ERROR reading {p}: {e}")
        return None


def get_metric(df, col, idx=-1):
    if df is None or col not in df.columns:
        return None
    idx = min(idx, len(df) - 1) if idx >= 0 else -1
    return float(df[col].iloc[idx])


def mean_sd(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    arr = np.array(v, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd


def fmt_pct(mean, sd):
    """'65.0 ± 2.5' or '—'"""
    if mean is None:
        return '—'
    return f'{mean * 100:.1f} ± {sd * 100:.1f}'


def fmt_latex(mean, sd):
    """'$65.0 \\pm 2.5$' or '—'"""
    if mean is None:
        return '—'
    return f'${mean * 100:.1f} \\pm {sd * 100:.1f}$'


def fmt_raw(mean, sd, decimals=4):
    if mean is None:
        return '—'
    return f'{mean:.{decimals}f} ± {sd:.{decimals}f}'


# ── Combined table ────────────────────────────────────────────────────────────

def build_combined_table():
    """Final success rate mean ± SD for all methods × all tasks."""
    all_methods = ['Simple', 'CoT+GT', 'ReAct', 'Reflexion', 'ExpeL', 'RAR (Ours)']
    task_configs = [
        ('HotPotQA',           HOTPOT_TPL),
        ('ALFWorld',           ALFWORLD_TPL),
        ('HumanEval Hard 50',  HUMANEVAL_TPL),
    ]

    cols = ['Method']
    for task, _ in task_configs:
        cols += [f'{task}', f'{task} LaTeX']

    rows = []
    for method in all_methods:
        row = {'Method': method}
        for task, tpl in task_configs:
            if method not in tpl:
                row[task]               = '—'
                row[f'{task} LaTeX']    = '—'
                continue
            scores = [get_metric(load_df(r, tpl[method]), 'SuccessRate')
                      for r in RUNS]
            m, s = mean_sd(scores)
            row[task]            = fmt_pct(m, s)
            row[f'{task} LaTeX'] = fmt_latex(m, s)
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    out = os.path.join(OUTPUT_DIR, 'combined_table.csv')
    df.to_csv(out, index=False)
    print(f"\nCombined table → {out}")
    print(df.to_string(index=False))
    return df


# ── Per-task table ────────────────────────────────────────────────────────────

def build_task_table(tpl, trial_indices, task_name):
    """Per-trial success mean ± SD plus final fail/halt/steps."""
    rows = []
    for method, rel_path in tpl.items():
        dfs = [load_df(r, rel_path) for r in RUNS]
        row = {'Strategy': method}

        for idx in trial_indices:
            scores = [get_metric(df, 'SuccessRate', idx) for df in dfs]
            m, s = mean_sd(scores)
            label = ('Iter' if task_name.lower() == 'humaneval' else 'Trial')
            row[f'{label} {idx + 1}'] = fmt_pct(m, s)

        for metric, label in [('SuccessRate', 'FinalSuccess'),
                               ('FailRate',    'FinalFail'),
                               ('HaltedRate',  'FinalHalt'),
                               ('AvgSteps',    'AvgSteps')]:
            vals = [get_metric(df, metric) for df in dfs]
            m, s = mean_sd(vals)
            row[label] = fmt_pct(m, s) if metric != 'AvgSteps' else fmt_raw(m, s, 2)

        row['LaTeX'] = fmt_latex(*mean_sd(
            [get_metric(df, 'SuccessRate') for df in dfs]))
        rows.append(row)

    step_label = 'Iter' if task_name.lower() == 'humaneval' else 'Trial'
    trial_cols = [f'{step_label} {i + 1}' for i in trial_indices]
    df = pd.DataFrame(rows, columns=(
        ['Strategy'] + trial_cols +
        ['FinalSuccess', 'FinalFail', 'FinalHalt', 'AvgSteps', 'LaTeX']
    ))
    out = os.path.join(OUTPUT_DIR, f'{task_name.lower()}_table.csv')
    df.to_csv(out, index=False)
    print(f"\n{task_name} table → {out}")
    print(df.to_string(index=False))
    return df


# ── Per-method full-trial table ───────────────────────────────────────────────

def build_per_trial_tables(tpl, task_name, step_label='Trial'):
    """
    For each method: one CSV with every trial row, mean ± SD across runs,
    plus ΔSuccess over the previous trial.
    Saved to ./plots/trials/{task_name}_{method}.csv
    """
    trials_dir = os.path.join(OUTPUT_DIR, 'trials')
    os.makedirs(trials_dir, exist_ok=True)

    for method, rel_path in tpl.items():
        dfs = [load_df(r, rel_path) for r in RUNS]
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            print(f"  WARNING: no data for '{method}', skipping")
            continue

        n = min(len(df) for df in dfs)
        rows = []
        prev_mean = None
        for i in range(n):
            row = {step_label: i + 1}
            for metric, label, pct in [
                ('SuccessRate', 'Success', True),
                ('FailRate',    'Fail',    True),
                ('HaltedRate',  'Halt',    True),
                ('AvgSteps',    'AvgSteps',False),
            ]:
                vals = [get_metric(df, metric, i) for df in dfs]
                m, s = mean_sd(vals)
                row[label] = fmt_pct(m, s) if pct else fmt_raw(m, s, 2)
                if label == 'Success':
                    cur_mean = m

            if prev_mean is None:
                row['DeltaSuccess'] = '---'
            else:
                delta = (cur_mean - prev_mean) * 100
                row['DeltaSuccess'] = f'${delta:+.1f}$'
            prev_mean = cur_mean
            rows.append(row)

        df_out = pd.DataFrame(rows, columns=[
            step_label, 'Success', 'Fail', 'Halt', 'AvgSteps', 'DeltaSuccess'])
        safe = method.replace(' ', '_').replace('(', '').replace(')', '')
        out  = os.path.join(trials_dir, f'{task_name.lower()}_{safe}.csv')
        df_out.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print(df_out.to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using {len(RUNS)} run(s): {RUNS}\n")

    print("=" * 60)
    build_task_table(HOTPOT_TPL,    HOTPOT_TRIALS,   'HotPotQA')
    print("=" * 60)
    build_task_table(ALFWORLD_TPL,  ALFWORLD_TRIALS, 'ALFWorld')
    print("=" * 60)
    build_task_table(HUMANEVAL_TPL, HUMANEVAL_ITERS, 'HumanEval')
    print("=" * 60)
    build_combined_table()

    print("=" * 60)
    print("Per-method full-trial tables")
    print("=" * 60)
    build_per_trial_tables(HOTPOT_TPL,    'hotpotqa',  step_label='Trial')
    build_per_trial_tables(ALFWORLD_TPL,  'alfworld',  step_label='Trial')
    build_per_trial_tables(HUMANEVAL_TPL, 'humaneval', step_label='Iter')

    print(f"\nDone. All tables saved to {OUTPUT_DIR}/")
