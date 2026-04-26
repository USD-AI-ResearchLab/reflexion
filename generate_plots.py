"""
Learning Curve Plot Generator — Multi-Run (Mean ± SD bands)
=============================================================
Set RUNS to your 3 result root directories (same folder structure in each).
Generates publication-ready figures with shaded ±1 SD error bands.

Run: python generate_plots.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Set these to your 3 run root directories ─────────────────────────────────
RUNS = [
    os.path.expanduser('~/Downloads/reflexion-res/run1'),
    os.path.expanduser('~/Downloads/reflexion-res/run2'),
    os.path.expanduser('~/Downloads/reflexion-res/run3'),
]
OUTPUT_DIR = './plots'

# ── Path templates (relative to each run root) ───────────────────────────────
HOTPOT_TPL = {
    'ReAct':           'hotpot/react/',
    'CoT+GT':          'hotpot/cot/',
    'Reflexion':       'hotpot/reflexion/',
    'ExpeL':           'hotpot/expel/100_questions_eval_metrics.csv',
    'RAR (Ours)':      'hotpot/retrieval/',
}
ALFWORLD_TPL = {
    'ReAct':           'alf/react/',
    'Reflexion':       'alf/reflexion/',
    'ExpeL':           'alf/expel/134_envs_metrics_eval_metrics.csv',
    'RAR (Ours)':      'alf/retrieval/',
}
HUMANEVAL_TPL = {
    'Simple':          'prog/simple/',
    'CoT+GT':          'prog/cot_gt/',
    'Reflexion':       'prog/reflexion/',
    'ExpeL':           'prog/expel/50_problems_metrics_eval_metrics.csv',
    'RAR (Ours)':      'prog/retrieval/',
}

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    'ReAct':      '#2196F3',
    'Simple':     '#4CAF50',
    'CoT+GT':     '#4CAF50',
    'Reflexion':  '#FF9800',
    'ExpeL':      '#9C27B0',
    'RAR (Ours)': '#E91E63',
}
MARKERS = {
    'ReAct':      'o',
    'Simple':     's',
    'CoT+GT':     's',
    'Reflexion':  '^',
    'ExpeL':      'P',
    'RAR (Ours)': 'D',
}
LINESTYLES = {
    'ReAct':      '-',
    'Simple':     '--',
    'CoT+GT':     '--',
    'Reflexion':  '-.',
    'ExpeL':      ':',
    'RAR (Ours)': '-',
}
DEFAULT_COLOR     = '#607D8B'
DEFAULT_MARKER    = 'o'
DEFAULT_LINESTYLE = '-'

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi':      150,
    'axes.grid':       True,
    'grid.linestyle':  '--',
    'grid.alpha':      0.4,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _resolve_csv(path):
    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, '*.csv')))
        return csvs[0] if csvs else None
    return path if os.path.exists(path) else None


def load_data(tpl) -> dict:
    """
    Load multi-run data for a task config template.
    Returns {method: {'x', 'success_mean', 'success_sd',
                       'fail_mean', 'fail_sd',
                       'halted_mean', 'halted_sd',
                       'steps_mean',  'steps_sd'}}
    """
    result = {}
    for method, rel_path in tpl.items():
        dfs = []
        for root in RUNS:
            p = _resolve_csv(os.path.join(root, rel_path))
            if p:
                try:
                    dfs.append(pd.read_csv(p))
                except Exception as e:
                    print(f"  ERROR loading {p}: {e}")
        if not dfs:
            print(f"  WARNING: no data for '{method}'")
            continue

        n     = min(len(df) for df in dfs)
        x_col = dfs[0].columns[0]
        d     = {'x': dfs[0][x_col].tolist()[:n]}

        for key, col in [('success', 'SuccessRate'), ('fail', 'FailRate'),
                         ('halted', 'HaltedRate'),   ('steps', 'AvgSteps')]:
            arr = np.array([[float(df[col].iloc[i]) for i in range(n)]
                            for df in dfs if col in df.columns])
            if arr.size == 0:
                d[f'{key}_mean'] = [0.0] * n
                d[f'{key}_sd']   = [0.0] * n
            else:
                d[f'{key}_mean'] = arr.mean(axis=0).tolist()
                d[f'{key}_sd']   = (arr.std(axis=0, ddof=1)
                                    if arr.shape[0] > 1
                                    else np.zeros(n)).tolist()
        result[method] = d
        print(f"  Loaded '{method}': {len(dfs)} run(s), {n} trial(s)")
    return result


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_line(ax, data, metric):
    """Plot mean line + shaded ±1 SD band for each method."""
    for method, d in data.items():
        x    = d['x']
        mean = d[f'{metric}_mean']
        sd   = d[f'{metric}_sd']
        color = COLORS.get(method, DEFAULT_COLOR)
        ax.plot(x, mean,
                color=color,
                marker=MARKERS.get(method, DEFAULT_MARKER),
                linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                linewidth=2, markersize=5, label=method)
        lo = [m - s for m, s in zip(mean, sd)]
        hi = [m + s for m, s in zip(mean, sd)]
        ax.fill_between(x, lo, hi, color=color, alpha=0.12)


def _style_ax(ax, xlabel, ylabel, title, ylim=None, xticks=None, pct=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=8)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.legend(loc='best', framealpha=0.9)
    if pct:
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0))


# ── Figure builders ───────────────────────────────────────────────────────────

def make_task_figure(data, task_name, xlabel, output_path,
                     ylim_success=(0, 1.05), ylim_steps=None):
    if not data:
        print(f"No data for {task_name}, skipping.")
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f'{task_name} — Learning Curves (mean ± 1 SD)',
                 fontsize=13, fontweight='bold', y=1.02)
    xticks = list(data.values())[0]['x']

    _plot_line(axes[0], data, 'success')
    _style_ax(axes[0], xlabel, 'Success Rate',
              '(a) Success Rate', ylim_success, xticks)

    _plot_line(axes[1], data, 'fail')
    _style_ax(axes[1], xlabel, 'Fail Rate',
              '(b) Fail Rate', (0, 1), xticks)

    _plot_line(axes[2], data, 'halted')
    _style_ax(axes[2], xlabel, 'Halted Rate',
              '(c) Halted Rate', (0, 0.5), xticks)

    _plot_line(axes[3], data, 'steps')
    max_steps = max(v for d in data.values() for v in d['steps_mean'])
    _style_ax(axes[3], xlabel, 'Avg Steps',
              '(d) Avg Steps per Trial',
              ylim_steps or (0, max_steps * 1.2), xticks, pct=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_combined_success_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number',     (0.0, 0.75)),
        ('ALFWorld',  'alfworld',  'Trial Number',     (0.0, 1.05)),
        ('HumanEval', 'humaneval', 'Iteration Number', (0.65, 1.05)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Success Rate per Trial/Iteration — All Tasks (mean ± 1 SD)',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, (name, key, xlabel, ylim) in zip(axes, tasks):
        data = all_data.get(key, {})
        if not data:
            ax.set_title(f'{name} (no data)')
            continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            color = COLORS.get(method, DEFAULT_COLOR)
            mean  = d['success_mean']
            sd    = d['success_sd']
            ax.plot(d['x'], mean,
                    color=color,
                    marker=MARKERS.get(method, DEFAULT_MARKER),
                    linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                    linewidth=2, markersize=5, label=method)
            ax.fill_between(d['x'],
                            [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)],
                            color=color, alpha=0.12)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Success Rate')
        ax.set_ylim(ylim)
        ax.set_xticks(xticks)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_fail_halted_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number'),
        ('ALFWorld',  'alfworld',  'Trial Number'),
        ('HumanEval', 'humaneval', 'Iteration Number'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Fail Rate and Halted Rate — All Tasks (mean ± 1 SD)',
                 fontsize=13, fontweight='bold', y=1.02)
    for col, (name, key, xlabel) in enumerate(tasks):
        data = all_data.get(key, {})
        if not data:
            continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            color = COLORS.get(method, DEFAULT_COLOR)
            kw = dict(color=color,
                      marker=MARKERS.get(method, DEFAULT_MARKER),
                      linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                      linewidth=2, markersize=5, label=method)
            for row_idx, metric in enumerate(['fail', 'halted']):
                mean = d[f'{metric}_mean']
                sd   = d[f'{metric}_sd']
                axes[row_idx][col].plot(d['x'], mean, **kw)
                axes[row_idx][col].fill_between(
                    d['x'],
                    [m - s for m, s in zip(mean, sd)],
                    [m + s for m, s in zip(mean, sd)],
                    color=color, alpha=0.12)
        for row_idx, (ylabel, ylim) in enumerate([('Fail Rate', (0, 1)),
                                                   ('Halted Rate', (0, 0.5))]):
            axes[row_idx][col].set_title(name if row_idx == 0 else '',
                                         fontweight='bold')
            axes[row_idx][col].set_ylabel(ylabel)
            axes[row_idx][col].set_ylim(ylim)
            axes[row_idx][col].set_xticks(xticks)
            axes[row_idx][col].legend(loc='best', framealpha=0.9)
            axes[row_idx][col].yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            if row_idx == 1:
                axes[row_idx][col].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_avg_steps_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number'),
        ('ALFWorld',  'alfworld',  'Trial Number'),
        ('HumanEval', 'humaneval', 'Iteration Number'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Average Steps per Trial/Iteration — All Tasks (mean ± 1 SD)',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, (name, key, xlabel) in zip(axes, tasks):
        data = all_data.get(key, {})
        if not data:
            ax.set_title(f'{name} (no data)')
            continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            color = COLORS.get(method, DEFAULT_COLOR)
            mean  = d['steps_mean']
            sd    = d['steps_sd']
            ax.plot(d['x'], mean,
                    color=color,
                    marker=MARKERS.get(method, DEFAULT_MARKER),
                    linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                    linewidth=2, markersize=5, label=method)
            ax.fill_between(d['x'],
                            [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)],
                            color=color, alpha=0.12)
        max_steps = max(v for d in data.values() for v in d['steps_mean'])
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Avg Steps / Iterations')
        ax.set_ylim(0, max_steps * 1.2)
        ax.set_xticks(xticks)
        ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using {len(RUNS)} run(s): {RUNS}\n")

    print("Loading HotPotQA data...")
    hotpot_data   = load_data(HOTPOT_TPL)

    print("Loading ALFWorld data...")
    alfworld_data = load_data(ALFWORLD_TPL)

    print("Loading HumanEval data...")
    humaneval_data = load_data(HUMANEVAL_TPL)

    all_data = {
        'hotpot':    hotpot_data,
        'alfworld':  alfworld_data,
        'humaneval': humaneval_data,
    }

    print("\nGenerating per-task figures...")
    make_task_figure(hotpot_data,   'HotPotQA',  'Trial Number',
                     f'{OUTPUT_DIR}/hotpotqa_learning_curves.png',
                     ylim_success=(0.0, 0.75), ylim_steps=(0, 8))

    make_task_figure(alfworld_data, 'ALFWorld',  'Trial Number',
                     f'{OUTPUT_DIR}/alfworld_learning_curves.png',
                     ylim_success=(0.0, 1.05), ylim_steps=(0, 30))

    make_task_figure(humaneval_data, 'HumanEval', 'Iteration Number',
                     f'{OUTPUT_DIR}/humaneval_learning_curves.png',
                     ylim_success=(0.65, 1.05), ylim_steps=(0, 5))

    print("\nGenerating combined figures...")
    make_combined_success_figure(all_data, f'{OUTPUT_DIR}/combined_success_rate.png')
    make_fail_halted_figure(all_data,      f'{OUTPUT_DIR}/combined_fail_halted.png')
    make_avg_steps_figure(all_data,        f'{OUTPUT_DIR}/combined_avg_steps.png')

    print(f"\nDone. All plots saved to {OUTPUT_DIR}/")
