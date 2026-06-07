"""
run_eda_plots.py
Генерирует EDA графики для всех датасетов.
Сохраняет в experiments/results/{dataset}/figures/
"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs('experiments/results/magnit/figures', exist_ok=True)

STYLE = {
    'bg':      '#0a0a0f',
    'bg2':     '#12121a',
    'bg3':     '#1a1a26',
    'purple':  '#8b78ff',
    'teal':    '#4ecdc4',
    'coral':   '#ff6b6b',
    'amber':   '#ffd93d',
    'text':    '#e8e6f0',
    'dim':     '#7a7890',
    'border':  '#2a2a3a',
}

def style_ax(ax):
    ax.set_facecolor(STYLE['bg2'])
    ax.spines[['top','right','left','bottom']].set_color(STYLE['border'])
    ax.tick_params(colors=STYLE['dim'], labelsize=9)
    ax.yaxis.set_tick_params(labelcolor=STYLE['dim'])
    ax.xaxis.set_tick_params(labelcolor=STYLE['dim'])

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=STYLE['bg'])
    plt.close(fig)
    print(f'  Сохранено: {path}')


def plot_eda(df, treatment_col, outcome_col, out_dir, dataset_name):
    """Генерирует EDA графики для датасета."""
    os.makedirs(f'{out_dir}/figures', exist_ok=True)

    # ── 1. Outcome distribution ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(STYLE['bg'])
    fig.suptitle(f'{dataset_name} — Outcome Distribution', 
                 color=STYLE['text'], fontsize=13, y=1.02)

    # Treatment vs Control mean
    t_mean = df[df[treatment_col]==1][outcome_col].mean()
    c_mean = df[df[treatment_col]==0][outcome_col].mean()
    ax = axes[0]
    style_ax(ax)
    bars = ax.bar(['Control', 'Treatment'], [c_mean, t_mean],
                  color=[STYLE['dim'], STYLE['purple']], alpha=0.85, width=0.5)
    ax.set_title('Mean outcome by group', color=STYLE['text'], fontsize=11)
    ax.set_ylabel(outcome_col, color=STYLE['text'], fontsize=10)
    for bar, val in zip(bars, [c_mean, t_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001*bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', color=STYLE['text'], fontsize=10)
    ax.set_facecolor(STYLE['bg2'])

    # Zero rate
    ax2 = axes[1]
    style_ax(ax2)
    zero_t = (df[df[treatment_col]==1][outcome_col] == 0).mean()
    zero_c = (df[df[treatment_col]==0][outcome_col] == 0).mean()
    bars2 = ax2.bar(['Control\nzero rate', 'Treatment\nzero rate'],
                    [zero_c, zero_t],
                    color=[STYLE['teal'], STYLE['coral']], alpha=0.85, width=0.5)
    ax2.set_title('Zero rate by group', color=STYLE['text'], fontsize=11)
    ax2.set_ylabel('Zero rate', color=STYLE['text'], fontsize=10)
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars2, [zero_c, zero_t]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', va='bottom', color=STYLE['text'], fontsize=10)

    plt.tight_layout()
    save_fig(fig, f'{out_dir}/figures/eda_outcome.png')

    # ── 2. Treatment balance ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(STYLE['bg'])
    style_ax(ax)

    vc = df[treatment_col].value_counts()
    labels = ['Control', 'Treatment']
    sizes  = [vc.get(0, 0), vc.get(1, 0)]
    colors = [STYLE['teal'], STYLE['purple']]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': STYLE['text'], 'fontsize': 11},
    )
    for at in autotexts:
        at.set_color(STYLE['bg'])
        at.set_fontweight('bold')
    ax.set_title(f'{dataset_name} — Group Balance\n'
                 f'Treatment: {sizes[1]:,} | Control: {sizes[0]:,}',
                 color=STYLE['text'], fontsize=11)

    save_fig(fig, f'{out_dir}/figures/eda_balance.png')

    # ── 3. Top features by correlation ───────────────────────────────────────
    service = ['user_id', treatment_col, outcome_col]
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in service if c in df.columns], errors='ignore'
    )
    if not num_df.empty:
        corr = num_df.corrwith(df[outcome_col]).abs().sort_values(ascending=False).head(15)
        corr = corr.dropna()

        if len(corr) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(STYLE['bg'])
            style_ax(ax)

            colors_bar = [STYLE['purple'] if v > 0.1 else STYLE['dim'] for v in corr.values]
            bars = ax.barh(range(len(corr)), corr.values, color=colors_bar, alpha=0.85)
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.index, fontsize=9, color=STYLE['dim'])
            ax.set_xlabel('|Correlation with outcome|', color=STYLE['text'], fontsize=10)
            ax.set_title(f'{dataset_name} — Top features by |corr with outcome|',
                         color=STYLE['text'], fontsize=11)
            ax.axvline(0.1, color=STYLE['amber'], linewidth=1,
                       linestyle='--', alpha=0.6, label='threshold 0.1')
            ax.legend(facecolor=STYLE['bg2'], edgecolor=STYLE['border'],
                      labelcolor=STYLE['text'], fontsize=9)

            for bar, val in zip(bars, corr.values):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', color=STYLE['text'], fontsize=8)

            plt.tight_layout()
            save_fig(fig, f'{out_dir}/figures/eda_top_features.png')

    # ── 4. Missing values ─────────────────────────────────────────────────────
    miss = df.isnull().mean()
    miss = miss[miss > 0].sort_values(ascending=False).head(20)

    if len(miss) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(miss)*0.3 + 1)))
        fig.patch.set_facecolor(STYLE['bg'])
        style_ax(ax)

        colors_miss = [STYLE['coral'] if v > 0.5 else STYLE['amber'] if v > 0.1 else STYLE['teal']
                       for v in miss.values]
        ax.barh(range(len(miss)), miss.values * 100, color=colors_miss, alpha=0.85)
        ax.set_yticks(range(len(miss)))
        ax.set_yticklabels(miss.index, fontsize=9, color=STYLE['dim'])
        ax.set_xlabel('Missing rate (%)', color=STYLE['text'], fontsize=10)
        ax.set_title(f'{dataset_name} — Missing values (top {len(miss)})',
                     color=STYLE['text'], fontsize=11)
        ax.axvline(50, color=STYLE['coral'], linewidth=1,
                   linestyle='--', alpha=0.6, label='50% threshold (drop)')
        ax.legend(facecolor=STYLE['bg2'], edgecolor=STYLE['border'],
                  labelcolor=STYLE['text'], fontsize=9)
        plt.tight_layout()
        save_fig(fig, f'{out_dir}/figures/eda_missing.png')
    else:
        print(f'  Нет пропусков в {dataset_name}')

    print(f'  EDA графики готовы: {out_dir}/figures/')


# ── Запускаем для всех датасетов ──────────────────────────────────────────────

print('=' * 50)
print('EDA ГРАФИКИ — ВСЕ ДАТАСЕТЫ')
print('=' * 50)

# Hillstrom
print('\n[1] Hillstrom...')
from sklift.datasets import fetch_hillstrom
d = fetch_hillstrom()
df_h = d.data.copy()
df_h['spend']     = d.target
df_h['treatment'] = (d.treatment != 'No E-Mail').astype(int)
plot_eda(df_h, 'treatment', 'spend', 'experiments/results/hillstrom', 'Hillstrom')

# Megafon
print('\n[2] Megafon...')
from sklift.datasets import fetch_megafon
d = fetch_megafon()
df_m = d.data.copy()
df_m['target']        = d.target
df_m['treatment_flg'] = (d.treatment == 'treatment').astype(int)
plot_eda(df_m, 'treatment_flg', 'target', 'experiments/results/megafon', 'Megafon')

# Lenta
print('\n[3] Lenta...')
from sklift.datasets import fetch_lenta
d = fetch_lenta()
df_l = d.data.copy()
df_l['response_att']  = d.target
df_l['treatment_flg'] = d.treatment
for col in df_l.select_dtypes(include=['object','string']).columns:
    df_l[col] = df_l[col].astype('category').cat.codes
plot_eda(df_l, 'treatment_flg', 'response_att', 'experiments/results/lenta', 'Lenta')

# Synthetic
print('\n[4] Synthetic...')
rng = np.random.RandomState(42)
n   = 50_000
X   = pd.DataFrame(rng.randn(n, 20), columns=[f'f{i}' for i in range(20)])
w   = rng.binomial(1, 0.5, n)
tau = 0.5 + 0.3*X['f0'] + 0.2*X['f1']
y   = (tau * w + rng.randn(n) * 0.5).clip(0)
df_s = X.copy()
df_s['treatment'] = w
df_s['outcome']   = y
plot_eda(df_s, 'treatment', 'outcome', 'experiments/results/synthetic', 'Synthetic')

# Magnit
print('\n[5] Magnit...')
if os.path.exists('data/train.parquet'):
    df_mag = pd.read_parquet('data/train.parquet')
    plot_eda(df_mag, 'treatment_flg', 'rec_spend',
             'experiments/results/magnit', 'Magnit')
else:
    print('  data/train.parquet не найден — пропускаем')

print('\n' + '=' * 50)
print('ГОТОВО. Все EDA графики сохранены.')
print('=' * 50)