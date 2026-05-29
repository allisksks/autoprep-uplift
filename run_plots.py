import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

os.makedirs('experiments/results/figures', exist_ok=True)

# Результаты CV
results = pd.DataFrame({
    'model': ['dr_learner', 'r_learner', 't_learner_ridge',
              'hurdle', 'x_learner', 't_learner_lgb'],
    'cv_point':    [21.9716, 18.779, 18.6657, 17.3208, 18.7345, 17.6857],
    'cv_lower_ci': [16.3668, 13.1483, 12.9805, 12.874, 12.6417, 11.5939],
})

# ── График 1: сравнение моделей ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#12121a')

colors_point = ['#8b78ff' if i == 0 else '#3a3a52' for i in range(len(results))]
colors_ci    = ['#4ecdc4' if i == 0 else '#2a4a48' for i in range(len(results))]

x = np.arange(len(results))
w = 0.35

bars1 = ax.bar(x - w/2, results['cv_point'],    w, label='CV point',    color=colors_point, alpha=0.9)
bars2 = ax.bar(x + w/2, results['cv_lower_ci'], w, label='CV lower CI', color=colors_ci,    alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(results['model'], rotation=20, ha='right',
                   color='#7a7890', fontsize=11)
ax.set_ylabel('uplift@10', color='#e8e6f0', fontsize=12)
ax.set_title('Model comparison — uplift@10 CV results\n(Proprietary retail dataset, 355K rows)',
             color='#e8e6f0', fontsize=13, pad=16)
ax.tick_params(colors='#7a7890')
ax.spines[['top','right','left','bottom']].set_color('#1a1a26')
ax.yaxis.set_tick_params(labelcolor='#7a7890')
ax.legend(facecolor='#1a1a26', edgecolor='#3a3a52', labelcolor='#e8e6f0')
ax.set_ylim(0, 28)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom',
            color='#8b78ff', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom',
            color='#4ecdc4', fontsize=9)

plt.tight_layout()
plt.savefig('experiments/results/figures/model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print('Сохранено: model_comparison.png')

# ── График 2: stability (gap между point и lower CI) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#12121a')

results['gap'] = results['cv_point'] - results['cv_lower_ci']
colors_gap = ['#ff6b6b' if g > 6 else '#4ecdc4' for g in results['gap']]

bars = ax.bar(results['model'], results['gap'], color=colors_gap, alpha=0.9)
ax.set_ylabel('Train/val gap (lower = more stable)', color='#e8e6f0', fontsize=12)
ax.set_title('Overfitting check — point estimate vs lower CI gap',
             color='#e8e6f0', fontsize=13, pad=16)
ax.tick_params(colors='#7a7890')
ax.set_xticklabels(results['model'], rotation=20, ha='right', color='#7a7890', fontsize=11)
ax.spines[['top','right','left','bottom']].set_color('#1a1a26')
ax.yaxis.set_tick_params(labelcolor='#7a7890')
ax.axhline(6, color='#ffd93d', linewidth=1, linestyle='--', alpha=0.6, label='threshold')
ax.legend(facecolor='#1a1a26', edgecolor='#3a3a52', labelcolor='#e8e6f0')

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            color='#e8e6f0', fontsize=10)

plt.tight_layout()
plt.savefig('experiments/results/figures/stability_check.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print('Сохранено: stability_check.png')
print('\nВсе графики сохранены в experiments/results/figures/')