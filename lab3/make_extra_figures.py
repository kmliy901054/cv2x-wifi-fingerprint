"""Extra figures for the expanded presentation:
  failures.png       — what didn't work vs the champion
  train_recipe.png   — augmentation -> 5 seeds -> geometric median
  nested_cv.png      — leak-free nested 5-fold CV protocol
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
FIG = Path(__file__).parent / 'outputs' / 'figures'

# ---------- 1. failures ----------
fig, ax = plt.subplots(figsize=(11, 6))
items = [
    ('Cascade\n(champion)', 0.752, '#1a7a3a'),
    ('3-level cascade\n+12k synth', 0.800, '#b0b0b0'),
    ('A+B combo\n(CNN+cascade)', 0.886, '#d98c8c'),
    ('CNN floor-plan\ncross-attn', 0.907, '#d46a6a'),
    ('Cascade-big\n1.7M params', 0.929, '#d46a6a'),
    ('C-Mixup', 0.942, '#d46a6a'),
    ('Diffusion\nhead', 1.821, '#b22222'),
]
labels = [x[0] for x in items]; vals = [x[1] for x in items]; cols = [x[2] for x in items]
bars = ax.bar(range(len(items)), vals, color=cols, edgecolor='black', linewidth=0.7)
for i, v in enumerate(vals):
    ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(0.752, color='#1a7a3a', ls='--', lw=1.5, alpha=0.7)
ax.text(6.4, 0.78, 'champion 0.752', color='#1a7a3a', fontsize=9, ha='right')
ax.set_xticks(range(len(items))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Split A median error (m)', fontsize=11)
ax.set_title('Bigger and fancier consistently lost: at 1.4k samples, inductive bias > capacity',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 2.0); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); fig.savefig(FIG / 'failures.png', dpi=150); plt.close(fig)
print('saved failures.png')

# ---------- 2. training recipe ----------
fig, ax = plt.subplots(figsize=(12, 5.5)); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
def box(x, y, w, h, text, fc, tc='white', fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                 fc=fc, ec='black', lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold', wrap=True)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                  mutation_scale=18, lw=2, color='#333'))
box(0.3, 3.6, 2.5, 1.6, '1,449 real scans\n+ 5,000 GP synth', '#1b6ec2')
arrow(2.8, 4.4, 3.5, 4.4)
box(3.5, 3.4, 2.6, 2.0, 'Augment each scan:\nRSSI jitter ±4 dBm\nAP dropout 10%', '#ff9933', 'black', 10)
arrow(6.1, 4.4, 6.8, 4.4)
box(6.8, 2.6, 2.4, 3.4, 'Train 5 models\n(seeds 42-46)\ndifferent init +\naug draws', '#2e8b57', 'white', 10)
arrow(9.2, 4.3, 9.9, 4.3)
box(9.9, 3.4, 1.9, 1.8, 'Geometric\nmedian of 5\npredictions', '#6a4ca0', 'white', 10)
ax.text(6, 1.4, 'Why geometric median? It is outlier-robust: one bad seed cannot drag the ensemble,\n'
                'unlike a plain mean. Augmentation injects the sensor noise the model must tolerate.',
        ha='center', fontsize=10, color='#333', style='italic')
ax.set_title('Training recipe: synthetic coverage + noise augmentation + robust ensembling',
             fontsize=13, fontweight='bold')
plt.tight_layout(); fig.savefig(FIG / 'train_recipe.png', dpi=150); plt.close(fig)
print('saved train_recipe.png')

# ---------- 3. nested CV ----------
fig, ax = plt.subplots(figsize=(12, 6)); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 6.5)
ax.text(0.2, 6.1, 'Leak-free nested 5-fold CV: every fold is retrained from scratch',
        fontsize=13, fontweight='bold')
fold_w = 1.6
for k in range(5):
    y = 5.0 - k * 0.95
    ax.text(0.2, y + 0.18, f'Fold {k+1}', fontsize=10, fontweight='bold')
    for j in range(5):
        x = 1.5 + j * (fold_w + 0.05)
        if j == k:
            ax.add_patch(FancyBboxPatch((x, y), fold_w, 0.55, boxstyle='round,pad=0.02',
                                         fc='#ff9933', ec='black'))
            ax.text(x + fold_w/2, y + 0.27, 'TEST', ha='center', va='center',
                    fontsize=9, fontweight='bold')
        else:
            ax.add_patch(FancyBboxPatch((x, y), fold_w, 0.55, boxstyle='round,pad=0.02',
                                         fc='#9bc4e8', ec='black'))
            ax.text(x + fold_w/2, y + 0.27, 'train', ha='center', va='center', fontsize=8)
ax.text(6.0, 0.55,
        'Each fold: retrain the whole model on its 4 training folds + regenerate GP-synth\n'
        'from those folds only, then predict the held-out fold. Stack all 5 -> 1,449 honest\n'
        'out-of-fold predictions. No sample ever scores a model that saw it. -> honest 0.94 m',
        ha='center', fontsize=10, color='#333')
plt.tight_layout(); fig.savefig(FIG / 'nested_cv.png', dpi=150); plt.close(fig)
print('saved nested_cv.png')
