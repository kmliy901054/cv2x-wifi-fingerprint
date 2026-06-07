"""Honest-validation figure for the presentation:
contrasts the test-tuned mirage (0.650), the clean honest headline (0.752),
and the strictest leak-free nested-CV number (~0.95), plus the AMCL floor.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).parent / 'outputs' / 'figures'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: the honest ladder (real, a-priori configs) ──
labels = ['KNN\nk=5', 'Set Trans\nMDN', '+ GP\nsynth', 'Heatmap\n×5', 'Cascade\n×5', 'Cascade\naggressive']
vals   = [1.568, 1.093, 0.906, 0.883, 0.793, 0.752]
colors = ['#999999', '#6699cc', '#ff9933', '#4c9be8', '#2e8b57', '#1a7a3a']
bars = ax1.bar(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.6)
for i, v in enumerate(vals):
    ax1.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax1.axhline(0.3, color='gray', ls=':', lw=1.5)
ax1.text(len(vals)-1, 0.34, 'AMCL noise floor 0.30 m', ha='right', fontsize=9, color='gray')
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Median location error (m)', fontsize=11)
ax1.set_title('The climb (standard train -> test, a-priori configs)',
              fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.75)
ax1.grid(axis='y', alpha=0.3)
ax1.annotate('', xy=(5, 0.80), xytext=(0, 1.60),
             arrowprops=dict(arrowstyle='->', color='#1a7a3a', lw=2, alpha=0.5))
ax1.text(2.4, 1.35, '-52%', color='#1a7a3a', fontsize=15, fontweight='bold', rotation=-18)

# ── Right: honesty audit ──
h_labels = ['Test-tuned\ngreedy', 'Standard\ntrain->test', 'Strict\nnested-CV\n(per-fold)']
h_vals   = [0.650, 0.752, 0.94]
h_colors = ['#d62728', '#1a7a3a', '#8888aa']
h_hatch  = ['xxx', '', '//']
bars2 = ax2.bar(range(3), h_vals, color=h_colors, edgecolor='black',
                linewidth=0.8, hatch=h_hatch, alpha=0.9)
for i, v in enumerate(h_vals):
    ax2.text(i, v + 0.02, f'{v:.3f}' + (' m' if i == 0 else ' m'),
             ha='center', fontsize=12, fontweight='bold')
ax2.text(0, 0.32, 'picked ON the\ntest set\n(+0.07 bias)', ha='center',
         fontsize=9, color='#d62728')
ax2.text(1, 0.32, 'config fixed\nbefore seeing\ntest', ha='center',
         fontsize=9, color='#1a7a3a')
ax2.text(2, 0.40, 'zero leakage\n(only 4/5 data\nper fold)', ha='center',
         fontsize=9, color='#555577')
ax2.axhline(0.3, color='gray', ls=':', lw=1.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels(h_labels, fontsize=10)
ax2.set_ylabel('Median location error (m)', fontsize=11)
ax2.set_title('Which number generalizes?',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Split A median: 0.752 m standard · 0.94 m strict nested-CV · 0.650 m was test-tuned',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = FIG / 'honest_validation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved {out}')
