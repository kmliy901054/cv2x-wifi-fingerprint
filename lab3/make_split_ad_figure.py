"""A vs D split results figure (both in-distribution), with C as faded context.
Left: grouped bars (median / mean / p90). Right: error CDF.
Reads the committed prediction npz files; D is produced by train_split_d.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PRED = Path(__file__).parent / 'outputs' / 'predictions'
FIG = Path(__file__).parent / 'outputs' / 'figures'


def err_of(name):
    f = PRED / name
    if not f.exists():
        return None
    d = np.load(f)
    if 'err' in d:
        return d['err']
    return np.linalg.norm(d['pred'] - d['y_true'], axis=1)


eA = err_of('A_random__CascadeAggressiveEnsemble.npz')
eD = err_of('D_stratified__CascadeAggressiveEnsemble.npz')
assert eA is not None and eD is not None, 'need A and D prediction npz'


def stats(e):
    return np.median(e), e.mean(), np.percentile(e, 90)


mA = stats(eA); mD = stats(eD)
print(f'A: median {mA[0]:.3f} mean {mA[1]:.3f} p90 {mA[2]:.3f}  (n={len(eA)})')
print(f'D: median {mD[0]:.3f} mean {mD[1]:.3f} p90 {mD[2]:.3f}  (n={len(eD)})')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── grouped bars ──
metrics = ['median', 'mean', 'p90']
x = np.arange(3); w = 0.36
bA = ax1.bar(x - w/2, mA, w, label=f'A_random (test {len(eA)})',
             color='#1b6ec2', edgecolor='black')
bD = ax1.bar(x + w/2, mD, w, label=f'D_stratified (test {len(eD)})',
             color='#2e8b57', edgecolor='black')
for bars in (bA, bD):
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.03,
                 f'{b.get_height():.2f}', ha='center', fontsize=10, fontweight='bold')
ax1.axhline(0.3, color='gray', ls=':', lw=1.4)
ax1.text(2.4, 0.34, 'AMCL ~0.3 m', ha='right', fontsize=8, color='gray')
ax1.set_xticks(x); ax1.set_xticklabels(['median', 'mean', 'p90'])
ax1.set_ylabel('Location error (m)', fontsize=11)
ax1.set_title('A vs D — both in-distribution, both strong', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

# ── CDF ──
for e, lab, col in [(eA, f'A_random (med {mA[0]:.2f})', '#1b6ec2'),
                     (eD, f'D_stratified (med {mD[0]:.2f})', '#2e8b57')]:
    xs = np.sort(e); ys = np.arange(1, len(xs)+1)/len(xs)
    ax2.plot(xs, ys, color=col, lw=2, label=lab)
ax2.axvline(0.3, color='gray', ls=':', lw=1.4, label='AMCL ~0.3 m')
ax2.set_xlim(0, 4); ax2.set_xlabel('Location error (m)', fontsize=11)
ax2.set_ylabel('Cumulative fraction', fontsize=11)
ax2.set_title('Error CDF: A and D nearly overlap', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle('In-distribution splits A & D: balanced all-times coverage holds accuracy',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = FIG / 'split_a_vs_d.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
