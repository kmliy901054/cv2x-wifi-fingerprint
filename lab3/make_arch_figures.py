"""Produce architecture diagrams for the 5 key Lab 3 models.

Outputs to outputs/figures/architectures/:
  - arch_knn.png              KNN k=5 baseline (1.568 m)
  - arch_masked_mdn.png       MaskedMDN baseline (1.37 m)
  - arch_set_transformer_mdn.png   Set Transformer + MDN (1.09 m)
  - arch_heatmap.png          Heatmap classification head (0.836 m)
  - arch_cascade.png          Cascade coarse→fine WINNER (0.793 m)
  - arch_ladder.drawio        single editable draw.io file with the whole story

Both PNG (for slides) and draw.io XML (for editing) are produced.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PMingLiU', 'SimHei',
                                     'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = Path(__file__).parent / 'outputs' / 'figures' / 'architectures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Block-type colors (matches a "draw.io-ish" palette)
COLORS = {
    'input':  ('#dae8fc', '#6c8ebf'),   # light blue
    'embed':  ('#d5e8d4', '#82b366'),   # light green
    'encoder':('#fff2cc', '#d6b656'),   # light yellow
    'head':   ('#f8cecc', '#b85450'),   # light red
    'output': ('#e1d5e7', '#9673a6'),   # light purple
    'note':   ('#f5f5f5', '#999999'),   # light gray
}


def draw_block(ax, x, y, w, h, text, kind='encoder', fontsize=9, font_weight='normal'):
    fc, ec = COLORS[kind]
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.02,rounding_size=0.05',
                         facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=font_weight, zorder=3, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label=None, fontsize=8, curved=False):
    if curved:
        rad = 0.25
        style = 'arc3,rad=%s' % rad
    else:
        style = 'arc3,rad=0'
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='-|>', mutation_scale=15,
                          color='#333333', linewidth=1.2,
                          connectionstyle=style, zorder=4)
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label,
                fontsize=fontsize, color='#555555', style='italic')


def setup_ax(ax, title, w=10, h=12):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


# ──────────────────────────── individual model diagrams ────────────────────
def fig_knn():
    fig, ax = plt.subplots(figsize=(6, 8))
    setup_ax(ax, 'KNN k=5  (baseline, median 1.568 m)', w=6, h=10)
    draw_block(ax, 3, 9,   4.4, 0.8, 'WiFi scan\n{(BSSID, RSSI)} × N', 'input')
    draw_block(ax, 3, 7.5, 4.4, 0.8, 'Vectorize to fixed 80-D\n(missing BSSID → −100 dBm)', 'embed')
    draw_block(ax, 3, 6,   4.4, 0.8, 'KNN search\n(brute-force, k=5)', 'encoder')
    draw_block(ax, 3, 4.5, 4.4, 0.8, 'Mean of 5 neighbors\' (x, y)', 'head')
    draw_block(ax, 3, 3,   4.4, 0.8, 'Predicted (x, y)', 'output', font_weight='bold')
    draw_block(ax, 3, 1.3, 5,   1.0,
               'No training. Distance metric = Euclidean in 80-D\n'
               'RSSI space. Test ⇒ median 1.568 m.', 'note', fontsize=8)
    for y in [8.6, 7.1, 5.6, 4.1]:
        draw_arrow(ax, 3, y, 3, y - 0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'arch_knn.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ arch_knn.png')


def fig_masked_mdn():
    fig, ax = plt.subplots(figsize=(6, 9))
    setup_ax(ax, 'MaskedMDN  (NN baseline, median 1.37 m)', w=6, h=11)
    draw_block(ax, 3, 10,  4.4, 0.8, 'WiFi scan vector\n80-D RSSI + 80-D mask', 'input')
    draw_block(ax, 3, 8.5, 4.4, 0.8, 'Linear (160 → 256)\nReLU + Dropout', 'encoder')
    draw_block(ax, 3, 7.2, 4.4, 0.8, 'Linear (256 → 128)\nReLU + Dropout', 'encoder')
    draw_block(ax, 3, 5.9, 4.4, 0.8, 'MDN head: K=3 mixtures\nlog π[3] + μ[3,2] + log σ[3,2]', 'head')
    draw_block(ax, 3, 4.5, 4.4, 0.8, 'argmax over π:  pick μ_{k*}', 'head')
    draw_block(ax, 3, 3.1, 4.4, 0.8, 'Predicted (x, y)', 'output', font_weight='bold')
    draw_block(ax, 3, 1.3, 5,   1.2,
               'NLL training loss on 3-mixture Gaussian.\n'
               'Masking lets it handle missing BSSIDs without bias.',
               'note', fontsize=8)
    # edge-to-edge arrows: box-bottom (center-0.4) -> next box-top (center+0.4)
    for y0, y1 in [(9.6, 8.9), (8.1, 7.6), (6.8, 6.3), (5.5, 4.9), (4.1, 3.5)]:
        draw_arrow(ax, 3, y0, 3, y1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'arch_masked_mdn.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ arch_masked_mdn.png')


def fig_set_transformer_mdn():
    fig, ax = plt.subplots(figsize=(7, 10))
    setup_ax(ax, 'Set Transformer + MDN  (median 1.09 m)', w=7, h=12)
    draw_block(ax, 3.5, 11, 5.4, 0.9,
               'Variable-length set of (BSSID_idx, RSSI)\nup to 50 tokens, padded with mask',
               'input')
    draw_block(ax, 3.5, 9.5, 5.4, 0.8, 'Embed BSSID (80+1 → 48-D)', 'embed')
    draw_block(ax, 3.5, 8.2, 5.4, 0.8, 'Token = concat(embed, RSSI_scaled)\n→ Linear → 192-D', 'embed')
    draw_block(ax, 3.5, 6.7, 5.4, 1.0,
               'Set Attention Block × 3\n(heads=4, model_dim=192, masked attention)',
               'encoder')
    draw_block(ax, 3.5, 5.2, 5.4, 0.7, 'Masked mean pool → 192-D', 'encoder')
    draw_block(ax, 3.5, 3.9, 5.4, 0.8, 'MDN head: K=3 mixtures', 'head')
    draw_block(ax, 3.5, 2.6, 5.4, 0.8, 'Predicted (x, y) = μ_{argmax π}', 'output', font_weight='bold')
    draw_block(ax, 3.5, 1.1, 6,   1.0,
               'Permutation-invariant over scan order.\n'
               '5-seed ensemble + GP synth augmentation → 0.89 m',
               'note', fontsize=8)
    for y in [10.55, 9.1, 7.8, 6.2, 4.85, 3.5]:
        draw_arrow(ax, 3.5, y, 3.5, y - 0.45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'arch_set_transformer_mdn.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ arch_set_transformer_mdn.png')


def fig_heatmap():
    fig, ax = plt.subplots(figsize=(7, 10))
    setup_ax(ax, 'Set Transformer + Heatmap  (median 0.883 m, ×5-ens)', w=7, h=12)
    draw_block(ax, 3.5, 11, 5.4, 0.8, 'Set of (BSSID, RSSI) tokens', 'input')
    draw_block(ax, 3.5, 9.6, 5.4, 1.0,
               'Set Transformer encoder\n(embed + SAB×3 + mean pool → 192-D)',
               'encoder')
    draw_block(ax, 3.5, 8, 5.4, 0.8,
               'Linear head: 192 → 1320', 'head')
    draw_block(ax, 3.5, 6.5, 5.4, 0.9,
               '40 × 33 fine grid logits\nover [-6, 10] × [-2, 11] m, 0.4 m/cell',
               'head')
    draw_block(ax, 3.5, 4.95, 5.4, 0.9,
               'softmax(logits) ⊙ free_cell_mask\nrenormalize',
               'head')
    draw_block(ax, 3.5, 3.45, 5.4, 0.8,
               'Expected position\nx̂ = Σ p[c] · cell_center[c]',
               'output', font_weight='bold')
    draw_block(ax, 3.5, 1.5, 6, 1.5,
               'Classification not regression.\nLoss = CE on Gaussian-smoothed soft target (σ=0.4 m)\n'
               '+ free-mask prevents predictions in walls.\n5-seed ensemble (mean xy).',
               'note', fontsize=8)
    for y in [10.6, 9.1, 7.6, 6.05, 4.5]:
        draw_arrow(ax, 3.5, y, 3.5, y - 0.55)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'arch_heatmap.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ arch_heatmap.png')


def fig_cascade():
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_ax(ax, 'Coarse-to-Fine Cascade  [WINNER, median 0.793 m, ×5-ens]', w=10, h=12)
    # Encoder stack centered
    draw_block(ax, 5, 11, 6, 0.8, 'Set of (BSSID, RSSI) tokens', 'input')
    draw_block(ax, 5, 9.6, 6, 1.0,
               'Set Transformer encoder\n(embed + SAB×3 + mean pool → 192-D)',
               'encoder')
    # Two parallel heads
    draw_block(ax, 2.3, 8, 3.6, 0.8, 'Coarse head\nLinear 192 → 90', 'head')
    draw_block(ax, 7.7, 8, 3.6, 0.8, 'Fine head\nLinear 192 → 1320', 'head')
    draw_block(ax, 2.3, 6.6, 3.6, 0.9, '10 × 9 coarse grid\n1.6 m/cell', 'head')
    draw_block(ax, 7.7, 6.6, 3.6, 0.9, '40 × 33 fine grid\n0.4 m/cell', 'head')
    # Gating + multiply
    draw_block(ax, 5, 4.9, 7, 1.1,
               'Cascade gate:\np_fine[c] ← softmax(fine)[c] · softmax(coarse)[parent(c)]\n'
               '⊙ free_cell_mask → renormalize',
               'head', fontsize=9)
    draw_block(ax, 5, 3.3, 6, 0.8,
               'Expected (x, y) = Σ p_fine[c] · cell_center[c]',
               'output', font_weight='bold')
    # Note
    draw_block(ax, 5, 1.6, 8, 1.6,
               'Loss = 0.5 · CE(fine, soft σ=0.4) + 0.3 · CE(coarse, soft σ=1.0) + 0.2 · SmoothL1(E[xy], y_true)\n'
               'Coarse gate kills fine-grid hallucinations far from the coarse mode.\n'
               '5-seed geometric-median ensemble.',
               'note', fontsize=8)
    # Arrows
    draw_arrow(ax, 5, 10.6, 5, 10.1)
    draw_arrow(ax, 5, 9.1, 5, 8.6)            # encoder → split point
    draw_arrow(ax, 4, 8.6, 2.3, 8.4)          # left to coarse head
    draw_arrow(ax, 6, 8.6, 7.7, 8.4)          # right to fine head
    draw_arrow(ax, 2.3, 7.6, 2.3, 7.1)        # coarse head → grid
    draw_arrow(ax, 7.7, 7.6, 7.7, 7.1)        # fine head → grid
    draw_arrow(ax, 2.3, 6.15, 4, 5.5)         # coarse → cascade gate
    draw_arrow(ax, 7.7, 6.15, 6, 5.5)         # fine → cascade gate
    draw_arrow(ax, 5, 4.35, 5, 3.7)           # cascade → expected xy
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'arch_cascade.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ arch_cascade.png')


# ──────────────────────────── draw.io XML for the full ladder ──────────────
DRAWIO_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" agent="lab3-arch-gen" version="23.0.0">
  <diagram name="Lab 3 model ladder" id="ladder">
    <mxGraphModel dx="1422" dy="757" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {cells}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


def drawio_block(cid, x, y, w, h, text, fc='#dae8fc', sc='#6c8ebf'):
    style = (f'rounded=1;whiteSpace=wrap;html=1;'
             f'fillColor={fc};strokeColor={sc};'
             f'fontSize=12;align=center;verticalAlign=middle;')
    return (f'<mxCell id="{cid}" value="{escape(text)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>')


def drawio_edge(eid, src, tgt):
    style = ('endArrow=block;html=1;rounded=0;strokeColor=#333333;'
             'edgeStyle=orthogonalEdgeStyle;')
    return (f'<mxCell id="{eid}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')


def make_drawio_ladder():
    """Single side-by-side draw.io page with the 5 architectures + median."""
    cells = []
    cid = 100
    # Five columns
    columns = [
        # (title, blocks=[(text, kind)], median)
        ('KNN k=5', 1.568, [
            ('WiFi scan\n(variable BSSID/RSSI)', 'input'),
            ('Vectorize 80-D RSSI\n(missing → −100)', 'embed'),
            ('KNN search, k=5', 'encoder'),
            ('Mean of 5 neighbors\' (x, y)', 'head'),
            ('Predicted (x, y)', 'output'),
        ]),
        ('MaskedMDN', 1.371, [
            ('RSSI 80-D + mask 80-D', 'input'),
            ('Linear 160 → 256 → 128', 'encoder'),
            ('MDN head (K=3)\nlog π / μ / σ', 'head'),
            ('argmax π → μ_{k*}', 'head'),
            ('Predicted (x, y)', 'output'),
        ]),
        ('Set Transformer MDN', 1.09, [
            ('Set of (BSSID, RSSI) ≤50', 'input'),
            ('Embed + Linear → 192-D', 'embed'),
            ('SAB × 3\nmasked attention', 'encoder'),
            ('Mean pool → MDN K=3', 'head'),
            ('Predicted (x, y)', 'output'),
        ]),
        ('Heatmap ×5-ens', 0.883, [
            ('Set tokens', 'input'),
            ('Set Transformer encoder', 'encoder'),
            ('Linear → 1320 logits\n(40 × 33 grid, 0.4 m)', 'head'),
            ('softmax ⊙ free_mask', 'head'),
            ('E[xy] = Σ p · cell_center', 'output'),
        ]),
        ('Cascade ×5-ens  [WINNER]', 0.793, [
            ('Set tokens', 'input'),
            ('Set Transformer encoder', 'encoder'),
            ('Coarse 10×9  +  Fine 40×33', 'head'),
            ('Gate: p_fine · p_coarse[parent]\n⊙ free_mask', 'head'),
            ('E[xy] + geom-median over 5 seeds', 'output'),
        ]),
    ]
    BLOCK_W = 260
    BLOCK_H = 60
    GAP_X = 320
    GAP_Y = 90
    HDR_H = 36
    for col_i, (title, med, blocks) in enumerate(columns):
        x = 60 + col_i * GAP_X
        y0 = 60
        # Header
        cells.append(drawio_block(cid, x, y0, BLOCK_W, HDR_H,
                                  f'{title}\nmedian {med:.3f} m',
                                  fc='#f5f5f5', sc='#666666'))
        header_id = cid
        cid += 1
        prev_id = None
        for j, (text, kind) in enumerate(blocks):
            yy = y0 + HDR_H + 20 + j * GAP_Y
            fc, sc = COLORS[kind]
            cells.append(drawio_block(cid, x, yy, BLOCK_W, BLOCK_H, text, fc, sc))
            if prev_id is not None:
                cid += 1
                cells.append(drawio_edge(cid, prev_id, cid - 1))
            prev_id = cid
            cid += 1
    out = DRAWIO_TEMPLATE.format(cells='\n        '.join(cells))
    p = FIG_DIR / 'arch_ladder.drawio'
    p.write_text(out, encoding='utf-8')
    print(f'  ✓ {p.name}  (open in diagrams.net / draw.io)')


def main():
    print(f'[arch] writing to {FIG_DIR}')
    fig_knn()
    fig_masked_mdn()
    fig_set_transformer_mdn()
    fig_heatmap()
    fig_cascade()
    make_drawio_ladder()
    print('Done.')


if __name__ == '__main__':
    main()
