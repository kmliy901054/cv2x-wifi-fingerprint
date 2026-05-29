"""Print each model's architecture, parameter count, and per-layer breakdown."""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch

sys.path.insert(0, str(Path(__file__).parent))
import models


def count(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n=== {name} ===')
    print(f'  total params:     {total:>10,}')
    print(f'  trainable params: {trainable:>10,}')
    print(f'  layer breakdown:')
    for n, p in model.named_parameters():
        shape = ' × '.join(str(s) for s in p.shape)
        print(f'    {n:<35s} [{shape:<20s}] {p.numel():>8,}')


D = 80                 # feature dim (BSSIDs)

print('Models built for in_dim={}, K=3 (MDN), embed_dim=32, model_dim=128'.format(D))

count(models.MLPRegressor(in_dim=D),          'MLP (vec input)')
count(models.MDNRegressor(in_dim=D),          'MDN (vec input)')
count(models.MaskedMLP(in_dim=D),             'MaskedMLP (vec + mask)')
count(models.MaskedMDN(in_dim=D),             'MaskedMDN (vec + mask)')
count(models.SetTransformerMDN(num_bssids=D), 'SetTransformerMDN')

# KNN has 0 parameters but stores the training set
print('\n=== KNN ===')
print('  trainable params: 0  (no learning — stores all training (X, y))')
print('  memory cost: train_size × (80 + 2) floats')
