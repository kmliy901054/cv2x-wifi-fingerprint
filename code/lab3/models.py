"""Models for Lab 3 — WiFi indoor localization regression.

  KNNRegressor   sklearn k-NN wrapper, sklearn baseline (deterministic point)
  MLPRegressor   3-layer MLP, MSE loss, deterministic point estimate
  MDNRegressor   3-layer MLP + Mixture Density head (K=3 Gaussians), NLL loss
                 outputs full distribution → calibrated uncertainty
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────
# KNN baseline (classical fingerprinting)
# ─────────────────────────────────────────────────────────────────────

class KNNRegressor:
    """k-NN in RSSI space with inverse-distance weighted pose averaging."""

    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.nn = None
        self.y_tr = None

    def fit(self, X_tr, y_tr):
        self.nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric).fit(X_tr)
        self.y_tr = y_tr
        return self

    def predict(self, X_te):
        dists, idxs = self.nn.kneighbors(X_te)
        if self.k == 1:
            return self.y_tr[idxs[:, 0]]
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        return (w[..., None] * self.y_tr[idxs]).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────
# Deterministic MLP regressor
# ─────────────────────────────────────────────────────────────────────

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=2, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def loss(self, pred, target):
        # Huber loss is more robust to outliers than MSE; (x, y) in meters,
        # delta = 1.0 m is reasonable for indoor scale
        return F.smooth_l1_loss(pred, target, beta=1.0)

    @torch.no_grad()
    def predict_xy(self, x):
        return self.forward(x).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────
# Mixture Density Network (probabilistic)
# ─────────────────────────────────────────────────────────────────────

class MDNRegressor(nn.Module):
    """MDN with K Gaussian mixtures over (x, y).  Diagonal covariance per
    component.  Outputs: log_pi[K], mu[K, 2], log_sigma[K, 2].

    Loss = NLL = -log Σ_k π_k N(y | μ_k, diag(σ_k²))
    """

    def __init__(self, in_dim, hidden=(256, 128), K=3, out_dim=2, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.K = K
        self.out_dim = out_dim
        self.pi_head = nn.Linear(prev, K)
        self.mu_head = nn.Linear(prev, K * out_dim)
        self.logsig_head = nn.Linear(prev, K * out_dim)

    def forward(self, x):
        h = self.backbone(x)
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)              # (B, K)
        mu = self.mu_head(h).view(-1, self.K, self.out_dim)          # (B, K, 2)
        log_sigma = self.logsig_head(h).view(-1, self.K, self.out_dim)
        # clamp σ to [exp(-3)≈0.05 m, exp(2.5)≈12 m] — covers indoor scale
        log_sigma = log_sigma.clamp(min=-3.0, max=2.5)
        return log_pi, mu, log_sigma

    def loss(self, out, target):
        log_pi, mu, log_sigma = out
        # log N(y | μ, diag σ²)  summed over dims
        y = target.unsqueeze(1).expand_as(mu)                         # (B, K, 2)
        sigma = log_sigma.exp()
        log_p_per_dim = -0.5 * ((y - mu) / sigma) ** 2 - log_sigma \
                        - 0.5 * math.log(2 * math.pi)
        log_p = log_p_per_dim.sum(-1)                                  # (B, K)
        log_p_full = log_pi + log_p                                    # log π + log N
        return -torch.logsumexp(log_p_full, dim=-1).mean()             # NLL

    @torch.no_grad()
    def predict_xy(self, x, mode='map'):
        """mode='map': pick highest-π component → its μ.
           mode='mean': mixture mean Σ π_k μ_k."""
        log_pi, mu, log_sigma = self.forward(x)
        if mode == 'map':
            k_star = log_pi.argmax(dim=-1)                             # (B,)
            return mu[torch.arange(mu.size(0)), k_star].cpu().numpy()
        else:
            pi = log_pi.exp().unsqueeze(-1)                            # (B, K, 1)
            return (pi * mu).sum(1).cpu().numpy()

    @torch.no_grad()
    def predict_distribution(self, x):
        """Return (pi[B, K], mu[B, K, 2], sigma[B, K, 2]) as numpy."""
        log_pi, mu, log_sigma = self.forward(x)
        return (log_pi.exp().cpu().numpy(),
                mu.cpu().numpy(),
                log_sigma.exp().cpu().numpy())


# ─────────────────────────────────────────────────────────────────────
# Masked variants — input = concat(rssi_vec, presence_mask)
# Zero-cost upgrade per arxiv 2506.00656 critique of -100 padding
# ─────────────────────────────────────────────────────────────────────

class MaskedMLP(nn.Module):
    """MLP that sees (rssi_vec, presence_mask) — tells the model which APs
    are real vs missing, so it doesn't treat -100 fill as a real signal."""

    def __init__(self, in_dim, hidden=(256, 128), out_dim=2, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        prev = in_dim * 2     # rssi + mask
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask):
        return self.net(torch.cat([x, mask], dim=-1))

    def loss(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=1.0)

    @torch.no_grad()
    def predict_xy(self, x, mask):
        return self.forward(x, mask).cpu().numpy()


class MaskedMDN(nn.Module):
    """MDN counterpart of MaskedMLP."""

    def __init__(self, in_dim, hidden=(256, 128), K=3, out_dim=2, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        prev = in_dim * 2
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.K = K
        self.out_dim = out_dim
        self.pi_head = nn.Linear(prev, K)
        self.mu_head = nn.Linear(prev, K * out_dim)
        self.logsig_head = nn.Linear(prev, K * out_dim)

    def forward(self, x, mask):
        h = self.backbone(torch.cat([x, mask], dim=-1))
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, self.out_dim)
        log_sigma = self.logsig_head(h).view(-1, self.K, self.out_dim)
        log_sigma = log_sigma.clamp(min=-3.0, max=2.5)
        return log_pi, mu, log_sigma

    def loss(self, out, target):
        return MDNRegressor.loss(self, out, target)

    @torch.no_grad()
    def predict_xy(self, x, mask, mode='map'):
        log_pi, mu, log_sigma = self.forward(x, mask)
        if mode == 'map':
            k = log_pi.argmax(dim=-1)
            return mu[torch.arange(mu.size(0)), k].cpu().numpy()
        pi = log_pi.exp().unsqueeze(-1)
        return (pi * mu).sum(1).cpu().numpy()

    @torch.no_grad()
    def predict_distribution(self, x, mask):
        log_pi, mu, log_sigma = self.forward(x, mask)
        return (log_pi.exp().cpu().numpy(),
                mu.cpu().numpy(),
                log_sigma.exp().cpu().numpy())


# ─────────────────────────────────────────────────────────────────────
# Set Transformer (Lee et al. 2019) — adapted for WiFi RSSI per arxiv 2506.00656
#
# Input per scan = unordered set of (BSSID_embedding, RSSI_scalar) pairs.
# Padding handled via attention mask.  Output: K-component MDN over (x, y).
# ─────────────────────────────────────────────────────────────────────

class MAB(nn.Module):
    """Multi-head Attention Block with optional key-padding mask."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        assert dim_V % num_heads == 0
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None

    def forward(self, Q, K, key_mask=None):
        """Q: (B, n, dim_Q), K: (B, m, dim_K), key_mask: (B, m) 1=valid 0=pad"""
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        d = self.dim_V // self.num_heads
        # split heads (along feature) → stack along batch dim
        Q_ = torch.cat(Q.split(d, 2), 0)
        K_ = torch.cat(K.split(d, 2), 0)
        V_ = torch.cat(V.split(d, 2), 0)
        scores = Q_ @ K_.transpose(1, 2) / math.sqrt(self.dim_V)         # (B*H, n, m)
        if key_mask is not None:
            # tile mask across heads
            m_tiled = key_mask.repeat(self.num_heads, 1).unsqueeze(1)    # (B*H, 1, m)
            scores = scores.masked_fill(m_tiled == 0, float('-inf'))
        A = torch.softmax(scores, dim=-1)
        # handle rows where ALL keys are pad (would produce NaN) — set to zero
        A = torch.nan_to_num(A, nan=0.0)
        O = torch.cat((Q_ + A @ V_).split(Q.size(0), 0), 2)              # (B, n, dim_V)
        if self.ln0 is not None:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.ln1 is not None:
            O = self.ln1(O)
        return O


class SAB(nn.Module):
    """Self-Attention Block: MAB(X, X) — captures pairwise AP interactions."""

    def __init__(self, dim, num_heads, ln=True):
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, key_mask=None):
        return self.mab(X, X, key_mask=key_mask)


class PMA(nn.Module):
    """Pooling by Multi-head Attention: maps set of n elements → k learnable seeds."""

    def __init__(self, dim, num_heads, num_seeds, ln=True):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, key_mask=None):
        return self.mab(self.S.expand(X.size(0), -1, -1), X, key_mask=key_mask)


class SetTransformerMDN(nn.Module):
    """Set Transformer encoder + MDN head, taking variable-size scans as sets.

    Per-AP feature: cat(BSSID_embedding[D], normalized_rssi[1]) → input_dim
    """

    def __init__(self, num_bssids, embed_dim=32, model_dim=128, num_heads=4,
                  num_sab=2, K=3, out_dim=2, dropout=0.1):
        super().__init__()
        # +1 for the PAD index
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        self.K = K
        self.out_dim = out_dim
        self.pi_head = nn.Linear(model_dim, K)
        self.mu_head = nn.Linear(model_dim, K * out_dim)
        self.logsig_head = nn.Linear(model_dim, K * out_dim)

    def encode(self, bssid_idx, rssi, mask):
        """bssid_idx: (B, M) long, rssi: (B, M) float, mask: (B, M) float."""
        emb = self.embed(bssid_idx)                            # (B, M, embed_dim)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)        # (B, M, embed+1)
        x = self.input_proj(x)                                  # (B, M, model_dim)
        # zero out padded positions (so input doesn't have garbage even before attention)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)         # (B, model_dim)
        return self.dropout(pooled)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, self.out_dim)
        log_sigma = self.logsig_head(h).view(-1, self.K, self.out_dim)
        log_sigma = log_sigma.clamp(min=-3.0, max=2.5)
        return log_pi, mu, log_sigma

    def loss(self, out, target):
        return MDNRegressor.loss(self, out, target)

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask, mode='map'):
        log_pi, mu, log_sigma = self.forward(bssid_idx, rssi, mask)
        if mode == 'map':
            k = log_pi.argmax(dim=-1)
            return mu[torch.arange(mu.size(0)), k].cpu().numpy()
        pi = log_pi.exp().unsqueeze(-1)
        return (pi * mu).sum(1).cpu().numpy()

    @torch.no_grad()
    def predict_distribution(self, bssid_idx, rssi, mask):
        log_pi, mu, log_sigma = self.forward(bssid_idx, rssi, mask)
        return (log_pi.exp().cpu().numpy(),
                mu.cpu().numpy(),
                log_sigma.exp().cpu().numpy())


class SetTransformerHeatmap(nn.Module):
    """Set Transformer encoder + heatmap output head over a 2-D grid.

    Replaces the K-component Gaussian MDN with a per-cell logits head.  At
    inference we softmax the logits, multiply by a precomputed free-cell mask
    (from psquare.pgm) so impossible cells are zeroed, renormalize, and take
    the expected (x, y) over cell centres.  This gives sub-cell precision
    while enforcing the physical constraint that predictions must lie in
    free space.

    Training loss is a mix:
      L = α · CE(logits, soft_target)  +  (1 − α) · SmoothL1(expected_xy, y)
    where soft_target is a Gaussian-smoothed distribution centred at y with
    bandwidth `label_sigma`, masked by free cells.  α = ce_weight.

    cell_xy / free_mask are passed in as numpy arrays and registered as
    buffers so they move with .to(device).
    """

    def __init__(self, num_bssids, cell_xy, free_mask,
                 label_sigma=0.5, ce_weight=0.5,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, out_dim=2, dropout=0.3):
        super().__init__()
        # encoder identical to SetTransformerMDN ────────────────────────
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        # heatmap head ─────────────────────────────────────────────────
        G = int(cell_xy.shape[0])
        self.G = G
        self.head = nn.Linear(model_dim, G)
        self.label_sigma = float(label_sigma)
        self.ce_weight = float(ce_weight)
        # buffers (move with .to(device), saved in state_dict)
        self.register_buffer('cell_xy',
                              torch.from_numpy(cell_xy.astype(np.float32)))   # (G, 2)
        self.register_buffer('free_mask',
                              torch.from_numpy(free_mask.astype(np.float32))) # (G,)
        # precomputed log of mask (with floor to avoid -inf in unused paths)
        log_mask = np.where(free_mask, 0.0, -1e9).astype(np.float32)
        self.register_buffer('log_free_mask', torch.from_numpy(log_mask))      # (G,)

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        return self.dropout(pooled)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        return self.head(h)                          # (B, G) raw logits

    def _soft_target(self, target):
        """Gaussian-smoothed label distribution masked by free cells.
        target: (B, 2)  →  (B, G) probabilities.
        """
        d2 = ((target.unsqueeze(1) - self.cell_xy.unsqueeze(0)) ** 2).sum(-1)
        log_p = -d2 / (2.0 * self.label_sigma ** 2)
        log_p = log_p + self.log_free_mask.unsqueeze(0)
        return F.softmax(log_p, dim=-1)               # (B, G)

    def _expected_xy(self, logits):
        """Masked-softmax → expected (x, y) over free cells."""
        masked = logits + self.log_free_mask.unsqueeze(0)
        prob = F.softmax(masked, dim=-1)              # (B, G)
        return prob @ self.cell_xy                    # (B, 2)

    def loss(self, logits, target):
        soft = self._soft_target(target)              # (B, G)
        log_prob = F.log_softmax(logits, dim=-1)
        ce = -(soft * log_prob).sum(-1).mean()
        exp_xy = self._expected_xy(logits)
        mse = F.smooth_l1_loss(exp_xy, target, beta=1.0)
        return self.ce_weight * ce + (1.0 - self.ce_weight) * mse

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        logits = self.forward(bssid_idx, rssi, mask)
        return self._expected_xy(logits).cpu().numpy()

    @torch.no_grad()
    def predict_heatmap(self, bssid_idx, rssi, mask):
        """Return the masked softmax distribution as (B, G) numpy array."""
        logits = self.forward(bssid_idx, rssi, mask)
        masked = logits + self.log_free_mask.unsqueeze(0)
        return F.softmax(masked, dim=-1).cpu().numpy()


class MapEncoder(nn.Module):
    """Encode psquare.pgm as a fixed sequence of spatial feature tokens.

    Input: 3-channel (free, occupied, unknown) downsampled occupancy grid.
    CNN: 3 stride-2 convs → flatten to (T, model_dim) tokens.
    Learnable 2D positional encoding added so each token knows where it sits.

    Forward returns (1, T, model_dim) — broadcast to batch in the caller.
    """

    def __init__(self, map_pgm_path, target_hw=(56, 64), model_dim=192):
        super().__init__()
        import yaml
        from PIL import Image
        with open(map_pgm_path) as f:
            m = yaml.safe_load(f)
        pgm = np.array(Image.open(map_pgm_path.parent / m['image']))
        free = (pgm >= 200).astype(np.float32)
        occ = (pgm <= 50).astype(np.float32)
        unk = ((pgm > 50) & (pgm < 200)).astype(np.float32)
        rgb = np.stack([free, occ, unk], axis=0)            # (3, H, W)
        t = torch.from_numpy(rgb).unsqueeze(0)              # (1, 3, H, W)
        t = F.interpolate(t, size=target_hw, mode='bilinear',
                            align_corners=False)
        self.register_buffer('map_input', t)                 # (1, 3, h, w) fixed

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, model_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(model_dim), nn.ReLU(),
        )
        with torch.no_grad():
            out = self.conv(self.map_input)
            _, _, h, w = out.shape
        self.num_tokens = h * w
        self.pos_enc = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))
        nn.init.normal_(self.pos_enc, std=0.02)

    def forward(self):
        feat = self.conv(self.map_input)                     # (1, C, h, w)
        tokens = feat.flatten(2).transpose(1, 2)             # (1, T, C)
        return tokens + self.pos_enc


class SetTransformerHeatmapMap(nn.Module):
    """Set Transformer encoder + Floor-plan CNN Cross-Attention + Heatmap head.

    Adds spatial geometry awareness on top of plain Heatmap model:
      1. CNN encodes the floor plan once into (T, model_dim) tokens
      2. After PMA pool, the scan vector queries those map tokens via
         cross-attention → refined vector that "knows the geometry"
      3. Heatmap head + free-cell-masked softmax → expected (x, y)

    Loss identical to SetTransformerHeatmap (mixed CE + SmoothL1).
    """

    def __init__(self, num_bssids, cell_xy, free_mask, map_pgm_path,
                 label_sigma=0.5, ce_weight=0.5,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, out_dim=2, dropout=0.3,
                 map_target_hw=(56, 64)):
        super().__init__()
        # set encoder ─────────────────────────────────────────────
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        # map encoder + cross-attn ────────────────────────────────
        self.map_encoder = MapEncoder(map_pgm_path,
                                        target_hw=map_target_hw,
                                        model_dim=model_dim)
        self.cross_attn = MAB(model_dim, model_dim, model_dim,
                                num_heads, ln=True)
        self.dropout = nn.Dropout(dropout)
        # heatmap head + buffers (same shape as plain heatmap) ────
        G = int(cell_xy.shape[0])
        self.G = G
        self.head = nn.Linear(model_dim, G)
        self.label_sigma = float(label_sigma)
        self.ce_weight = float(ce_weight)
        self.register_buffer('cell_xy',
                              torch.from_numpy(cell_xy.astype(np.float32)))
        self.register_buffer('free_mask',
                              torch.from_numpy(free_mask.astype(np.float32)))
        log_mask = np.where(free_mask, 0.0, -1e9).astype(np.float32)
        self.register_buffer('log_free_mask', torch.from_numpy(log_mask))

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)          # (B, D)
        # cross-attention to map tokens
        B = pooled.size(0)
        map_tokens = self.map_encoder()                          # (1, T, D)
        map_tokens = map_tokens.expand(B, -1, -1)                # (B, T, D)
        q = pooled.unsqueeze(1)                                   # (B, 1, D)
        refined = self.cross_attn(q, map_tokens).squeeze(1)      # (B, D)
        return self.dropout(refined)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        return self.head(h)

    def _soft_target(self, target):
        d2 = ((target.unsqueeze(1) - self.cell_xy.unsqueeze(0)) ** 2).sum(-1)
        log_p = -d2 / (2.0 * self.label_sigma ** 2) + self.log_free_mask.unsqueeze(0)
        return F.softmax(log_p, dim=-1)

    def _expected_xy(self, logits):
        masked = logits + self.log_free_mask.unsqueeze(0)
        prob = F.softmax(masked, dim=-1)
        return prob @ self.cell_xy

    def loss(self, logits, target):
        soft = self._soft_target(target)
        log_prob = F.log_softmax(logits, dim=-1)
        ce = -(soft * log_prob).sum(-1).mean()
        exp_xy = self._expected_xy(logits)
        mse = F.smooth_l1_loss(exp_xy, target, beta=1.0)
        return self.ce_weight * ce + (1.0 - self.ce_weight) * mse

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        logits = self.forward(bssid_idx, rssi, mask)
        return self._expected_xy(logits).cpu().numpy()


class SetTransformerHeatmapCascade(nn.Module):
    """Coarse-to-fine cascaded heatmap head.

    Two heads on the same Set-Transformer-pooled (B, model_dim) vector:
      coarse: (B, G_c)  e.g. 10×8 = 80 regions of ~1.6 m each
      fine:   (B, G_f)  e.g. 40×33 = 1320 cells of 0.4 m each

    Training loss combines:
      0.5 · CE(fine,  Gaussian-smoothed label, σ_f = 0.4 m)
    + 0.3 · CE(coarse, Gaussian-smoothed label, σ_c = 1.0 m)
    + 0.2 · SmoothL1(E[xy_fine_gated_by_coarse], y_true)

    Inference applies a cascade refinement:
      p_fine_refined[c]  =  softmax(fine)[c] · softmax(coarse)[parent(c)]
      renormalize, expected (x, y).
    The coarse gate kills fine-grid hallucinations far from the coarse mode.

    Parent mapping `fine_to_coarse` is precomputed in data.build_fine_to_coarse.
    """

    def __init__(self, num_bssids,
                 fine_cell_xy, fine_free_mask,
                 coarse_cell_xy, coarse_free_mask,
                 fine_to_coarse,
                 fine_sigma=0.4, coarse_sigma=1.0,
                 ce_fine_w=0.5, ce_coarse_w=0.3, mse_w=0.2,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, out_dim=2, dropout=0.3):
        super().__init__()
        # encoder identical to SetTransformerMDN ────────────────────────
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        # two heads ──────────────────────────────────────────────────
        Gf = int(fine_cell_xy.shape[0])
        Gc = int(coarse_cell_xy.shape[0])
        self.Gf = Gf; self.Gc = Gc
        self.fine_head = nn.Linear(model_dim, Gf)
        self.coarse_head = nn.Linear(model_dim, Gc)
        self.fine_sigma = float(fine_sigma)
        self.coarse_sigma = float(coarse_sigma)
        self.ce_fine_w = float(ce_fine_w)
        self.ce_coarse_w = float(ce_coarse_w)
        self.mse_w = float(mse_w)
        # buffers
        self.register_buffer('fine_xy',
                              torch.from_numpy(fine_cell_xy.astype(np.float32)))
        self.register_buffer('coarse_xy',
                              torch.from_numpy(coarse_cell_xy.astype(np.float32)))
        self.register_buffer('fine_mask',
                              torch.from_numpy(fine_free_mask.astype(np.float32)))
        self.register_buffer('coarse_mask',
                              torch.from_numpy(coarse_free_mask.astype(np.float32)))
        f_log = np.where(fine_free_mask, 0.0, -1e9).astype(np.float32)
        c_log = np.where(coarse_free_mask, 0.0, -1e9).astype(np.float32)
        self.register_buffer('log_fine_mask', torch.from_numpy(f_log))
        self.register_buffer('log_coarse_mask', torch.from_numpy(c_log))
        self.register_buffer('fine_to_coarse',
                              torch.from_numpy(fine_to_coarse.astype(np.int64)))

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        return self.dropout(pooled)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        return self.fine_head(h), self.coarse_head(h)   # (B, Gf), (B, Gc)

    def _soft_target(self, target, cell_xy, sigma, log_mask):
        d2 = ((target.unsqueeze(1) - cell_xy.unsqueeze(0)) ** 2).sum(-1)
        log_p = -d2 / (2.0 * sigma ** 2) + log_mask.unsqueeze(0)
        return F.softmax(log_p, dim=-1)

    def _gated_fine_prob(self, fine_logits, coarse_logits):
        """fine_prob[c] ← softmax(fine)[c] · softmax(coarse)[parent(c)], renormalized."""
        fine_p = F.softmax(fine_logits + self.log_fine_mask.unsqueeze(0),
                            dim=-1)                                  # (B, Gf)
        coarse_p = F.softmax(coarse_logits + self.log_coarse_mask.unsqueeze(0),
                              dim=-1)                                 # (B, Gc)
        # gate: lookup coarse prob at each fine cell's parent
        gate = coarse_p[:, self.fine_to_coarse]                       # (B, Gf)
        refined = fine_p * gate
        refined = refined / refined.sum(-1, keepdim=True).clamp(min=1e-12)
        return refined

    def loss(self, out, target):
        fine_logits, coarse_logits = out
        soft_f = self._soft_target(target, self.fine_xy,
                                     self.fine_sigma, self.log_fine_mask)
        soft_c = self._soft_target(target, self.coarse_xy,
                                     self.coarse_sigma, self.log_coarse_mask)
        log_f = F.log_softmax(fine_logits, dim=-1)
        log_c = F.log_softmax(coarse_logits, dim=-1)
        ce_f = -(soft_f * log_f).sum(-1).mean()
        ce_c = -(soft_c * log_c).sum(-1).mean()
        refined = self._gated_fine_prob(fine_logits, coarse_logits)
        exp_xy = refined @ self.fine_xy                               # (B, 2)
        mse = F.smooth_l1_loss(exp_xy, target, beta=1.0)
        return self.ce_fine_w * ce_f + self.ce_coarse_w * ce_c + self.mse_w * mse

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        fine_logits, coarse_logits = self.forward(bssid_idx, rssi, mask)
        refined = self._gated_fine_prob(fine_logits, coarse_logits)
        return (refined @ self.fine_xy).cpu().numpy()


class DiffusionRegressionHead(nn.Module):
    """Conditional diffusion model over (x, y) ∈ ℝ².

    Forward process: y_t = sqrt(α̅_t) · y_0 + sqrt(1 − α̅_t) · ε
    Network ε_θ(y_t, t, cond) predicts the noise ε from a noisy sample y_t,
    a diffusion step t, and a condition vector (the Set Transformer pooled
    output).

    Inference uses DDIM with `num_steps` (≪ T) → (x, y) prediction.

    Score scale: lab is ~16×13 m, so we standardize y to roughly N(0, 1)
    by dividing by `pose_scale` (~5 m) before adding noise — keeps signal
    and noise balanced.
    """

    def __init__(self, cond_dim=192, hidden=256, T=100, pose_scale=5.0):
        super().__init__()
        self.T = T
        self.pose_scale = float(pose_scale)
        # Cosine β schedule (Nichol & Dhariwal 2021)
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float32)
        alpha_bar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        # clamp to avoid div by zero at t = T
        alpha_bar = alpha_bar.clamp(min=1e-6)
        self.register_buffer('alpha_bar', alpha_bar)         # (T+1,)
        # Time embedding MLP
        self.t_emb_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(self.t_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Noise predictor: input = (y_t[2] | t_emb[hidden] | cond[cond_dim])
        in_dim = 2 + hidden + cond_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 2)

    def _sinusoidal_t(self, t):
        """t: (B,) long → (B, t_emb_dim) sinusoidal positional embedding."""
        half = self.t_emb_dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                            torch.arange(half, device=t.device,
                                          dtype=torch.float32) / max(half - 1, 1))
        x = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def _predict_noise(self, y_t, t, cond):
        t_emb = self.time_mlp(self._sinusoidal_t(t))
        x = torch.cat([y_t, t_emb, cond], dim=-1)
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h)) + h
        h = F.silu(self.fc3(h)) + h
        return self.fc_out(h)

    def diffusion_loss(self, y_0_raw, cond):
        """y_0_raw: (B, 2) in meters, cond: (B, cond_dim)."""
        y_0 = y_0_raw / self.pose_scale                       # normalize
        B = y_0.size(0)
        t = torch.randint(1, self.T + 1, (B,), device=y_0.device)
        eps = torch.randn_like(y_0)
        ab = self.alpha_bar[t].unsqueeze(-1)                 # (B, 1)
        y_t = torch.sqrt(ab) * y_0 + torch.sqrt(1.0 - ab) * eps
        eps_pred = self._predict_noise(y_t, t, cond)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, cond, num_steps=20, num_samples=1):
        """DDIM sampling.  Returns (B, 2) in meters.
        num_samples>1: average multiple trajectories per sample."""
        B = cond.size(0)
        device = cond.device
        # initial y_T ~ N(0, 1) in normalized coords
        all_preds = []
        for _ in range(num_samples):
            y = torch.randn(B, 2, device=device)
            step_idx = np.linspace(self.T, 1, num_steps + 1).astype(int)
            for i in range(num_steps):
                t_now = step_idx[i]
                t_next = step_idx[i + 1]
                t_t = torch.full((B,), t_now, device=device, dtype=torch.long)
                eps_pred = self._predict_noise(y, t_t, cond)
                ab_now = self.alpha_bar[t_now]
                ab_next = self.alpha_bar[t_next]
                # estimate y_0 then DDIM step
                y_0_hat = (y - torch.sqrt(1.0 - ab_now) * eps_pred) / torch.sqrt(ab_now)
                y_0_hat = y_0_hat.clamp(-3.0, 3.0)            # bounded lab coords
                y = torch.sqrt(ab_next) * y_0_hat + torch.sqrt(1.0 - ab_next) * eps_pred
            all_preds.append(y * self.pose_scale)             # un-normalize
        return torch.stack(all_preds, dim=0).mean(dim=0)


class SetTransformerDiffusion(nn.Module):
    """Set Transformer encoder + conditional diffusion head over (x, y).

    Encoder identical to SetTransformerMDN.  Output is (x, y) sampled from a
    learned conditional distribution P(y | scan).  Training is denoising-
    MSE on noise prediction; inference is DDIM with `num_sample_steps` steps.

    Note: no free-cell mask in diffusion (continuous output) → may underperform
    Cascade for this structured-output problem, but is a useful contrast.
    """

    def __init__(self, num_bssids,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, dropout=0.3,
                 diff_T=100, diff_hidden=256, pose_scale=5.0,
                 num_sample_steps=20, num_inf_samples=4):
        super().__init__()
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        self.diff_head = DiffusionRegressionHead(
            cond_dim=model_dim, hidden=diff_hidden, T=diff_T,
            pose_scale=pose_scale)
        self.num_sample_steps = num_sample_steps
        self.num_inf_samples = num_inf_samples

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        return self.dropout(pooled)

    def forward(self, bssid_idx, rssi, mask):
        # Returns condition vector; loss handled separately
        return self.encode(bssid_idx, rssi, mask)

    def loss(self, cond, target):
        return self.diff_head.diffusion_loss(target, cond)

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        cond = self.encode(bssid_idx, rssi, mask)
        return self.diff_head.sample(cond,
                                       num_steps=self.num_sample_steps,
                                       num_samples=self.num_inf_samples).cpu().numpy()


class SetTransformerHeatmap3Cascade(nn.Module):
    """Three-resolution coarse-medium-fine cascade heatmap.

    Three heads on the same Set Transformer pooled vector:
      coarse: ~1.6 m cells   (e.g. 10x9   = 90 cells)
      medium: ~0.6 m cells   (e.g. 27x22  = 594 cells)
      fine:   ~0.25 m cells  (e.g. 64x52  = 3328 cells)

    Hierarchical gating at inference:
      p_medium_gated[m] ∝ softmax(medium)[m] · softmax(coarse)[parent_c(m)]
      p_fine_gated[f]   ∝ softmax(fine)[f]   · p_medium_gated[parent_m(f)]

    Training loss combines CE at each level (Gaussian-smoothed labels with
    bandwidth proportional to cell size) and a SmoothL1 on the gated
    expected (x, y).
    """

    def __init__(self, num_bssids,
                 grids,                                 # dict from build_3level_grids
                 fine_sigma=0.25, medium_sigma=0.6, coarse_sigma=1.0,
                 w_fine=0.4, w_medium=0.3, w_coarse=0.15, w_mse=0.15,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, out_dim=2, dropout=0.3):
        super().__init__()
        # set encoder (same shape) ─────────────────────────────────
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        # three heads ──────────────────────────────────────────────
        Gf = int(grids['fine_xy'].shape[0])
        Gm = int(grids['medium_xy'].shape[0])
        Gc = int(grids['coarse_xy'].shape[0])
        self.Gf, self.Gm, self.Gc = Gf, Gm, Gc
        self.fine_head = nn.Linear(model_dim, Gf)
        self.medium_head = nn.Linear(model_dim, Gm)
        self.coarse_head = nn.Linear(model_dim, Gc)
        self.fine_sigma = float(fine_sigma)
        self.medium_sigma = float(medium_sigma)
        self.coarse_sigma = float(coarse_sigma)
        self.w_fine = float(w_fine)
        self.w_medium = float(w_medium)
        self.w_coarse = float(w_coarse)
        self.w_mse = float(w_mse)
        # buffers ──────────────────────────────────────────────────
        self.register_buffer('fine_xy',
                              torch.from_numpy(grids['fine_xy'].astype(np.float32)))
        self.register_buffer('medium_xy',
                              torch.from_numpy(grids['medium_xy'].astype(np.float32)))
        self.register_buffer('coarse_xy',
                              torch.from_numpy(grids['coarse_xy'].astype(np.float32)))
        f_log = np.where(grids['fine_mask'], 0.0, -1e9).astype(np.float32)
        m_log = np.where(grids['medium_mask'], 0.0, -1e9).astype(np.float32)
        c_log = np.where(grids['coarse_mask'], 0.0, -1e9).astype(np.float32)
        self.register_buffer('log_fine_mask', torch.from_numpy(f_log))
        self.register_buffer('log_medium_mask', torch.from_numpy(m_log))
        self.register_buffer('log_coarse_mask', torch.from_numpy(c_log))
        self.register_buffer('fine_to_medium',
                              torch.from_numpy(grids['fine_to_medium'].astype(np.int64)))
        self.register_buffer('medium_to_coarse',
                              torch.from_numpy(grids['medium_to_coarse'].astype(np.int64)))

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        return self.dropout(pooled)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        return self.fine_head(h), self.medium_head(h), self.coarse_head(h)

    def _soft_target(self, target, cell_xy, sigma, log_mask):
        d2 = ((target.unsqueeze(1) - cell_xy.unsqueeze(0)) ** 2).sum(-1)
        log_p = -d2 / (2.0 * sigma ** 2) + log_mask.unsqueeze(0)
        return F.softmax(log_p, dim=-1)

    def _gated_fine_prob(self, fine_l, medium_l, coarse_l):
        """fine_prob ∝ softmax(fine) · softmax(medium)[parent_m] · softmax(coarse)[parent_c]."""
        fine_p = F.softmax(fine_l + self.log_fine_mask.unsqueeze(0), dim=-1)
        medium_p = F.softmax(medium_l + self.log_medium_mask.unsqueeze(0), dim=-1)
        coarse_p = F.softmax(coarse_l + self.log_coarse_mask.unsqueeze(0), dim=-1)
        # gate medium by coarse first
        med_gate = coarse_p[:, self.medium_to_coarse]      # (B, Gm)
        medium_gated = medium_p * med_gate
        medium_gated = medium_gated / medium_gated.sum(-1, keepdim=True).clamp(min=1e-12)
        # gate fine by medium_gated
        fine_gate = medium_gated[:, self.fine_to_medium]   # (B, Gf)
        refined = fine_p * fine_gate
        refined = refined / refined.sum(-1, keepdim=True).clamp(min=1e-12)
        return refined

    def loss(self, out, target):
        fine_l, medium_l, coarse_l = out
        soft_f = self._soft_target(target, self.fine_xy,
                                     self.fine_sigma, self.log_fine_mask)
        soft_m = self._soft_target(target, self.medium_xy,
                                     self.medium_sigma, self.log_medium_mask)
        soft_c = self._soft_target(target, self.coarse_xy,
                                     self.coarse_sigma, self.log_coarse_mask)
        ce_f = -(soft_f * F.log_softmax(fine_l, dim=-1)).sum(-1).mean()
        ce_m = -(soft_m * F.log_softmax(medium_l, dim=-1)).sum(-1).mean()
        ce_c = -(soft_c * F.log_softmax(coarse_l, dim=-1)).sum(-1).mean()
        refined = self._gated_fine_prob(fine_l, medium_l, coarse_l)
        exp_xy = refined @ self.fine_xy
        mse = F.smooth_l1_loss(exp_xy, target, beta=1.0)
        return (self.w_fine * ce_f + self.w_medium * ce_m
                + self.w_coarse * ce_c + self.w_mse * mse)

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        fine_l, medium_l, coarse_l = self.forward(bssid_idx, rssi, mask)
        refined = self._gated_fine_prob(fine_l, medium_l, coarse_l)
        return (refined @ self.fine_xy).cpu().numpy()


class SetTransformerHeatmapMapCascade(nn.Module):
    """A + B combined: Floor-plan CNN cross-attention encoder + cascade heads.

    Encoder identical to SetTransformerHeatmapMap (Set Transformer pooled
    vector → cross-attention with CNN-encoded floor plan tokens → refined
    vector that "knows the geometry").

    Output side identical to SetTransformerHeatmapCascade (coarse + fine
    heatmap heads with cascade gating at inference).

    If A and B both improve over plain Heatmap, this combo should compound.
    """

    def __init__(self, num_bssids,
                 fine_cell_xy, fine_free_mask,
                 coarse_cell_xy, coarse_free_mask,
                 fine_to_coarse, map_pgm_path,
                 fine_sigma=0.4, coarse_sigma=1.0,
                 ce_fine_w=0.5, ce_coarse_w=0.3, mse_w=0.2,
                 embed_dim=48, model_dim=192, num_heads=4,
                 num_sab=3, out_dim=2, dropout=0.3,
                 map_target_hw=(56, 64)):
        super().__init__()
        # set encoder
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        # map encoder + cross-attn (from A)
        self.map_encoder = MapEncoder(map_pgm_path,
                                        target_hw=map_target_hw,
                                        model_dim=model_dim)
        self.cross_attn = MAB(model_dim, model_dim, model_dim,
                                num_heads, ln=True)
        self.dropout = nn.Dropout(dropout)
        # two heads + buffers (from B)
        Gf = int(fine_cell_xy.shape[0])
        Gc = int(coarse_cell_xy.shape[0])
        self.Gf = Gf; self.Gc = Gc
        self.fine_head = nn.Linear(model_dim, Gf)
        self.coarse_head = nn.Linear(model_dim, Gc)
        self.fine_sigma = float(fine_sigma)
        self.coarse_sigma = float(coarse_sigma)
        self.ce_fine_w = float(ce_fine_w)
        self.ce_coarse_w = float(ce_coarse_w)
        self.mse_w = float(mse_w)
        self.register_buffer('fine_xy',
                              torch.from_numpy(fine_cell_xy.astype(np.float32)))
        self.register_buffer('coarse_xy',
                              torch.from_numpy(coarse_cell_xy.astype(np.float32)))
        self.register_buffer('fine_mask',
                              torch.from_numpy(fine_free_mask.astype(np.float32)))
        self.register_buffer('coarse_mask',
                              torch.from_numpy(coarse_free_mask.astype(np.float32)))
        f_log = np.where(fine_free_mask, 0.0, -1e9).astype(np.float32)
        c_log = np.where(coarse_free_mask, 0.0, -1e9).astype(np.float32)
        self.register_buffer('log_fine_mask', torch.from_numpy(f_log))
        self.register_buffer('log_coarse_mask', torch.from_numpy(c_log))
        self.register_buffer('fine_to_coarse',
                              torch.from_numpy(fine_to_coarse.astype(np.int64)))

    def encode(self, bssid_idx, rssi, mask):
        emb = self.embed(bssid_idx)
        x = torch.cat([emb, rssi.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        # cross-attention to map
        B = pooled.size(0)
        map_tokens = self.map_encoder().expand(B, -1, -1)
        q = pooled.unsqueeze(1)
        refined = self.cross_attn(q, map_tokens).squeeze(1)
        return self.dropout(refined)

    def forward(self, bssid_idx, rssi, mask):
        h = self.encode(bssid_idx, rssi, mask)
        return self.fine_head(h), self.coarse_head(h)

    def _soft_target(self, target, cell_xy, sigma, log_mask):
        d2 = ((target.unsqueeze(1) - cell_xy.unsqueeze(0)) ** 2).sum(-1)
        log_p = -d2 / (2.0 * sigma ** 2) + log_mask.unsqueeze(0)
        return F.softmax(log_p, dim=-1)

    def _gated_fine_prob(self, fine_logits, coarse_logits):
        fine_p = F.softmax(fine_logits + self.log_fine_mask.unsqueeze(0),
                            dim=-1)
        coarse_p = F.softmax(coarse_logits + self.log_coarse_mask.unsqueeze(0),
                              dim=-1)
        gate = coarse_p[:, self.fine_to_coarse]
        refined = fine_p * gate
        refined = refined / refined.sum(-1, keepdim=True).clamp(min=1e-12)
        return refined

    def loss(self, out, target):
        fine_logits, coarse_logits = out
        soft_f = self._soft_target(target, self.fine_xy,
                                     self.fine_sigma, self.log_fine_mask)
        soft_c = self._soft_target(target, self.coarse_xy,
                                     self.coarse_sigma, self.log_coarse_mask)
        log_f = F.log_softmax(fine_logits, dim=-1)
        log_c = F.log_softmax(coarse_logits, dim=-1)
        ce_f = -(soft_f * log_f).sum(-1).mean()
        ce_c = -(soft_c * log_c).sum(-1).mean()
        refined = self._gated_fine_prob(fine_logits, coarse_logits)
        exp_xy = refined @ self.fine_xy
        mse = F.smooth_l1_loss(exp_xy, target, beta=1.0)
        return self.ce_fine_w * ce_f + self.ce_coarse_w * ce_c + self.mse_w * mse

    @torch.no_grad()
    def predict_xy(self, bssid_idx, rssi, mask):
        fine_logits, coarse_logits = self.forward(bssid_idx, rssi, mask)
        refined = self._gated_fine_prob(fine_logits, coarse_logits)
        return (refined @ self.fine_xy).cpu().numpy()


class SetTransformerMDNv2(nn.Module):
    """Enhanced Set Transformer MDN with multi-channel per-AP features.

    Per-AP input = cat(BSSID_embedding[D], extra_features[n_features])
    where extra_features comes from build_set_input_v2:
      [rssi_norm, rank_norm, rel_rssi_norm]  (n_features=3 by default)

    Otherwise identical to SetTransformerMDN.
    """

    def __init__(self, num_bssids, n_features=3, embed_dim=32, model_dim=128,
                  num_heads=4, num_sab=2, K=3, out_dim=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(num_bssids + 1, embed_dim,
                                    padding_idx=num_bssids)
        self.input_proj = nn.Linear(embed_dim + n_features, model_dim)
        self.encoder = nn.ModuleList([
            SAB(model_dim, num_heads, ln=True) for _ in range(num_sab)
        ])
        self.pool = PMA(model_dim, num_heads, num_seeds=1, ln=True)
        self.dropout = nn.Dropout(dropout)
        self.K = K
        self.out_dim = out_dim
        self.n_features = n_features
        self.pi_head = nn.Linear(model_dim, K)
        self.mu_head = nn.Linear(model_dim, K * out_dim)
        self.logsig_head = nn.Linear(model_dim, K * out_dim)

    def encode(self, bssid_idx, features, mask):
        emb = self.embed(bssid_idx)                            # (B, M, embed_dim)
        x = torch.cat([emb, features], dim=-1)                  # (B, M, embed+n_feat)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        for sab in self.encoder:
            x = sab(x, key_mask=mask)
        pooled = self.pool(x, key_mask=mask).squeeze(1)
        return self.dropout(pooled)

    def forward(self, bssid_idx, features, mask):
        h = self.encode(bssid_idx, features, mask)
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, self.out_dim)
        log_sigma = self.logsig_head(h).view(-1, self.K, self.out_dim)
        log_sigma = log_sigma.clamp(min=-3.0, max=2.5)
        return log_pi, mu, log_sigma

    def loss(self, out, target):
        return MDNRegressor.loss(self, out, target)

    @torch.no_grad()
    def predict_xy(self, bssid_idx, features, mask, mode='map'):
        log_pi, mu, log_sigma = self.forward(bssid_idx, features, mask)
        if mode == 'map':
            k = log_pi.argmax(dim=-1)
            return mu[torch.arange(mu.size(0)), k].cpu().numpy()
        pi = log_pi.exp().unsqueeze(-1)
        return (pi * mu).sum(1).cpu().numpy()

    @torch.no_grad()
    def predict_distribution(self, bssid_idx, features, mask):
        log_pi, mu, log_sigma = self.forward(bssid_idx, features, mask)
        return (log_pi.exp().cpu().numpy(),
                mu.cpu().numpy(),
                log_sigma.exp().cpu().numpy())
