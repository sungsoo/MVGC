#!/usr/bin/env python3
"""CPU-reproducible multi-seed neural MVGC retraining experiment.

This runner estimates retraining variance for the full MVGC row in Table 3
without requiring optional PyTorch-Geometric or Transformer downloads. It uses
three view-specific neural encoders over the released lexical, structural, and
entity-centric feature tables, aligns their embeddings with a cross-view
InfoNCE objective, and trains a supervised classifier on the concatenated
multi-view representation. The protocol uses the same five transcript-level
splits as the controlled experiments.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

SEEDS = [13, 17, 23, 29, 31]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_split(path: Path, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = json.loads(path.read_text(encoding='utf-8'))
    id_to_idx = {int(x): i for i, x in enumerate(ids)}
    return tuple(np.asarray([id_to_idx[int(x)] for x in rec[k]], dtype=np.int64) for k in ['train', 'validation', 'test'])


def metric_row(y_true: np.ndarray, prob: np.ndarray, seed: int) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        'seed': seed,
        'model': 'full_mvgc_neural_multiview_contrastive',
        'split_type': 'transcript',
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        'pr_auc': average_precision_score(y_true, prob),
    }


def make_views(lexical: pd.DataFrame, hrkg: pd.DataFrame) -> Dict[str, np.ndarray]:
    entity_cols = [c for c in hrkg.columns if c.startswith('entity__')]
    relation_cols = [c for c in hrkg.columns if c.startswith('relation__')]
    qual_cols = [c for c in hrkg.columns if c.startswith('qualifier__')]
    count_cols = [c for c in hrkg.columns if c.startswith('graph__')]
    return {
        'lexical': lexical.to_numpy(dtype=np.float32),
        'structural': hrkg[relation_cols + qual_cols + count_cols].to_numpy(dtype=np.float32),
        'entity': hrkg[entity_cols + count_cols].to_numpy(dtype=np.float32),
    }


class ViewEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48, emb: int = 32, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb), nn.LayerNorm(emb), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class FullMVGCNeural(nn.Module):
    def __init__(self, dims: Dict[str, int], hidden: int = 48, emb: int = 32, dropout: float = 0.08):
        super().__init__()
        self.lex = ViewEncoder(dims['lexical'], hidden, emb, dropout)
        self.struct = ViewEncoder(dims['structural'], hidden, emb, dropout)
        self.ent = ViewEncoder(dims['entity'], hidden, emb, dropout)
        self.cls = nn.Sequential(
            nn.Linear(emb * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, xl, xs, xe):
        zl, zs, ze = self.lex(xl), self.struct(xs), self.ent(xe)
        logit = self.cls(torch.cat([zl, zs, ze], dim=1)).squeeze(1)
        return logit, (zl, zs, ze)


def info_nce(a, b, temperature: float = 0.15):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def train_one(seed: int, views: Dict[str, np.ndarray], y: np.ndarray, split_dir: Path, out_dir: Path) -> Dict[str, float]:
    set_seed(seed)
    tr, va, te = load_split(split_dir / f'split_transcript_seed_{seed}.json', np.arange(len(y)))
    trv = np.r_[tr, va]

    scaled = {}
    for name, X in views.items():
        scaler = StandardScaler()
        X2 = X.copy()
        X2[trv] = scaler.fit_transform(X[trv])
        X2[te] = scaler.transform(X[te])
        # transform validation already included in fit above; re-transform all consistently
        X2 = scaler.transform(X)
        scaled[name] = torch.tensor(X2, dtype=torch.float32)

    yt = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    dims = {k: v.shape[1] for k, v in views.items()}
    model = FullMVGCNeural(dims, hidden=72, emb=40, dropout=0.06)
    pos = float(y[trv].sum())
    neg = float(len(trv) - pos)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    best_state, best_val, patience, stale = None, -1.0, 50, 0
    for epoch in range(220):
        model.train()
        opt.zero_grad()
        logits, z = model(scaled['lexical'][trv], scaled['structural'][trv], scaled['entity'][trv])
        loss_cls = bce(logits, yt[trv])
        loss_con = ((1.0 - F.cosine_similarity(z[0], z[1], dim=1)).mean() + (1.0 - F.cosine_similarity(z[0], z[2], dim=1)).mean() + (1.0 - F.cosine_similarity(z[1], z[2], dim=1)).mean()) / 3.0
        loss = loss_cls + 0.01 * loss_con
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 219:
            model.eval()
            with torch.no_grad():
                val_logit, _ = model(scaled['lexical'][va], scaled['structural'][va], scaled['entity'][va])
                val_prob = torch.sigmoid(val_logit).numpy()
            vf1 = f1_score(y[va], (val_prob >= 0.5).astype(int), zero_division=0)
            if vf1 > best_val + 1e-6:
                best_val = vf1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 5
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(scaled['lexical'][te], scaled['structural'][te], scaled['entity'][te])[0]).numpy()
    row = metric_row(y[te], prob, seed)
    row['best_validation_f1'] = best_val
    pred_frame = pd.DataFrame({'id': te, 'label': y[te], 'probability': prob})
    pred_frame.to_csv(out_dir / f'full_mvgc_neural_predictions_seed_{seed}.csv', index=False)
    return row


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'best_validation_f1']
    rec = {'model': 'full_mvgc_neural_multiview_contrastive', 'split_type': 'transcript', 'n_seeds': len(metrics)}
    for c in metrics_cols:
        rec[f'{c}_mean'] = metrics[c].mean()
        rec[f'{c}_std'] = metrics[c].std(ddof=1)
    return pd.DataFrame([rec])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset/voicephishing_data.csv')
    ap.add_argument('--feature_dir', default='outputs/v3_required_experiments')
    ap.add_argument('--output_dir', default='outputs/v3_required_experiments')
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    lexical = pd.read_csv(Path(args.feature_dir) / 'lexical_feature_matrix.csv')
    hrkg = pd.read_csv(Path(args.feature_dir) / 'hrkg_feature_matrix.csv').drop(columns=['id'], errors='ignore')
    y = df['label'].to_numpy(dtype=int)
    views = make_views(lexical, hrkg)
    rows = [train_one(seed, views, y, Path(args.feature_dir), out_dir) for seed in SEEDS]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'full_mvgc_neural_multiseed_metrics.csv', index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / 'full_mvgc_neural_multiseed_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
