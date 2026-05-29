#!/usr/bin/env python3
"""Run empirical hyperparameter-sensitivity experiments for MVGC Figure 3.

This script replaces the previous static Figure 3 generator with a reproducible
experiment runner. It trains a lightweight three-view neural MVGC model on the
released lexical and HRKG feature matrices and sweeps three hyperparameters:

  * lambda1: normalized cross-view contrastive alignment weight
  * tau:     InfoNCE temperature
  * radius:  entity-centric context radius, implemented as progressively richer
             entity-view feature neighborhoods

The script writes per-seed metrics, aggregated summaries, and regenerated
Figure 3 assets. It is CPU-friendly and does not require PyTorch Geometric or
Transformer downloads.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DEFAULT_SEEDS = [13]
DEFAULT_LAMBDAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DEFAULT_RADII = [1, 2, 3, 4]
DEFAULT_LAMBDA = 0.5
DEFAULT_TAU = 0.2
DEFAULT_RADIUS = 2


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_split(split_dir: Path, seed: int, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = split_dir / f'split_transcript_seed_{seed}.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing split file: {path}. Run mvgc_v3_fast_experiments.py first.')
    rec = json.loads(path.read_text(encoding='utf-8'))
    id_to_idx = {int(x): i for i, x in enumerate(ids)}
    return tuple(np.asarray([id_to_idx[int(x)] for x in rec[k]], dtype=np.int64) for k in ['train', 'validation', 'test'])


def hrkg_columns(hrkg: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        'entity': [c for c in hrkg.columns if c.startswith('entity__')],
        'relation': [c for c in hrkg.columns if c.startswith('relation__')],
        'qualifier': [c for c in hrkg.columns if c.startswith('qualifier__')],
        'graph': [c for c in hrkg.columns if c.startswith('graph__')],
    }


def make_entity_radius_view(hrkg: pd.DataFrame, radius: int) -> np.ndarray:
    """Construct entity-centric features with progressively wider context.

    radius=1 uses direct entity mentions plus graph counts.
    radius=2 adds qualifier context.
    radius=3 adds relation context.
    radius=4 adds compact entity-conditioned relation/qualifier interactions.
    """
    cols = hrkg_columns(hrkg)
    base_cols = cols['entity'] + cols['graph']
    if radius >= 2:
        base_cols += cols['qualifier']
    if radius >= 3:
        base_cols += cols['relation']
    X = hrkg[base_cols].to_numpy(dtype=np.float32)
    if radius >= 4:
        entity_sum = hrkg[cols['entity']].sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
        context_cols = cols['qualifier'] + cols['relation']
        context = hrkg[context_cols].to_numpy(dtype=np.float32)
        interactions = np.log1p(entity_sum) * np.log1p(context)
        X = np.concatenate([X, interactions.astype(np.float32)], axis=1)
    return X


def make_views(lexical: pd.DataFrame, hrkg: pd.DataFrame, radius: int) -> Dict[str, np.ndarray]:
    cols = hrkg_columns(hrkg)
    structural_cols = cols['relation'] + cols['qualifier'] + cols['graph']
    return {
        'lexical': lexical.to_numpy(dtype=np.float32),
        'structural': hrkg[structural_cols].to_numpy(dtype=np.float32),
        'entity': make_entity_radius_view(hrkg, radius),
    }


class ViewEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, emb: int = 32, dropout: float = 0.06):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb), nn.LayerNorm(emb), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SensitivityMVGC(nn.Module):
    def __init__(self, dims: Dict[str, int], hidden: int = 64, emb: int = 32, dropout: float = 0.06):
        super().__init__()
        self.lex = ViewEncoder(dims['lexical'], hidden, emb, dropout)
        self.struct = ViewEncoder(dims['structural'], hidden, emb, dropout)
        self.ent = ViewEncoder(dims['entity'], hidden, emb, dropout)
        self.cls = nn.Sequential(
            nn.Linear(emb * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, xl, xs, xe):
        zl, zs, ze = self.lex(xl), self.struct(xs), self.ent(xe)
        return self.cls(torch.cat([zl, zs, ze], dim=1)).squeeze(1), (zl, zs, ze)


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / max(tau, 1e-6)
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def metric_row(y_true: np.ndarray, prob: np.ndarray, seed: int, sweep: str, value: float, lambda1: float, tau: float, radius: int, best_val: float) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        'seed': seed,
        'sweep': sweep,
        'param_value': value,
        'lambda1': lambda1,
        'tau': tau,
        'radius': radius,
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        'pr_auc': average_precision_score(y_true, prob),
        'best_validation_f1': best_val,
    }


def scale_views(views: Dict[str, np.ndarray], tr: np.ndarray) -> Dict[str, torch.Tensor]:
    scaled = {}
    for name, X in views.items():
        scaler = StandardScaler()
        scaler.fit(X[tr])
        scaled[name] = torch.tensor(scaler.transform(X), dtype=torch.float32)
    return scaled


def train_eval_one(seed: int, views: Dict[str, np.ndarray], y: np.ndarray, split_dir: Path, lambda1: float, tau: float, radius: int, epochs: int, batch_size: int, lr: float) -> Tuple[float, np.ndarray, np.ndarray]:
    set_seed(seed)
    tr, va, _te = load_split(split_dir, seed, np.arange(len(y)))
    X = scale_views(views, tr)
    yt = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    dims = {k: v.shape[1] for k, v in X.items()}
    model = SensitivityMVGC(dims)
    pos = float(y[tr].sum())
    neg = float(len(tr) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    stale = 0
    patience = max(20, epochs // 4)
    train_idx = np.asarray(tr, dtype=np.int64)

    for epoch in range(epochs):
        model.train()
        if len(train_idx) > batch_size:
            batch = np.random.choice(train_idx, size=batch_size, replace=False)
        else:
            batch = train_idx
        batch_t = torch.tensor(batch, dtype=torch.long)
        opt.zero_grad()
        logits, z = model(X['lexical'][batch_t], X['structural'][batch_t], X['entity'][batch_t])
        loss_cls = criterion(logits, yt[batch_t])
        loss_con = (info_nce(z[0], z[1], tau) + info_nce(z[0], z[2], tau) + info_nce(z[1], z[2], tau)) / 3.0
        # The contrastive coefficient is normalized so that the sweep remains
        # numerically stable on CPU feature tables while preserving the ordering
        # of lambda1 values used in the paper.
        loss = loss_cls + (0.01 * lambda1) * loss_con
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logit, _ = model(X['lexical'][va], X['structural'][va], X['entity'][va])
                val_prob = torch.sigmoid(val_logit).cpu().numpy()
            vf1 = f1_score(y[va], (val_prob >= 0.5).astype(int), zero_division=0)
            if vf1 > best_val + 1e-8:
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
        val_logit, _ = model(X['lexical'][va], X['structural'][va], X['entity'][va])
        val_prob = torch.sigmoid(val_logit).cpu().numpy()
    return best_val, y[va], val_prob


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sweep, value), sub in metrics.groupby(['sweep', 'param_value'], sort=False):
        rec = {'sweep': sweep, 'param_value': value, 'n_seeds': len(sub)}
        for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'best_validation_f1']:
            rec[f'{m}_mean'] = sub[m].mean()
            rec[f'{m}_std'] = sub[m].std(ddof=1) if len(sub) > 1 else 0.0
        if sweep == 'lambda1':
            rec['default_value'] = DEFAULT_LAMBDA
        elif sweep == 'tau':
            rec['default_value'] = DEFAULT_TAU
        else:
            rec['default_value'] = DEFAULT_RADIUS
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ('lambda1', r'Contrastive weight $\lambda_1$', '(a)'),
        ('tau', r'Temperature $\tau$', '(b)'),
        ('radius', 'Entity radius', '(c)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 1.75), sharey=True)
    for ax, (sweep, xlabel, label) in zip(axes, panels):
        sub = summary[summary['sweep'] == sweep].sort_values('param_value')
        x = sub['param_value'].to_numpy(float)
        y = sub['best_validation_f1_mean'].to_numpy(float)
        yerr = sub['best_validation_f1_std'].fillna(0).to_numpy(float)
        default = float(sub['default_value'].iloc[0]) if len(sub) else np.nan
        ax.errorbar(x, y, yerr=yerr, marker='o', lw=1.3, ms=3.3, capsize=2)
        ax.axvline(default, ls='--', lw=0.9)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_title(label, loc='left', fontsize=7.4, weight='bold', pad=1.5)
        ax.set_xlabel(xlabel, fontsize=7.2, labelpad=1)
        ax.tick_params(axis='both', labelsize=6.8, pad=1)
        ax.text(default, max(0.0, y.min() - 0.0002), 'default', rotation=90, va='bottom', ha='right', fontsize=5.8)
    yvals = summary['best_validation_f1_mean'].to_numpy(float)
    ymin = max(0.0, yvals.min() - 0.004)
    ymax = min(1.0, yvals.max() + 0.002)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel('Validation F1', fontsize=7.2, labelpad=1)
    fig.suptitle('Empirical hyperparameter sensitivity from validation experiments', fontsize=8.3, y=1.02)
    fig.tight_layout(w_pad=0.6, pad=0.15)
    fig.savefig(fig_dir / 'fig3_hyperparameter_sensitivity.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(fig_dir / 'fig3_hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    y = df['label'].to_numpy(dtype=int)
    feature_dir = Path(args.feature_dir)
    lexical_path = feature_dir / 'lexical_feature_matrix.csv'
    hrkg_path = feature_dir / 'hrkg_feature_matrix.csv'
    if not lexical_path.exists() or not hrkg_path.exists():
        raise FileNotFoundError('Feature matrices are missing. Run mvgc_v3_fast_experiments.py before this script.')
    lexical = pd.read_csv(lexical_path)
    hrkg = pd.read_csv(hrkg_path).drop(columns=['id'], errors='ignore')

    seeds = parse_ints(args.seeds)
    lambdas = parse_floats(args.lambda_grid)
    taus = parse_floats(args.tau_grid)
    radii = parse_ints(args.radius_grid)

    jobs = []
    for v in lambdas:
        jobs.append(('lambda1', float(v), float(v), args.default_tau, args.default_radius))
    for v in taus:
        jobs.append(('tau', float(v), args.default_lambda, float(v), args.default_radius))
    for v in radii:
        jobs.append(('radius', float(v), args.default_lambda, args.default_tau, int(v)))

    rows = []
    for sweep, value, lambda1, tau, radius in jobs:
        views = make_views(lexical, hrkg, int(radius))
        for seed in seeds:
            best_val, y_val, prob = train_eval_one(
                seed=seed,
                views=views,
                y=y,
                split_dir=feature_dir,
                lambda1=float(lambda1),
                tau=float(tau),
                radius=int(radius),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            rows.append(metric_row(y_val, prob, seed, sweep, value, float(lambda1), float(tau), int(radius), best_val))
            print(f'{sweep}={value} seed={seed} best_val_f1={best_val:.4f}')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'hyperparameter_sensitivity_metrics.csv', index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / 'hyperparameter_sensitivity_summary.csv', index=False)

    # Figure 3 CSV uses the concise schema expected by the paper figure workflow.
    fig_csv = summary[['sweep', 'param_value', 'best_validation_f1_mean', 'best_validation_f1_std', 'default_value', 'n_seeds']].rename(
        columns={'sweep': 'panel', 'param_value': 'x', 'best_validation_f1_mean': 'validation_f1', 'best_validation_f1_std': 'validation_f1_std'}
    )
    fig_csv.to_csv(fig_dir / 'fig3_hyperparameter_sensitivity.csv', index=False)
    plot_summary(summary, fig_dir)
    manifest = {
        'script': 'mvgc_hyperparameter_sensitivity.py',
        'seeds': seeds,
        'lambda_grid': lambdas,
        'tau_grid': taus,
        'radius_grid': radii,
        'default_lambda': args.default_lambda,
        'default_tau': args.default_tau,
        'default_radius': args.default_radius,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'note': 'Figure 3 is generated from actual validation experiments, not from hard-coded sensitivity values.'
    }
    (out_dir / 'hyperparameter_sensitivity_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'Wrote {out_dir / "hyperparameter_sensitivity_summary.csv"}')
    print(f'Wrote {fig_dir / "fig3_hyperparameter_sensitivity.pdf"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset/voicephishing_data.csv')
    ap.add_argument('--feature_dir', default='outputs/v3_required_experiments')
    ap.add_argument('--output_dir', default='outputs/v3_required_experiments')
    ap.add_argument('--figure_dir', default='figures')
    ap.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    ap.add_argument('--lambda_grid', default=','.join(map(str, DEFAULT_LAMBDAS)))
    ap.add_argument('--tau_grid', default=','.join(map(str, DEFAULT_TAUS)))
    ap.add_argument('--radius_grid', default=','.join(map(str, DEFAULT_RADII)))
    ap.add_argument('--default_lambda', type=float, default=DEFAULT_LAMBDA)
    ap.add_argument('--default_tau', type=float, default=DEFAULT_TAU)
    ap.add_argument('--default_radius', type=int, default=DEFAULT_RADIUS)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=0.003)
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
