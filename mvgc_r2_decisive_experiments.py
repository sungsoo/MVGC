#!/usr/bin/env python3
"""Decisive R1 experiments addressing Reviewer #1/#2/#3 binding comments.

This runner produces, under one shared and reproducible CPU protocol:

  (E1) A separately tuned text-only lexical baseline, separately tuned
       single-view (structural, entity, HRKG) baselines, and separately
       tuned two-view baselines, each with its own light hyperparameter
       search on validation (Reviewer #3 Comment 3).

  (E2) The full multi-view model trained WITH the cross-view contrastive
       alignment (lambda>0) and a matched CONTRASTIVE-OFF variant
       (lambda=0) under an otherwise identical training and evaluation
       protocol (Reviewer #3 Comment 2; "contrastive" attribution).

  (E3) Multi-seed statistics and PAIRED significance tests (paired t-test
       and Wilcoxon signed-rank over seeds) for the two key comparisons:
         - text-only vs full multi-view, and
         - best non-contrastive fusion vs full multi-view
       (Reviewer #3 Comment 4; Reviewer #1 Comment 2).

  (E4) A stricter, length/source-stratified OUT-OF-DISTRIBUTION split:
       short transcripts (which are disproportionately phishing) are
       quarantined to the test fold so the model cannot exploit the
       transcript-length shortcut. Reported alongside the in-distribution
       result (Reviewer #3 Comment 5; Reviewer #1 Comment 3).

All numbers are written to outputs/required_experiments/r2_*.csv so the
manuscript macros and tables can be regenerated from raw features.

Designed to complete well within an 8-hour single-workstation budget; on a
MacBook Pro M3 the full run finishes in a few minutes on CPU.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler

SEEDS = [13, 17, 23, 29, 31]
FEATURE_DIR = Path('outputs/required_experiments')
OUT_DIR = Path('outputs/required_experiments')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Feature assembly                                                            #
# --------------------------------------------------------------------------- #
def make_views(lexical: pd.DataFrame, hrkg: pd.DataFrame) -> Dict[str, np.ndarray]:
    entity_cols = [c for c in hrkg.columns if c.startswith('entity__')]
    relation_cols = [c for c in hrkg.columns if c.startswith('relation__')]
    qual_cols = [c for c in hrkg.columns if c.startswith('qualifier__')]
    count_cols = [c for c in hrkg.columns if c.startswith('graph__')]
    return {
        'lexical': lexical.to_numpy(dtype=np.float32),
        'structural': hrkg[relation_cols + qual_cols + count_cols].to_numpy(dtype=np.float32),
        'entity': hrkg[entity_cols + count_cols].to_numpy(dtype=np.float32),
        'hrkg': hrkg[entity_cols + relation_cols + qual_cols + count_cols].to_numpy(dtype=np.float32),
    }


def metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float('nan'),
        'pr_auc': average_precision_score(y_true, prob),
    }


# --------------------------------------------------------------------------- #
# Encoders / model                                                            #
# --------------------------------------------------------------------------- #
class ViewEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb), nn.LayerNorm(emb), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiViewModel(nn.Module):
    """Multi-view model over an arbitrary subset of views with optional
    cross-view contrastive alignment. Setting use_views to a single view
    yields a tuned single-view baseline; lambda_con=0 yields the
    contrastive-off variant under the identical architecture."""

    def __init__(self, dims: Dict[str, int], use_views: List[str],
                 hidden: int, emb: int, dropout: float):
        super().__init__()
        self.use_views = use_views
        self.encoders = nn.ModuleDict({v: ViewEncoder(dims[v], hidden, emb, dropout)
                                       for v in use_views})
        self.cls = nn.Sequential(
            nn.Linear(emb * len(use_views), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        zs = [self.encoders[v](batch[v]) for v in self.use_views]
        logit = self.cls(torch.cat(zs, dim=1)).squeeze(1)
        return logit, zs


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def train_eval(seed: int, views: Dict[str, np.ndarray], y: np.ndarray,
               train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
               use_views: List[str], lambda_con: float,
               hp: Dict[str, float]) -> Dict[str, float]:
    set_seed(seed)
    trv = np.r_[train_idx, val_idx]
    scaled: Dict[str, torch.Tensor] = {}
    for name in use_views:
        X = views[name]
        scaler = StandardScaler().fit(X[trv])
        scaled[name] = torch.tensor(scaler.transform(X), dtype=torch.float32)

    yt = torch.tensor(y.astype(np.float32))
    dims = {k: views[k].shape[1] for k in use_views}
    model = MultiViewModel(dims, use_views, int(hp['hidden']), int(hp['emb']), hp['dropout'])
    pos = float(y[train_idx].sum())
    neg = float(len(train_idx) - pos)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

    def slice_batch(idx):
        return {v: scaled[v][idx] for v in use_views}

    best_state, best_val, patience, stale = None, -1.0, 60, 0
    n_tr = len(train_idx)
    cbs = 256  # contrastive mini-batch size to bound the NxN similarity matrix
    rng = np.random.RandomState(seed)
    for epoch in range(260):
        model.train()
        opt.zero_grad()
        logits, zs = model(slice_batch(train_idx))
        loss = bce(logits, yt[train_idx])
        if lambda_con > 0 and len(zs) >= 2:
            sel = rng.choice(n_tr, size=min(cbs, n_tr), replace=False)
            zsel = [z[sel] for z in zs]
            cl, cnt = 0.0, 0
            for i in range(len(zsel)):
                for j in range(i + 1, len(zsel)):
                    cl = cl + info_nce(zsel[i], zsel[j], hp['temperature'])
                    cnt += 1
            loss = loss + lambda_con * (cl / max(cnt, 1))
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == 259:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(slice_batch(val_idx))[0]).numpy()
            vf1 = f1_score(y[val_idx], (vp >= 0.5).astype(int), zero_division=0)
            if vf1 > best_val + 1e-6:
                best_val, stale = vf1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 5
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(slice_batch(test_idx))[0]).numpy()
    out = metrics(y[test_idx], prob)
    out['best_validation_f1'] = best_val
    out['_prob'] = prob
    out['_test_idx'] = test_idx
    return out


# --------------------------------------------------------------------------- #
# Splits                                                                       #
# --------------------------------------------------------------------------- #
def load_transcript_split(seed: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = json.loads((FEATURE_DIR / f'split_transcript_seed_{seed}.json').read_text())
    id_to_idx = {int(x): i for i, x in enumerate(range(n))}
    # split files store row ids 0..n-1
    return (np.asarray(rec['train'], dtype=np.int64),
            np.asarray(rec['validation'], dtype=np.int64),
            np.asarray(rec['test'], dtype=np.int64))


def make_length_ood_split(lengths: np.ndarray, y: np.ndarray, seed: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stricter OOD split that quarantines the transcript-length shortcut.

    Phishing calls in KorCCVi v2 are markedly shorter than benign calls.
    A model can therefore exploit length as a near-deterministic shortcut
    under a random split. Here the SHORTEST 25% of transcripts are forced
    into the test fold and the LONGEST transcripts dominate training, so
    the length distribution differs systematically between train and test.
    This is a deliberately adversarial generalization probe rather than a
    deployment estimate."""
    rng = np.random.RandomState(seed)
    n = len(lengths)
    order = np.argsort(lengths)              # short -> long
    short_cut = int(0.25 * n)
    test_pool = order[:short_cut]            # shortest quartile -> OOD test
    rest = order[short_cut:]
    rng.shuffle(rest)
    n_val = int(0.15 * len(rest))
    val_idx = rest[:n_val]
    train_idx = rest[n_val:]
    # guarantee both classes appear in test; if not, top up from rest tail
    test_idx = test_pool
    if len(np.unique(y[test_idx])) < 2:
        extra = rest[-int(0.1 * len(rest)):]
        test_idx = np.r_[test_idx, extra]
        train_idx = np.setdiff1d(train_idx, extra)
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Light hyperparameter search (separately tuned baselines)                     #
# --------------------------------------------------------------------------- #
HP_GRID = [
    {'hidden': 64, 'emb': 32, 'dropout': 0.05, 'lr': 0.003, 'wd': 1e-4, 'temperature': 0.15},
    {'hidden': 96, 'emb': 48, 'dropout': 0.10, 'lr': 0.002, 'wd': 1e-4, 'temperature': 0.20},
]


def tune_and_run(name: str, use_views: List[str], lambda_con: float,
                 views, y, split_fn, do_tune: bool = True) -> pd.DataFrame:
    """For each seed: pick HP by validation F1 (averaged over a single seed
    proxy when do_tune), then evaluate on test. Records per-seed test rows."""
    rows = []
    for seed in SEEDS:
        tr, va, te = split_fn(seed)
        if do_tune and len(HP_GRID) > 1:
            best_r, best_vf1 = None, -1.0
            for hp in HP_GRID:
                r = train_eval(seed, views, y, tr, va, te, use_views, lambda_con, hp)
                if r['best_validation_f1'] > best_vf1:
                    best_vf1, best_r = r['best_validation_f1'], r
            r = best_r
        else:
            r = train_eval(seed, views, y, tr, va, te, use_views, lambda_con, HP_GRID[0])
        rows.append({'model': name, 'seed': seed, 'lambda_con': lambda_con,
                     'views': '+'.join(use_views),
                     **{k: v for k, v in r.items() if not k.startswith('_')}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group=('model', 'views', 'lambda_con')) -> pd.DataFrame:
    cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    recs = []
    for keys, g in df.groupby(list(group)):
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec['n_seeds'] = len(g)
        for c in cols:
            rec[f'{c}_mean'] = g[c].mean()
            rec[f'{c}_std'] = g[c].std(ddof=1)
        recs.append(rec)
    return pd.DataFrame(recs)


def paired_tests(a: pd.DataFrame, b: pd.DataFrame, metric='f1') -> Dict[str, float]:
    a = a.sort_values('seed'); b = b.sort_values('seed')
    x = a[metric].to_numpy(); z = b[metric].to_numpy()
    diff = z - x
    out = {'mean_a': float(x.mean()), 'mean_b': float(z.mean()),
           'mean_diff': float(diff.mean()), 'std_diff': float(diff.std(ddof=1))}
    if np.allclose(diff, 0):
        out.update({'t_stat': 0.0, 't_p': 1.0, 'wilcoxon_p': 1.0, 'cohens_dz': 0.0})
        return out
    t = stats.ttest_rel(z, x)
    out['t_stat'] = float(t.statistic); out['t_p'] = float(t.pvalue)
    try:
        w = stats.wilcoxon(z, x)
        out['wilcoxon_p'] = float(w.pvalue)
    except ValueError:
        out['wilcoxon_p'] = float('nan')
    out['cohens_dz'] = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return out


# --------------------------------------------------------------------------- #
def main():
    import argparse
    global FEATURE_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description='MVGC decisive experiments (Tables 1-3): '
                                 'separately tuned baselines, contrastive-off ablation, '
                                 'paired significance tests, length-stratified OOD split, '
                                 'and near-duplicate audit.')
    ap.add_argument('--data', default='dataset/voicephishing_data.csv',
                    help='raw transcript CSV (columns: id, transcript, label)')
    ap.add_argument('--feature_dir', default='outputs/required_experiments',
                    help='directory containing lexical/HRKG feature tables and split files '
                         '(produced by mvgc_v3_fast_experiments.py)')
    ap.add_argument('--output_dir', default='outputs/required_experiments',
                    help='directory to write r2_*.csv result files')
    args = ap.parse_args()
    FEATURE_DIR = Path(args.feature_dir)
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    n = len(df)
    y = df['label'].to_numpy(dtype=int)
    lengths = df['transcript'].astype(str).str.len().to_numpy()
    lexical = pd.read_csv(FEATURE_DIR / 'lexical_feature_matrix.csv')
    hrkg = pd.read_csv(FEATURE_DIR / 'hrkg_feature_matrix.csv').drop(columns=['id'], errors='ignore')
    views = make_views(lexical, hrkg)

    id_split = lambda seed: load_transcript_split(seed, n)
    ood_split = lambda seed: make_length_ood_split(lengths, y, seed)

    print('=== In-distribution protocol (separately tuned baselines + contrastive on/off) ===')
    configs = [
        ('text_only_lexical',   ['lexical'],                         0.0),
        ('structural_only',     ['structural'],                      0.0),
        ('entity_only',         ['entity'],                          0.0),
        ('hrkg_only',           ['hrkg'],                            0.0),
        ('lexical+structural',  ['lexical', 'structural'],           0.0),
        ('lexical+hrkg',        ['lexical', 'hrkg'],                 0.0),
        ('full_contrastive_off',['lexical', 'structural', 'entity'], 0.0),
        ('full_contrastive_on', ['lexical', 'structural', 'entity'], 0.5),
    ]
    all_rows = []
    for name, vs, lam in configs:
        ckpt = OUT_DIR / f'r2_indist_part_{name}.csv'
        if ckpt.exists():
            d = pd.read_csv(ckpt)
        else:
            d = tune_and_run(name, vs, lam, views, y, id_split, do_tune=True)
            d.to_csv(ckpt, index=False)
        all_rows.append(d)
        print(f"  {name:24s} F1={d['f1'].mean():.4f}±{d['f1'].std(ddof=1):.4f}", flush=True)
    id_df = pd.concat(all_rows, ignore_index=True)
    id_df.to_csv(OUT_DIR / 'r2_indist_perseed.csv', index=False)
    summarize(id_df).to_csv(OUT_DIR / 'r2_indist_summary.csv', index=False)

    print('\n=== Paired significance tests (in-distribution, over 5 seeds) ===')
    def rows_for(name):
        return id_df[id_df['model'] == name]
    sig = {}
    sig['text_only_vs_full_on'] = paired_tests(rows_for('text_only_lexical'),
                                               rows_for('full_contrastive_on'))
    sig['full_off_vs_full_on'] = paired_tests(rows_for('full_contrastive_off'),
                                              rows_for('full_contrastive_on'))
    sig['best_fusion_vs_full_on'] = paired_tests(rows_for('lexical+hrkg'),
                                                 rows_for('full_contrastive_on'))
    sig_df = pd.DataFrame([{'comparison': k, **v} for k, v in sig.items()])
    sig_df.to_csv(OUT_DIR / 'r2_significance.csv', index=False)
    for k, v in sig.items():
        print(f"  {k:26s} dF1={v['mean_diff']:+.4f}  t_p={v['t_p']:.3f}  "
              f"wilcoxon_p={v['wilcoxon_p']:.3f}  dz={v['cohens_dz']:+.2f}")

    print('\n=== Stricter length-stratified OOD protocol ===')
    ood_configs = [
        ('text_only_lexical',    ['lexical'],                         0.0),
        ('hrkg_only',            ['hrkg'],                            0.0),
        ('lexical+hrkg',         ['lexical', 'hrkg'],                 0.0),
        ('full_contrastive_off', ['lexical', 'structural', 'entity'], 0.0),
        ('full_contrastive_on',  ['lexical', 'structural', 'entity'], 0.5),
    ]
    ood_rows = []
    for name, vs, lam in ood_configs:
        ckpt = OUT_DIR / f'r2_ood_part_{name}.csv'
        if ckpt.exists():
            d = pd.read_csv(ckpt)
        else:
            d = tune_and_run(name + '_ood', vs, lam, views, y, ood_split, do_tune=True)
            d.to_csv(ckpt, index=False)
        ood_rows.append(d)
        print(f"  {name:24s} F1={d['f1'].mean():.4f}±{d['f1'].std(ddof=1):.4f}  "
              f"AUROC={d['roc_auc'].mean():.4f}", flush=True)
    ood_df = pd.concat(ood_rows, ignore_index=True)
    ood_df.to_csv(OUT_DIR / 'r2_ood_perseed.csv', index=False)
    summarize(ood_df).to_csv(OUT_DIR / 'r2_ood_summary.csv', index=False)

    ood_sig = paired_tests(ood_df[ood_df['model'] == 'text_only_lexical_ood'],
                           ood_df[ood_df['model'] == 'full_contrastive_on_ood'])
    pd.DataFrame([{'comparison': 'ood_text_only_vs_full_on', **ood_sig}]).to_csv(
        OUT_DIR / 'r2_ood_significance.csv', index=False)
    print(f"  OOD text-only vs full: dF1={ood_sig['mean_diff']:+.4f} "
          f"t_p={ood_sig['t_p']:.3f} wilcoxon_p={ood_sig['wilcoxon_p']:.3f}")

    # near-duplicate / paraphrase-cluster diagnostic on lengths + lexical bins
    print('\n=== Near-duplicate diagnostic ===')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = df['transcript'].astype(str).tolist()
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    # block-wise max off-diagonal cosine to flag near-duplicates (memory safe)
    thr = 0.9
    n_dup = 0
    bs = 400
    for i in range(0, n, bs):
        sims = cosine_similarity(X[i:i+bs], X)
        for r in range(sims.shape[0]):
            gi = i + r
            sims[r, gi] = 0.0
            if sims[r].max() >= thr:
                n_dup += 1
    dup_frac = n_dup / n
    pd.DataFrame([{'cosine_threshold': thr, 'n_near_duplicate': n_dup,
                   'fraction': dup_frac, 'n_total': n}]).to_csv(
        OUT_DIR / 'r2_near_duplicate.csv', index=False)
    print(f"  near-duplicate (TF-IDF cos>={thr}): {n_dup}/{n} = {dup_frac:.3f}")
    print('\nDone. Wrote r2_*.csv to', OUT_DIR)


if __name__ == '__main__':
    main()
