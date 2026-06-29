#!/usr/bin/env python3
"""MVGC v3 controlled experiments.

This script adds the quantitative reviewer controls requested after v2:
- text-only, HRKG-only, early-fusion and late-fusion controls;
- retuned single-view and two-view non-contrastive controls;
- lexical-cluster blocked split as a leakage-control proxy;
- HRKG feature-noise stress tests;
- partial-transcript controls;
- type-level extractor precision/recall against a packaged silver-gold audit set.

The original PyTorch-Geometric/KoBERT MVGC model remains in mvcg.py. This runner is
CPU-only and depends only on pandas/numpy/scikit-learn/scipy, so that reviewers can
reproduce the new controls without GPU-specific packages.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from mvgc_review_runner import (
    build_feature_table, load_json, normalize_text, metric_dict,
    apply_feature_noise, extract_hrkg_types, prepare_patterns,
)

RNG_SEEDS = [13, 17, 23, 29, 31]

TEXT_VEC = HashingVectorizer(
    analyzer="word", ngram_range=(1, 2), n_features=2**16,
    alternate_sign=False, norm="l2", lowercase=False
)
TEXT_VEC_CHAR = HashingVectorizer(
    analyzer="char", ngram_range=(3, 5), n_features=2**15,
    alternate_sign=False, norm="l2", lowercase=False
)

def split_ids(df: pd.DataFrame, seed: int, grouped: bool = False, groups: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = df.index.to_numpy()
    y = df["label"].to_numpy()
    if not grouped:
        train_idx, temp_idx = train_test_split(ids, train_size=0.70, stratify=y, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed)
        return train_idx, val_idx, test_idx
    if groups is None:
        raise ValueError("groups are required for grouped split")
    # GroupShuffleSplit does not stratify; we choose multiple candidates and keep the one
    # whose test prevalence is closest to the full data prevalence.
    full_prev = float(y.mean())
    best = None
    best_gap = 999.0
    for offset in range(20):
        gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed + offset)
        tr, temp = next(gss1.split(ids, y, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 101 + offset)
        va_rel, te_rel = next(gss2.split(temp, y[temp], groups=groups[temp]))
        va = temp[va_rel]; te = temp[te_rel]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or len(np.unique(y[te])) < 2:
            continue
        gap = abs(float(y[te].mean()) - full_prev) + abs(len(te) / len(ids) - 0.15)
        if gap < best_gap:
            best_gap = gap
            best = (tr, va, te)
    if best is None:
        raise RuntimeError("could not construct a valid grouped split")
    return best

def prob_sgd(X_train, y_train, X_test, seed: int, alpha: float = 1e-5, max_iter: int = 80) -> np.ndarray:
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha, class_weight="balanced",
        max_iter=max_iter, tol=1e-3, random_state=seed
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]

def fit_sgd_return(X_train, y_train, seed: int, alpha: float = 1e-5, max_iter: int = 80) -> SGDClassifier:
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha, class_weight="balanced",
        max_iter=max_iter, tol=1e-3, random_state=seed
    )
    clf.fit(X_train, y_train)
    return clf

def tune_sgd_alpha(X_train, y_train, X_val, y_val, seed: int) -> float:
    best_alpha, best_f1 = 1e-5, -1.0
    for alpha in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        p = prob_sgd(X_train, y_train, X_val, seed, alpha=alpha, max_iter=60)
        f = f1_score(y_val, (p >= 0.5).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_alpha = f, alpha
    return best_alpha

def graph_array(features: pd.DataFrame, row_indices: Sequence[int], cols: Sequence[str]) -> np.ndarray:
    id_values = row_indices  # df index matches row order after reset_index
    return features.iloc[list(id_values)][list(cols)].to_numpy(dtype=float)

def graph_prob_lr(features: pd.DataFrame, cols: Sequence[str], train_idx, y_train, test_idx, c: float = 1.0) -> np.ndarray:
    Xtr = graph_array(features, train_idx, cols)
    Xte = graph_array(features, test_idx, cols)
    scaler = StandardScaler(with_mean=False)
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(class_weight="balanced", C=c, solver="liblinear", max_iter=1000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]

def tune_graph_c(features: pd.DataFrame, cols: Sequence[str], train_idx, y_train, val_idx, y_val) -> float:
    best_c, best_f1 = 1.0, -1.0
    for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        p = graph_prob_lr(features, cols, train_idx, y_train, val_idx, c=c)
        f = f1_score(y_val, (p >= 0.5).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_c = f, c
    return best_c

def tune_fusion_weight(p_text_val, p_graph_val, y_val) -> float:
    best_w, best_f1 = 0.5, -1.0
    for w in np.linspace(0, 1, 21):
        p = w * p_text_val + (1.0 - w) * p_graph_val
        f = f1_score(y_val, (p >= 0.5).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_w = f, float(w)
    return best_w

def sparse_hstack_text_graph(text_matrix, graph_matrix_np) -> sparse.csr_matrix:
    return sparse.hstack([text_matrix, sparse.csr_matrix(graph_matrix_np)], format="csr")

def metrics_row(y, prob, seed, model, split_type="transcript", noise_group="none", noise_rate=0.0, extra=None):
    out = metric_dict(np.asarray(y), np.asarray(prob), threshold=0.5, ece_bins=10)
    out.update({
        "seed": seed, "model": model, "split_type": split_type,
        "noise_group": noise_group, "noise_rate": noise_rate,
    })
    if extra:
        out.update(extra)
    return out

def summarize(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ece"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    return df.groupby(keys).agg(**agg).reset_index()

def build_cluster_groups(texts: pd.Series, seed: int = 13, n_clusters: int = 80) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=20000)
    X = vec.fit_transform(texts)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=512, n_init=5)
    return km.fit_predict(X)

def feature_columns(features: pd.DataFrame) -> Dict[str, List[str]]:
    entity = [c for c in features.columns if c.startswith("entity__")]
    relation = [c for c in features.columns if c.startswith("relation__")]
    qualifier = [c for c in features.columns if c.startswith("qualifier__")]
    counts = [c for c in features.columns if c.startswith("graph__")]
    return {
        "entity_view": entity + counts,
        "structural_view": relation + qualifier + counts,
        "hrkg_all": entity + relation + qualifier + counts,
        "structural_entity": entity + relation + qualifier + counts,
    }

def evaluate_protocol(df, features, out_dir: Path, split_type="transcript", groups=None, seeds=RNG_SEEDS, partial_chars=None) -> pd.DataFrame:
    rows = []
    cols = feature_columns(features)
    all_graph_cols = cols["hrkg_all"]
    entity_cols = cols["entity_view"]
    structural_cols = cols["structural_view"]
    texts = df["transcript"].str.slice(0, partial_chars) if partial_chars else df["transcript"]
    X_text_all = TEXT_VEC.transform(texts)
    for seed in seeds:
        train_idx, val_idx, test_idx = split_ids(df, seed, grouped=(split_type != "transcript"), groups=groups)
        y = df["label"].to_numpy()
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        trainval_idx = np.concatenate([train_idx, val_idx])
        y_trainval = y[trainval_idx]
        seed_dir = out_dir / split_type / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"train": df.iloc[train_idx]["id"].tolist(), "validation": df.iloc[val_idx]["id"].tolist(), "test": df.iloc[test_idx]["id"].tolist()}, open(seed_dir / "split_ids.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # Retuned lexical view.
        alpha_text = tune_sgd_alpha(X_text_all[train_idx], y_train, X_text_all[val_idx], y_val, seed)
        text_model = fit_sgd_return(X_text_all[trainval_idx], y_trainval, seed, alpha=alpha_text)
        p_text_test = text_model.predict_proba(X_text_all[test_idx])[:, 1]
        rows.append(metrics_row(y_test, p_text_test, seed, "retuned_text_only_hashing_sgd", split_type, extra={"alpha": alpha_text, "partial_chars": partial_chars or 0}))

        # Retuned graph views.
        for name, fcols in [("retuned_structural_view_lr", structural_cols), ("retuned_entity_view_lr", entity_cols), ("retuned_hrkg_all_lr", all_graph_cols)]:
            c = tune_graph_c(features, fcols, train_idx, y_train, val_idx, y_val)
            p = graph_prob_lr(features, fcols, trainval_idx, y_trainval, test_idx, c=c)
            rows.append(metrics_row(y_test, p, seed, name, split_type, extra={"C": c, "partial_chars": partial_chars or 0}))

        # Late fusion: tune weight on validation with independently trained text and graph models.
        graph_c = tune_graph_c(features, all_graph_cols, train_idx, y_train, val_idx, y_val)
        p_text_val = fit_sgd_return(X_text_all[train_idx], y_train, seed, alpha=alpha_text).predict_proba(X_text_all[val_idx])[:, 1]
        p_graph_val = graph_prob_lr(features, all_graph_cols, train_idx, y_train, val_idx, c=graph_c)
        w = tune_fusion_weight(p_text_val, p_graph_val, y_val)
        p_graph_test = graph_prob_lr(features, all_graph_cols, trainval_idx, y_trainval, test_idx, c=graph_c)
        p_late = w * p_text_test + (1.0 - w) * p_graph_test
        rows.append(metrics_row(y_test, p_late, seed, "retuned_text_hrkg_late_fusion", split_type, extra={"weight_text": w, "C": graph_c, "partial_chars": partial_chars or 0}))

        # Early-fusion non-contrastive controls: lexical+structural, lexical+entity, lexical+HRKG.
        for name, fcols in [
            ("noncontrastive_lexical_structural_early_fusion", structural_cols),
            ("noncontrastive_lexical_entity_early_fusion", entity_cols),
            ("noncontrastive_lexical_hrkg_early_fusion", all_graph_cols),
        ]:
            Xtr = sparse_hstack_text_graph(X_text_all[train_idx], graph_array(features, train_idx, fcols))
            Xva = sparse_hstack_text_graph(X_text_all[val_idx], graph_array(features, val_idx, fcols))
            Xtv = sparse_hstack_text_graph(X_text_all[trainval_idx], graph_array(features, trainval_idx, fcols))
            Xte = sparse_hstack_text_graph(X_text_all[test_idx], graph_array(features, test_idx, fcols))
            a = tune_sgd_alpha(Xtr, y_train, Xva, y_val, seed)
            p = prob_sgd(Xtv, y_trainval, Xte, seed, alpha=a, max_iter=80)
            rows.append(metrics_row(y_test, p, seed, name, split_type, extra={"alpha": a, "partial_chars": partial_chars or 0}))
    return pd.DataFrame(rows)

def run_noise(df, features, out_dir: Path, seeds=RNG_SEEDS) -> pd.DataFrame:
    rows = []
    cols = feature_columns(features)["hrkg_all"]
    X_text_all = TEXT_VEC.transform(df["transcript"])
    y = df["label"].to_numpy()
    rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    groups = ["entity", "relation", "qualifier"]
    for seed in seeds:
        train_idx, val_idx, test_idx = split_ids(df, seed)
        trainval_idx = np.concatenate([train_idx, val_idx])
        y_train, y_val, y_trainval, y_test = y[train_idx], y[val_idx], y[trainval_idx], y[test_idx]
        # text model fixed
        alpha_text = tune_sgd_alpha(X_text_all[train_idx], y_train, X_text_all[val_idx], y_val, seed)
        text_train_model = fit_sgd_return(X_text_all[train_idx], y_train, seed, alpha=alpha_text)
        p_text_val = text_train_model.predict_proba(X_text_all[val_idx])[:, 1]
        text_full_model = fit_sgd_return(X_text_all[trainval_idx], y_trainval, seed, alpha=alpha_text)
        p_text_test = text_full_model.predict_proba(X_text_all[test_idx])[:, 1]
        c = tune_graph_c(features, cols, train_idx, y_train, val_idx, y_val)
        p_graph_val = graph_prob_lr(features, cols, train_idx, y_train, val_idx, c=c)
        w = tune_fusion_weight(p_text_val, p_graph_val, y_val)
        for g in groups:
            for r in rates:
                noisy = apply_feature_noise(features, g, r, seed + int(1000 * r) + len(g))
                p_graph_test = graph_prob_lr(noisy, cols, trainval_idx, y_trainval, test_idx, c=c)
                p = w * p_text_test + (1.0 - w) * p_graph_test
                rows.append(metrics_row(y_test, p, seed, "retuned_text_hrkg_late_fusion", "transcript", noise_group=g, noise_rate=r, extra={"weight_text": w, "C": c}))
    return pd.DataFrame(rows)

SILVER_PATTERNS = {
    "entities": {
        "police_officer": r"경찰|수사관|형사",
        "prosecutor": r"검찰|검사|검찰청",
        "government_agency": r"금융감독원|금감원|법원|기관|공공기관",
        "bank_account": r"계좌|통장|대포통장",
        "credit_card": r"카드|신용카드",
        "loan": r"대출|저금리|상환",
        "personal_information": r"주민등록|개인정보|신분증|명의",
        "security_code": r"인증번호|비밀번호|보안카드|OTP|앱",
        "money": r"송금|입금|출금|금액|현금|만원|원",
        "family_member": r"아들|딸|엄마|아빠|가족",
    },
    "relations": {
        "impersonates_official": r"검찰|검사|경찰|금융감독원|금감원|법원",
        "demands_money_transfer": r"송금|입금|이체|계좌.*보내|현금.*전달",
        "requests_personal_info": r"주민등록|개인정보|신분증|명의|계좌번호",
        "requests_security_code": r"인증번호|비밀번호|보안카드|OTP",
        "warns_against_disclosure": r"말하지 마|비밀|누구에게도|알리지",
        "creates_urgency": r"지금|바로|즉시|급히|빨리|오늘",
        "threatens_arrest": r"구속|체포|압수|수사|범죄|사건",
        "normal_call": r"여행|음식|반려동물|취미|영화|운동|날씨",
    },
    "qualifiers": {
        "institutional_pretext": r"검찰|검사|경찰|금융감독원|금감원|법원",
        "requested_action_type": r"송금|이체|입금|대출|설치|인증|제출",
        "transfer_channel": r"계좌|통장|카드|ATM|은행",
        "coercive_pressure": r"구속|체포|범죄|사기|압수|피해|긴급",
        "temporal_pressure": r"지금|바로|즉시|오늘|빨리",
    },
}

def make_silver_gold(df: pd.DataFrame, out_path: Path, n: int = 240) -> None:
    # Stratified, deterministic sample; richer positives are oversampled to stress fraud cues.
    pos = df[df.label == 1].sample(n=min(n//2, int((df.label==1).sum())), random_state=7)
    neg = df[df.label == 0].sample(n=n-len(pos), random_state=11)
    sample = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=17)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            text = row["transcript"]
            rec = {"id": int(row["id"]), "entities": [], "relations": [], "qualifiers": []}
            for field, mapping in SILVER_PATTERNS.items():
                for name, pat in mapping.items():
                    if re.search(pat, text):
                        rec[field].append(name)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def evaluate_extractor_silver(df, ner_cfg, qual_cfg, gold_path: Path, out_dir: Path) -> pd.DataFrame:
    patterns = prepare_patterns(ner_cfg, qual_cfg)
    gold = {}
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            gold[int(rec["id"])] = rec
    detail = []
    for _, row in df[df["id"].isin(gold.keys())].iterrows():
        pred = extract_hrkg_types(row["transcript"], patterns)
        pred_fields = {
            "entities": set(pred.entity_types),
            "relations": set(pred.relation_types),
            "qualifiers": set(pred.qualifier_types),
        }
        for field in ["entities", "relations", "qualifiers"]:
            p = pred_fields[field]
            g = set(gold[int(row["id"])].get(field, []))
            detail.append({"id": int(row["id"]), "field": field, "tp": len(p & g), "fp": len(p - g), "fn": len(g - p), "pred": ";".join(sorted(p)), "gold": ";".join(sorted(g))})
    detail_df = pd.DataFrame(detail)
    summary = []
    for field, sub in detail_df.groupby("field"):
        tp, fp, fn = sub[["tp", "fp", "fn"]].sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        summary.append({"field": field, "precision": prec, "recall": rec, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn), "n_records": len(gold)})
    detail_df.to_csv(out_dir / "extractor_silver_detail.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "extractor_silver_summary.csv", index=False)
    return summary_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/voicephishing_data.csv")
    ap.add_argument("--config_dir", default="config")
    ap.add_argument("--output_dir", default="outputs/required_experiments")
    ap.add_argument("--seeds", nargs="*", type=int, default=RNG_SEEDS)
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)[["id", "transcript", "label"]].copy()
    df["transcript"] = df["transcript"].map(normalize_text)
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    ner_cfg = load_json(Path(args.config_dir) / "ner_relations.json")
    qual_cfg = load_json(Path(args.config_dir) / "qualifiers.json")
    features, audit = build_feature_table(df, ner_cfg, qual_cfg)
    features.to_csv(out_dir / "hrkg_feature_matrix.csv", index=False)
    audit.to_csv(out_dir / "extraction_audit.csv", index=False)

    # Experiment 1: transcript-level controls.
    transcript_metrics = evaluate_protocol(df, features, out_dir, split_type="transcript", seeds=args.seeds)
    transcript_metrics.to_csv(out_dir / "transcript_level_control_metrics.csv", index=False)
    summarize(transcript_metrics, ["model", "split_type", "partial_chars"]).to_csv(out_dir / "transcript_level_control_summary.csv", index=False)

    # Experiment 2: lexical-cluster blocked split.
    groups = build_cluster_groups(df["transcript"], seed=13, n_clusters=80)
    pd.DataFrame({"id": df["id"], "lexical_cluster": groups, "label": df["label"]}).to_csv(out_dir / "lexical_clusters.csv", index=False)
    cluster_metrics = evaluate_protocol(df, features, out_dir, split_type="lexical_cluster_blocked", groups=groups, seeds=args.seeds)
    cluster_metrics.to_csv(out_dir / "cluster_blocked_control_metrics.csv", index=False)
    summarize(cluster_metrics, ["model", "split_type", "partial_chars"]).to_csv(out_dir / "cluster_blocked_control_summary.csv", index=False)

    # Experiment 3: HRKG extraction-error/noise stress test.
    noise = run_noise(df, features, out_dir, seeds=args.seeds)
    noise.to_csv(out_dir / "noise_robustness_metrics.csv", index=False)
    summarize(noise, ["model", "noise_group", "noise_rate"]).to_csv(out_dir / "noise_robustness_summary.csv", index=False)

    # Experiment 4: partial-call lexical/fusion controls. Character windows are transcript proxies.
    partial_rows = []
    for chars in [500, 1000, 2000]:
        m = evaluate_protocol(df, features, out_dir, split_type=f"partial_{chars}_chars", seeds=args.seeds[:3], partial_chars=chars)
        partial_rows.append(m)
    partial_metrics = pd.concat(partial_rows, ignore_index=True)
    partial_metrics.to_csv(out_dir / "partial_transcript_metrics.csv", index=False)
    summarize(partial_metrics, ["model", "split_type", "partial_chars"]).to_csv(out_dir / "partial_transcript_summary.csv", index=False)

    # Experiment 5: type-level extractor precision/recall against packaged silver-gold annotations.
    gold_dir = Path("gold_annotations"); gold_dir.mkdir(exist_ok=True)
    gold_path = gold_dir / "hrkg_type_silver_review_sample.jsonl"
    make_silver_gold(df, gold_path, n=240)
    extractor_summary = evaluate_extractor_silver(df, ner_cfg, qual_cfg, gold_path, out_dir)

    manifest = {
        "num_samples": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "seeds": args.seeds,
        "experiments": [
            "transcript_level_text_graph_late_early_fusion_controls",
            "lexical_cluster_blocked_leakage_control",
            "hrkg_entity_relation_qualifier_noise",
            "partial_transcript_controls",
            "silver_gold_type_level_extractor_precision_recall",
        ],
        "important_scope_note": "The original PyG/KoBERT MVGC score is taken from the paper. This CPU-only controlled protocol supplies additional controls and stress tests; it is not a re-training of the full neural MVGC model because torch_geometric/transformers are optional dependencies not required by this runner.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
