#!/usr/bin/env python3
"""Reviewer-oriented MVGC reproducibility runner.

This script does not replace the full PyTorch-Geometric implementation in ``mvcg.py``.
It provides the deterministic experimental controls requested during review:
exact splits, extraction audits, extractor precision/recall hooks, text-only and
late-fusion controls, calibration metrics, and HRKG feature-noise stress tests.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy import sparse
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CANONICAL_QUALIFIER_MAP = {
    "urgent": "urgency",
    "threat": "coercive_pressure",
    "police": "institutional_pretext",
    "court": "institutional_pretext",
    "financial_authority": "institutional_pretext",
    "transfer": "requested_action_type",
    "account": "transfer_channel",
    "bankbook": "transfer_channel",
    "loan": "requested_action_type",
    "personal_info": "requested_action_type",
    "security": "requested_action_type",
    "confirmation": "temporal_pressure",
    "investigation": "institutional_pretext",
    "crime": "coercive_pressure",
    "identity_theft": "coercive_pressure",
    "virus": "coercive_pressure",
    "extortion": "coercive_pressure",
    "technical": "coercive_pressure",
    "family": "coercive_pressure",
    "delivery": "requested_action_type",
    "reward": "requested_action_type",
    "support": "requested_action_type",
    "medical": "coercive_pressure",
    "large_amount": "requested_action_type",
    "fraud": "coercive_pressure",
}

@dataclass
class ExtractedHRKG:
    entity_types: List[str]
    relation_types: List[str]
    qualifier_types: List[str]
    evidence: Dict[str, List[str]]


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_relation_patterns(raw_patterns: Sequence) -> List[str]:
    terms: List[str] = []
    for item in raw_patterns:
        if isinstance(item, str):
            terms.append(item)
        elif isinstance(item, Sequence):
            terms.extend(str(x) for x in item)
    return sorted(set(t for t in terms if t), key=len, reverse=True)


def compile_term_pattern(terms: Iterable[str]) -> re.Pattern | None:
    unique = sorted({str(t) for t in terms if str(t).strip()}, key=len, reverse=True)
    if not unique:
        return None
    # Limit each type-specific regex to the longest evidence phrases first. This keeps
    # review-protocol audits fast while preserving the most informative cues.
    unique = unique[:80]
    return re.compile("|".join(re.escape(t) for t in unique))


def prepare_patterns(ner_cfg: Dict, qual_cfg: Dict) -> Dict[str, Dict[str, re.Pattern]]:
    entity = {k: compile_term_pattern(v) for k, v in ner_cfg.get("entities", {}).items()}
    relation = {k: compile_term_pattern(flatten_relation_patterns(v)) for k, v in ner_cfg.get("relations", {}).items()}
    qualifier = {k: compile_term_pattern(v) for k, v in qual_cfg.items()}
    return {
        "entity": {k: v for k, v in entity.items() if v is not None},
        "relation": {k: v for k, v in relation.items() if v is not None},
        "qualifier": {k: v for k, v in qualifier.items() if v is not None},
    }


def regex_hits(pattern: re.Pattern, text: str, limit: int = 5) -> List[str]:
    hits = []
    for m in pattern.finditer(text):
        token = m.group(0)
        if token not in hits:
            hits.append(token)
        if len(hits) >= limit:
            break
    return hits


def extract_hrkg_types(text: str, patterns: Dict[str, Dict[str, re.Pattern]]) -> ExtractedHRKG:
    text = normalize_text(text)
    entity_types: List[str] = []
    relation_types: List[str] = []
    qualifier_types: List[str] = []
    evidence: Dict[str, List[str]] = {}

    for etype, pattern in patterns.get("entity", {}).items():
        hits = regex_hits(pattern, text)
        if hits:
            entity_types.append(etype)
            evidence[f"entity:{etype}"] = hits

    for rtype, pattern in patterns.get("relation", {}).items():
        hits = regex_hits(pattern, text)
        if hits:
            relation_types.append(rtype)
            evidence[f"relation:{rtype}"] = hits

    q_accum: Dict[str, List[str]] = {}
    for raw_q, pattern in patterns.get("qualifier", {}).items():
        canonical = CANONICAL_QUALIFIER_MAP.get(raw_q, raw_q)
        hits = regex_hits(pattern, text)
        if hits:
            q_accum.setdefault(canonical, []).extend(hits)
    for qtype, hits in q_accum.items():
        qualifier_types.append(qtype)
        evidence[f"qualifier:{qtype}"] = sorted(set(hits), key=hits.index)[:8]

    return ExtractedHRKG(
        entity_types=sorted(set(entity_types)),
        relation_types=sorted(set(relation_types)),
        qualifier_types=sorted(set(qualifier_types)),
        evidence=evidence,
    )


def build_feature_table(df: pd.DataFrame, ner_cfg: Dict, qual_cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audits = []
    patterns = prepare_patterns(ner_cfg, qual_cfg)
    for _, row in df.iterrows():
        sample_id = row["id"]
        extracted = extract_hrkg_types(row["transcript"], patterns)
        feat = {"id": sample_id}
        for etype in extracted.entity_types:
            feat[f"entity__{etype}"] = 1
        for rtype in extracted.relation_types:
            feat[f"relation__{rtype}"] = 1
        for qtype in extracted.qualifier_types:
            feat[f"qualifier__{qtype}"] = 1
        feat["graph__num_entity_types"] = len(extracted.entity_types)
        feat["graph__num_relation_types"] = len(extracted.relation_types)
        feat["graph__num_qualifier_types"] = len(extracted.qualifier_types)
        feat["graph__num_total_types"] = len(extracted.entity_types) + len(extracted.relation_types) + len(extracted.qualifier_types)
        rows.append(feat)
        audits.append({
            "id": sample_id,
            "label": row["label"],
            "entity_types": ";".join(extracted.entity_types),
            "relation_types": ";".join(extracted.relation_types),
            "qualifier_types": ";".join(extracted.qualifier_types),
            "evidence_json": json.dumps(extracted.evidence, ensure_ascii=False),
        })
    features = pd.DataFrame(rows).fillna(0)
    for col in features.columns:
        if col != "id":
            features[col] = features[col].astype(float)
    audit = pd.DataFrame(audits)
    return features, audit


def deterministic_split(df: pd.DataFrame, seed: int, split_cfg: Dict) -> Dict[str, List]:
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("validation", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train + validation + test ratios must sum to 1.0")
    stratify = df["label"] if split_cfg.get("stratify", True) else None
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed, stratify=stratify)
    temp_stratify = temp_df["label"] if split_cfg.get("stratify", True) else None
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=seed, stratify=temp_stratify)
    return {
        "train": train_df["id"].tolist(),
        "validation": val_df["id"].tolist(),
        "test": test_df["id"].tolist(),
    }


def ids_to_frame(df: pd.DataFrame, ids: Sequence) -> pd.DataFrame:
    idset = set(ids)
    return df[df["id"].isin(idset)].copy()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, ece_bins: int = 10) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ece": expected_calibration_error(y_true, y_prob, bins=ece_bins),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def make_text_vectorizer(cfg: Dict):
    ngram = tuple(cfg.get("ngram_range", [1, 2]))
    if cfg.get("vectorizer", "hashing") == "tfidf":
        return TfidfVectorizer(
            analyzer=cfg.get("analyzer", "word"),
            ngram_range=ngram,
            min_df=int(cfg.get("min_df", 2)),
            max_features=int(cfg.get("max_features", 30000)),
        )
    return HashingVectorizer(
        analyzer=cfg.get("analyzer", "word"),
        ngram_range=ngram,
        n_features=int(cfg.get("n_features", 32768)),
        alternate_sign=bool(cfg.get("alternate_sign", False)),
        norm=cfg.get("norm", "l2"),
    )


def make_text_classifier(cfg: Dict):
    # SGD with log loss is substantially faster than full logistic regression for
    # long Korean call transcripts and still provides probabilities for calibration.
    return SGDClassifier(
        loss="log_loss",
        penalty=cfg.get("penalty", "l2"),
        alpha=float(cfg.get("alpha", 1e-5)),
        class_weight=cfg.get("class_weight", "balanced"),
        max_iter=int(cfg.get("max_iter", 50)),
        tol=float(cfg.get("tol", 1e-3)),
        random_state=int(cfg.get("random_state", 13)),
    )


def make_text_pipeline(cfg: Dict) -> Pipeline:
    return Pipeline([
        ("vec", make_text_vectorizer(cfg)),
        ("clf", make_text_classifier(cfg)),
    ])



def prepare_text_series(series: pd.Series, cfg: Dict) -> pd.Series:
    max_chars = int(cfg.get("max_chars", 6000))
    return series.astype(str).str.slice(0, max_chars)


def fit_predict_text(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict) -> np.ndarray:
    pipe = make_text_pipeline(cfg)
    pipe.fit(prepare_text_series(train_df["transcript"], cfg), train_df["label"])
    return pipe.predict_proba(prepare_text_series(test_df["transcript"], cfg))[:, 1]


def graph_matrix(features: pd.DataFrame, ids: Sequence, columns: Sequence[str]) -> np.ndarray:
    sub = features[features["id"].isin(set(ids))].set_index("id").loc[list(ids)]
    return sub[list(columns)].to_numpy(dtype=float)


def fit_predict_graph(train_df: pd.DataFrame, test_df: pd.DataFrame, features: pd.DataFrame, cfg: Dict) -> np.ndarray:
    cols = [c for c in features.columns if c != "id"]
    X_train = graph_matrix(features, train_df["id"].tolist(), cols)
    X_test = graph_matrix(features, test_df["id"].tolist(), cols)
    clf = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(class_weight=cfg.get("class_weight", "balanced"), max_iter=int(cfg.get("max_iter", 2000)), solver="liblinear")),
    ])
    clf.fit(X_train, train_df["label"])
    return clf.predict_proba(X_test)[:, 1]


def fit_predict_late_fusion(train_df: pd.DataFrame, test_df: pd.DataFrame, features: pd.DataFrame, text_cfg: Dict, fusion_cfg: Dict) -> np.ndarray:
    """Decision-level late fusion of an independently trained text model and HRKG-feature model.

    This control is intentionally simple: it tests whether MVGC's joint multi-view
    learning improves over a non-contrastive text+graph combination.
    """
    text_prob = fit_predict_text(train_df, test_df, text_cfg)
    graph_prob = fit_predict_graph(train_df, test_df, features, fusion_cfg)
    weight_text = float(fusion_cfg.get("weight_text", 0.5))
    weight_text = min(1.0, max(0.0, weight_text))
    return weight_text * text_prob + (1.0 - weight_text) * graph_prob


def apply_feature_noise(features: pd.DataFrame, group: str, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noisy = features.copy()
    prefix = f"{group}__"
    cols = [c for c in noisy.columns if c.startswith(prefix)]
    if not cols or rate <= 0:
        return noisy
    mask = rng.random((len(noisy), len(cols))) < rate
    arr = noisy[cols].to_numpy(dtype=float).copy()
    arr[mask] = 0.0
    noisy.loc[:, cols] = arr
    return noisy


def evaluate_extractor(df: pd.DataFrame, ner_cfg: Dict, qual_cfg: Dict, gold_path: Path, out_dir: Path) -> None:
    patterns = prepare_patterns(ner_cfg, qual_cfg)
    if not gold_path or not gold_path.exists():
        template = out_dir / "hrkg_gold_template.jsonl"
        sample = df.head(min(25, len(df)))
        with template.open("w", encoding="utf-8") as f:
            for _, row in sample.iterrows():
                rec = {"id": row["id"], "entities": [], "relations": [], "qualifiers": []}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raise FileNotFoundError(f"Gold annotation file not found. Template written to {template}")

    gold = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gold[rec["id"]] = rec
    rows = []
    for _, row in df[df["id"].isin(gold.keys())].iterrows():
        pred = extract_hrkg_types(row["transcript"], patterns)
        for field, pred_values in [
            ("entities", pred.entity_types),
            ("relations", pred.relation_types),
            ("qualifiers", pred.qualifier_types),
        ]:
            p = set(pred_values)
            g = set(gold[row["id"]].get(field, []))
            tp = len(p & g)
            fp = len(p - g)
            fn = len(g - p)
            rows.append({"id": row["id"], "field": field, "tp": tp, "fp": fp, "fn": fn})
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "extractor_eval_detail.csv", index=False)
    summary = []
    for field, sub in detail.groupby("field"):
        tp, fp, fn = sub[["tp", "fp", "fn"]].sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        summary.append({"field": field, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    pd.DataFrame(summary).to_csv(out_dir / "extractor_eval_summary.csv", index=False)


def run_protocol(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(Path(args.config)) if args.config else {}
    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    required = {"id", "transcript", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[["id", "transcript", "label"]].copy()
    df["transcript"] = df["transcript"].map(normalize_text)
    df["label"] = df["label"].astype(int)

    cfg_dir = Path(args.config_dir)
    ner_cfg = load_json(cfg_dir / "ner_relations.json")
    qual_cfg = load_json(cfg_dir / "qualifiers.json")
    features, audit = build_feature_table(df, ner_cfg, qual_cfg)
    print(f"Built HRKG feature matrix: {features.shape}", flush=True)
    audit.to_csv(out_dir / "extraction_audit.csv", index=False)
    features.to_csv(out_dir / "hrkg_feature_matrix.csv", index=False)

    if args.mode == "extractor_eval":
        evaluate_extractor(df, ner_cfg, qual_cfg, Path(args.gold_annotations) if args.gold_annotations else None, out_dir)
        return

    seeds = args.seeds if args.seeds else config.get("seeds", [13, 17, 23])
    all_metrics: List[Dict] = []
    split_cfg = config.get("split", {"train": 0.70, "validation": 0.15, "test": 0.15, "stratify": True})
    ece_bins = int(config.get("calibration", {}).get("ece_bins", 10))
    text_cfg = config.get("text_baseline", {})
    graph_cfg = config.get("graph_feature_baseline", {})
    fusion_cfg = config.get("late_fusion", {})

    for seed in seeds:
        print(f"Running seed {seed}", flush=True)
        split = deterministic_split(df, int(seed), split_cfg)
        seed_dir = out_dir / f"split_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "split_ids.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        train_df = ids_to_frame(df, split["train"])
        val_df = ids_to_frame(df, split["validation"])
        test_df = ids_to_frame(df, split["test"])
        # Fit controls on train+validation for the final test report, while keeping validation IDs saved.
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        models = {}
        if args.mode in {"all", "baselines"}:
            print("  fitting text-only control", flush=True)
            models["text_only_tfidf_lr"] = fit_predict_text(trainval_df, test_df, text_cfg)
            print("  fitting HRKG-feature control", flush=True)
            models["hrkg_feature_lr"] = fit_predict_graph(trainval_df, test_df, features, graph_cfg)
            print("  fitting late-fusion control", flush=True)
            models["text_hrkg_late_fusion_lr"] = fit_predict_late_fusion(trainval_df, test_df, features, text_cfg, fusion_cfg)
        elif args.mode == "noise":
            # noise mode still trains the base graph and late-fusion controls below
            pass
        seed_metrics = []
        for name, prob in models.items():
            metrics = metric_dict(test_df["label"].to_numpy(), prob, ece_bins=ece_bins)
            metrics.update({"seed": seed, "model": name, "noise_group": "none", "noise_rate": 0.0})
            seed_metrics.append(metrics)
            pred = test_df[["id", "label"]].copy()
            pred["prob_phishing"] = prob
            pred["pred_label"] = (prob >= 0.5).astype(int)
            pred.to_csv(seed_dir / f"predictions_{name}.csv", index=False)

        if args.mode in {"all", "noise"}:
            noise_cfg = config.get("noise", {"rates": [0.0, 0.1, 0.2], "groups": ["entity", "relation", "qualifier"]})
            for group in noise_cfg.get("groups", ["entity", "relation", "qualifier"]):
                for rate in noise_cfg.get("rates", [0.0, 0.1, 0.2]):
                    noisy_features = apply_feature_noise(features, group, float(rate), int(seed) + int(float(rate) * 1000))
                    prob = fit_predict_late_fusion(trainval_df, test_df, noisy_features, text_cfg, fusion_cfg)
                    metrics = metric_dict(test_df["label"].to_numpy(), prob, ece_bins=ece_bins)
                    metrics.update({"seed": seed, "model": "text_hrkg_late_fusion_lr", "noise_group": group, "noise_rate": float(rate)})
                    seed_metrics.append(metrics)
        pd.DataFrame(seed_metrics).to_csv(seed_dir / "metrics.csv", index=False)
        print(f"  wrote metrics for seed {seed}", flush=True)
        all_metrics.extend(seed_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "all_metrics.csv", index=False)
    if not metrics_df.empty:
        summary = metrics_df.groupby(["model", "noise_group", "noise_rate"]).agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"), precision_std=("precision", "std"),
            recall_mean=("recall", "mean"), recall_std=("recall", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            ece_mean=("ece", "mean"), ece_std=("ece", "std"),
            roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        ).reset_index()
        summary.to_csv(out_dir / "summary_metrics.csv", index=False)
        noise = metrics_df[metrics_df["noise_group"] != "none"]
        if not noise.empty:
            noise.to_csv(out_dir / "noise_robustness.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "config": args.config,
        "num_samples": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "mode": args.mode,
        "seeds": [int(s) for s in seeds],
        "notes": "Review protocol controls; full neural MVGC implementation remains in mvcg.py.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVGC controlled protocol runner")
    parser.add_argument("--data", default="dataset/voicephishing_data.csv", help="CSV with id, transcript, label")
    parser.add_argument("--output_dir", default="outputs/reviewer_protocol", help="Output directory")
    parser.add_argument("--mode", choices=["all", "baselines", "noise", "extractor_eval"], default="all")
    parser.add_argument("--config", default="config/reviewer_protocol.json", help="Controlled protocol JSON")
    parser.add_argument("--config_dir", default="config", help="Directory containing ner_relations.json and qualifiers.json")
    parser.add_argument("--gold_annotations", default="", help="Optional JSONL gold HRKG annotations for extractor precision/recall")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_protocol(args)


if __name__ == "__main__":
    main()
