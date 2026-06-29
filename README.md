# MVGC: Multi-View Hyper-Relational Knowledge Graph Contrastive Learning for Voice Phishing Detection

This repository contains the official implementation, dataset artifacts, and full
reproduction pipeline for the paper:

> **MVGC: Multi-View Hyper-Relational Knowledge Graph Contrastive Learning for Voice Phishing Detection.**

MVGC represents each Korean call transcript as a **qualifier-aware hyper-relational
knowledge graph (HRKG)** and jointly learns three complementary views — *structural*,
*lexical*, and *entity-centric* — that are aligned by a cross-view contrastive
objective and combined by attention-based fusion.

The repository is designed so that **every quantitative result in the paper can be
regenerated end-to-end from the raw transcript CSV on a single CPU workstation**, with
no GPU and no cluster required. A full clean-clone reproduction completes in a few
minutes on a MacBook Pro M3 (128 GB RAM) and comfortably within an 8-hour budget on
any modern laptop.

---

## 1. Headline finding (what the experiments show)

The evaluation is deliberately transparent and significance-tested. Its central result
is **conditional**, which we consider more informative than a single accuracy number:

- **In-distribution, the benchmark is saturated.** A separately tuned text-only encoder
  already reaches F1 = 0.9905, and the full multi-view model (F1 = 0.9924) is **not**
  statistically better (paired *t*-test *p* = 0.37). On the standard split, transcript
  text alone is near-sufficient, so no in-distribution comparison can separate
  hyper-relational structure from lexical shortcuts.
- **The cause is a transcript-length shortcut.** Phishing transcripts are markedly
  shorter than benign ones. A near-duplicate audit (TF–IDF cosine ≥ 0.9) flags only
  0.5% near-duplicates, so the leakage is the length shortcut, not paraphrase
  duplication.
- **Under a stricter length-stratified out-of-distribution (OOD) split, structure
  wins decisively.** The text-only model collapses to F1 = 0.4824, while HRKG-bearing
  models degrade gracefully (HRKG-only = 0.7018, full MVGC = 0.6354). The advantage is
  large and significant (ΔF1 = +0.15 to +0.22, *p* < 0.01).

In short: **the hyper-relational structure provides a separable, statistically
significant benefit precisely under distribution shift.** The contrastive objective
contributes a consistently positive but not-yet-significant increment, which we report
honestly rather than overclaim.

---

## 2. Repository layout

```
MVGC/
├── README.md                         # this file
├── requirements.txt                  # Python dependencies
├── dataset/
│   └── voicephishing_data.csv        # KorCCVi v2 transcripts (id, transcript, label)
├── config/                           # extraction schema and parameters
│   ├── ner_relations.json            # entity/relation lexicon (active schema)
│   ├── ner_relations_all.json        # full lexicon variant
│   ├── ner_relations_integrated.json # integrated lexicon variant
│   ├── qualifiers.json               # six canonical qualifier families
│   ├── vpd_params.json               # default model/experiment parameters
│   ├── vpd_params_ablation.json      # ablation parameters
│   └── reviewer_protocol.json        # protocol constants (seeds, splits)
├── gold_annotations/
│   └── hrkg_type_silver_review_sample.jsonl   # 240-record type-level silver audit set
├── figures/                          # paper figure assets (Fig.1 source; Fig.2/3)
│
├── mvgc_review_runner.py             # HRKG extraction + feature-table builder (library)
├── mvgc_voicephishing.py             # thin CLI entry point to the extraction runner
├── mvgc_v3_fast_experiments.py       # STAGE 1: features, splits, extractor/noise/
│                                     #          scenario/partial tables (CPU)
├── mvgc_r2_decisive_experiments.py   # STAGE 2: Tables 1–3 (separately tuned baselines,
│                                     #          contrastive-off, paired tests, OOD,
│                                     #          near-duplicate audit) (CPU)
├── mvgc_full_neural_multiseed.py     # OPTIONAL: full neural multi-seed retraining
├── mvgc_v3_reviewer_experiments.py   # OPTIONAL: extended reviewer-protocol experiments
├── generate_paper_figures_v3.py      # regenerates Figure 3 (sensitivity); preserves Fig.1
└── mvcg.py                           # OPTIONAL: original PyG/KoBERT graph-baseline
                                      #           implementation (produces Figure 2);
                                      #           requires heavyweight optional deps
```

**Two reproduction tracks.**

1. **CPU-reproducible track (default, required for all paper conclusions).** Stages 1–2
   plus the figure script. Uses only NumPy/pandas/scikit-learn/SciPy/PyTorch (CPU). This
   regenerates every table and figure that the paper's conclusions rest on.
2. **Optional heavyweight track.** `mvcg.py` is the original PyTorch-Geometric + KoBERT
   implementation used to produce the graph-centric baseline comparison (Figure 2). It
   requires `torch-geometric`, `transformers`, and `sentencepiece`, and benefits from a
   GPU. It is **not** needed to reproduce any conclusion in the paper.

---

## 3. Installation from scratch

### 3.1 Prerequisites

- Python **3.10–3.12** (3.12 recommended)
- `pip` and `venv`
- No GPU required for the CPU-reproducible track.

### 3.2 Clone and create a virtual environment

```bash
git clone https://github.com/sungsoo/MVGC.git
cd MVGC

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3.3 Install dependencies

**Minimal install (CPU-reproducible track — sufficient for all paper conclusions):**

```bash
pip install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "scipy>=1.10" \
            "matplotlib>=3.7" "torch>=2.2"
```

**Full install (everything, including the optional PyG/KoBERT track):**

```bash
pip install -r requirements.txt
```

> Note: On CPU-only machines, install the CPU build of PyTorch from
> <https://pytorch.org/get-started/locally/> if the default wheel is unavailable.

### 3.4 Verify the install

```bash
python -c "import numpy, pandas, sklearn, scipy, torch; print('OK', torch.__version__)"
```

---

## 4. End-to-end reproduction (CPU track)

All commands are run from the repository root with the virtual environment activated.
Every command writes to `outputs/required_experiments/`, which is created
automatically.

### Step 1 — Build features, splits, and the controlled tables

```bash
python mvgc_v3_fast_experiments.py --output_dir outputs/required_experiments
```

This single command, starting from `dataset/voicephishing_data.csv`:

- builds the **lexical** and **HRKG** feature tables
  (`lexical_feature_matrix.csv`, `hrkg_feature_matrix.csv`);
- writes the **five stratified transcript-level split files** (seeds 13, 17, 23, 29, 31)
  and the partial-transcript / scenario-blocked split files;
- runs the **type-level extractor audit** against the silver review sample
  (`extractor_silver_summary.csv`) — **Table 4** of the paper;
- runs the **HRKG feature-noise stress test** (`noise_robustness_summary.csv`);
- runs the **scenario-signature / cluster-blocked control**
  (`cluster_blocked_control_summary.csv`);
- runs the **partial-transcript control** (`partial_transcript_summary.csv`) — **Table 5**;
- writes a run `manifest.json`.

Runtime: ~2–4 minutes on CPU.

### Step 2 — Regenerate the headline tables (Tables 1–3)

```bash
python mvgc_r2_decisive_experiments.py \
    --feature_dir outputs/required_experiments \
    --output_dir  outputs/required_experiments
```

This reads the feature tables and splits from Step 1 and regenerates, end to end:

- **Table 1** — in-distribution controls, each baseline **separately tuned**
  (`r2_indist_summary.csv`, `r2_indist_perseed.csv`), including the **contrastive-off**
  variant (λ₁ = 0);
- **Table 2** — paired significance tests over the five shared seeds
  (`r2_significance.csv`);
- **Table 3** — the stricter **length-stratified OOD split**
  (`r2_ood_summary.csv`, `r2_ood_significance.csv`);
- the **near-duplicate audit** (`r2_near_duplicate.csv`).

The script prints a live summary and is **resumable**: per-configuration checkpoints
(`r2_indist_part_*.csv`, `r2_ood_part_*.csv`) are reused if present, so an interrupted
run continues where it stopped. Runtime: ~3–6 minutes on CPU.

### Step 3 — Regenerate Figure 3 (hyperparameter sensitivity)

```bash
python generate_paper_figures_v3.py
```

Writes `figures/fig3_hyperparameter_sensitivity.{pdf,png,csv}` and preserves the
architecture diagram (Figure 1) and the graph-centric comparison (Figure 2) assets.

### Step 4 (optional) — Full neural multi-seed retraining

```bash
python mvgc_full_neural_multiseed.py \
    --feature_dir outputs/required_experiments \
    --output_dir  outputs/required_experiments
```

Trains the three-view neural model with cross-view alignment over the five seeds and
writes `full_mvgc_neural_multiseed_summary.csv` plus per-seed predictions. This
corroborates the full-model row with an explicit neural training loop. Runtime:
~2–3 minutes on CPU.

---

## 5. Expected results

After Steps 1–2 your `outputs/required_experiments/` will contain values matching the
paper (small floating-point differences across platforms are normal; the qualitative
conclusions and significance decisions are stable).

**Table 1 — in-distribution controls (F1, mean ± std over 5 seeds)**

| Model | F1 |
|---|---|
| Entity only | 0.9279 ± 0.0118 |
| Structural only | 0.9512 ± 0.0093 |
| HRKG only | 0.9638 ± 0.0204 |
| Text only (separately tuned) | 0.9905 ± 0.0088 |
| Lexical + HRKG | 0.9887 ± 0.0118 |
| Lexical + Structural | 0.9924 ± 0.0072 |
| Full, contrastive off (λ₁ = 0) | 0.9858 ± 0.0099 |
| **Full MVGC (λ₁ = 0.5)** | **0.9924 ± 0.0054** |

**Table 2 — paired significance (in-distribution)**

| Comparison | ΔF1 | *t*-test *p* | Wilcoxon *p* |
|---|---|---|---|
| Full MVGC vs. Text only | +0.0019 | 0.37 | 0.50 |
| Full MVGC vs. best fusion | +0.0038 | 0.40 | 0.50 |
| Full MVGC vs. contrastive-off | +0.0066 | 0.16 | 0.25 |

*No in-distribution comparison is significant at α = 0.05 → the benchmark is saturated.*

**Table 3 — length-stratified OOD split (F1, ROC-AUC)**

| Model | F1 | ROC-AUC |
|---|---|---|
| Text only | 0.4824 | 0.9032 |
| Lexical + HRKG | 0.6015 | 0.9689 |
| Full, contrastive off | 0.5596 | 0.9681 |
| **Full MVGC** | **0.6354** | 0.9642 |
| **HRKG only** | **0.7018** | 0.9690 |

OOD significance: text-only vs. Full *p* = 0.002; text-only vs. HRKG-only *p* < 0.001.

**Table 4 — extractor type-level audit (silver sample, 240 records)**

| Field | Precision | Recall | F1 |
|---|---|---|---|
| Entities | 0.336 | 0.871 | 0.485 |
| Relations | 0.154 | 0.840 | 0.261 |
| Qualifiers | 0.431 | 0.644 | 0.516 |

**Near-duplicate audit:** 16 / 2927 = 0.5% (TF–IDF cosine ≥ 0.9).

---

## 6. Mapping: paper artifact → command → output file

| Paper artifact | Command | Output file(s) |
|---|---|---|
| Table 1 (in-distribution controls) | Step 2 | `r2_indist_summary.csv`, `r2_indist_perseed.csv` |
| Table 2 (significance) | Step 2 | `r2_significance.csv` |
| Table 3 (OOD split) | Step 2 | `r2_ood_summary.csv`, `r2_ood_significance.csv` |
| Near-duplicate audit | Step 2 | `r2_near_duplicate.csv` |
| Table 4 (extractor audit) | Step 1 | `extractor_silver_summary.csv` |
| Table 5 (partial-transcript) | Step 1 | `partial_transcript_summary.csv` |
| Noise robustness | Step 1 | `noise_robustness_summary.csv` |
| Scenario-blocked control | Step 1 | `cluster_blocked_control_summary.csv` |
| Figure 2 (graph-centric comparison) | optional `mvcg.py` | `figures/fig2_f1_gap_ranking.*` |
| Figure 3 (sensitivity) | Step 3 | `figures/fig3_hyperparameter_sensitivity.*` |
| Full neural retraining | Step 4 (optional) | `full_mvgc_neural_multiseed_summary.csv` |

---

## 7. PyG/KoBERT graph-baseline track

The graph-centric baseline comparison (Figure 2) was produced by the original
implementation in `mvcg.py`, which trains ten GNN baselines and the CMVHRKG model under
one PyTorch-Geometric interface. This track requires the heavyweight optional
dependencies and benefits from a GPU; **it is not required for any conclusion in the
paper.**

```bash
pip install "torch-geometric>=2.5" "transformers>=4.40" "sentencepiece>=0.1.99" "alive-progress>=3.1"

# Train/evaluate a single model:
python mvcg.py --gnn_type CMVHRKG --epochs 10

# Or run all baselines (RGCN, HGT, HAN, GeneralConv, FiLMConv, HAHE, QUAD,
# LightHGNN, OnDeviceHRGNN, CMVHRKG, StarE):
python mvcg.py --epochs 10

# Quick smoke test on a subset:
python mvcg.py --gnn_type CMVHRKG --test_mode
```

---

## 8. Configuration and customization

- **Seeds.** The five evaluation seeds are `[13, 17, 23, 29, 31]`, defined in the runner
  scripts and in `config/reviewer_protocol.json`.
- **Extraction schema.** Entity/relation lexicons live in `config/ner_relations*.json`
  and the six qualifier families in `config/qualifiers.json`. Editing these changes the
  HRKG feature table built in Step 1.
- **Paths.** Both stage scripts accept `--data`, `--feature_dir` / `--config_dir`, and
  `--output_dir` so you can redirect inputs and outputs.

---

## 9. Dataset and ethics

- **Dataset:** KorCCVi v2, 2,927 Korean call transcripts (695 phishing, 2,232 benign),
  provided here as `dataset/voicephishing_data.csv` with columns `id, transcript, label`.
- The data is intended for **defensive fraud-detection research only**. Any deployment
  on real conversational telemetry must enforce data minimization (e.g., on-device or
  trusted-execution processing, PII redaction or salted hashing before ingestion, and
  retention of normalized graph schemata rather than raw text), as discussed in the
  paper's limitations section.

---

## 10. Hardware and runtime

| Stage | Hardware | Approx. runtime |
|---|---|---|
| Step 1 (features, splits, controlled tables) | CPU | 2–4 min |
| Step 2 (Tables 1–3) | CPU | 3–6 min |
| Step 3 (Figure 3) | CPU | < 1 min |
| Step 4 (neural multi-seed, optional) | CPU | 2–3 min |
| `mvcg.py` (optional Figure 2 track) | GPU recommended | varies |

Reference workstation: MacBook Pro M3, 128 GB RAM, CPU only.

---

## 11. Citation

```bibtex
@article{kim2026mvgc,
  title   = {MVGC: Multi-View Hyper-Relational Knowledge Graph Contrastive Learning for Voice Phishing Detection},
  author  = {Kim, Sungsoo},
  year    = {2026}
}
```

## 12. Contact

Sungsoo Kim — Electronics and Telecommunications Research Institute (ETRI) —
`sungsoo@etri.re.kr`

## 13. Acknowledgement

This work was supported by an Institute of Information & Communications Technology
Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)
(No. RS-2025-02215393, *Development of Detection and Prediction Technology for New and
Unknown Voice Phishing*).
