# MVGC: Multi-View Hyper-Relational Knowledge Graph Contrastive Learning for Voice Phishing Detection

This repository provides the implementation package for **MVGC**, a multi-view hyper-relational knowledge graph contrastive learning framework for voice phishing detection. MVGC represents each call transcript as a qualifier-aware hyper-relational knowledge graph (HRKG) and jointly learns structural, lexical, and entity-centric evidence for fraud detection.

The repository is designed to reproduce the experimental artifacts reported in the paper, including controlled text/graph/fusion experiments, Full MVGC multi-seed neural retraining, extractor audit, HRKG-noise robustness, scenario-signature blocking, partial-transcript evaluation, **hyperparameter sensitivity experiments**, and paper figures.

> Project repository: <https://github.com/sungsoo/MVGC>

---

## Overview

<p align="center">
  <img src="https://sungsoo.github.io/images/mvgc_diagram.png" alt="Figure 1. Overview of the MVGC framework" width="900"/>
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the MVGC framework. The pipeline converts a call transcript into a qualifier-aware hyper-relational knowledge graph and learns complementary structural, lexical, and entity-centric views through view-specific encoders, cross-view contrastive alignment, and attention-based fusion.
</p>


---

## 2. Repository Contents

After extracting the implementation package, the repository should contain files and directories similar to the following:

```text
dataset/
config/
figures/
outputs/
mvcg.py
mvgc_v3_fast_experiments.py
mvgc_full_neural_multiseed.py
mvgc_hyperparameter_sensitivity.py
generate_paper_figures_v3.py
requirements_mvgc.txt
README.md
```

Key files:

| Path | Description |
|---|---|
| `dataset/voicephishing_data.csv` | Main transcript-level dataset used in the experiments. |
| `config/` | Configuration files used by the controlled experiments. |
| `mvgc_v3_fast_experiments.py` | Reproduces controlled experiments: text-only, HRKG-only, fusion controls, extractor audit, HRKG-noise robustness, scenario blocking, and partial-transcript tests. |
| `mvgc_full_neural_multiseed.py` | Reproduces Full MVGC multi-seed neural retraining results. |
| `mvgc_hyperparameter_sensitivity.py` | Runs actual hyperparameter sensitivity experiments over `lambda1`, `tau`, and entity radius grids. |
| `generate_paper_figures_v3.py` | Regenerates Figure 3 from the experimental hyperparameter sensitivity CSV. |
| `mvcg.py` | Original PyTorch-Geometric-based GNN/MVGC training script for graph-centric baselines. |
| `figures/vpd_diagram.pdf` | Figure 1 source used in the paper. |
| `figures/vpd_diagram.png` | GitHub-renderable preview image for Figure 1 shown near the top of this README. |
| `figures/results.pdf` | Figure 2 used in the paper. |
| `figures/fig3_hyperparameter_sensitivity.pdf` | Figure 3 generated from sensitivity experiment outputs. |
| `outputs/v3_required_experiments/` | Main output directory for reproducibility artifacts. |

Although this package is versioned as **v4**, the main experiment output directory is named `outputs/v3_required_experiments/` because the experimental protocol was finalized in the v3 experiment stage. Version v4 updates the paper presentation and adds the reproducible hyperparameter sensitivity experiment pipeline while preserving the same core result structure.

---

## 3. Recommended Environment

We recommend using:

- Ubuntu 20.04/22.04 or macOS
- Python 3.10.13
- `pyenv`
- `pyenv-virtualenv`
- pip
- CPU execution for controlled experiments and hyperparameter sensitivity sweeps
- Optional GPU for full PyTorch-Geometric baseline retraining

The controlled experiments, Full MVGC multi-seed neural retraining, and hyperparameter sensitivity experiments can be reproduced on CPU with standard scientific Python packages. The original PyTorch-Geometric baseline retraining in `mvcg.py` requires additional dependencies such as `torch-geometric` and `transformers`.

---

## 4. Install `pyenv` and `pyenv-virtualenv`

If `pyenv` and `pyenv-virtualenv` are already installed, skip this section.

### 4.1 macOS

Using Homebrew:

```bash
brew update
brew install pyenv pyenv-virtualenv
```

For `zsh`, add the following to `~/.zshrc`:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
```

For `bash`, add the same lines to `~/.bashrc` or `~/.bash_profile`.

### 4.2 Ubuntu / Linux

Install build dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential curl git libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget llvm libncursesw5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Install `pyenv`:

```bash
curl https://pyenv.run | bash
```

For `bash`, add the following to `~/.bashrc`:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
```

For `zsh`, add the same lines to `~/.zshrc`.

Confirm installation:

```bash
pyenv --version
pyenv virtualenv --version
```

---

## 5. Prepare the MVGC Repository

Place the v4 implementation package in a working directory. For example:

```bash
mkdir -p ~/work/mvgc
cd ~/work/mvgc
unzip mvgc-code-v4-hyperparam-final.zip -d mvgc-v4
cd mvgc-v4
```

Check the package contents:

```bash
ls
cat VERSION.txt
```

The version file should indicate that this is the v4 implementation package.

---

## 6. Create a Python Environment with `pyenv` and `pyenv-virtualenv`

Install Python 3.10.13:

```bash
pyenv install 3.10.13
```

Create a dedicated virtual environment:

```bash
pyenv virtualenv 3.10.13 mvgc-v4-3.10
```

Activate this environment locally for the repository:

```bash
cd ~/work/mvgc/mvgc-v4
pyenv local mvgc-v4-3.10
```

Confirm that the correct Python interpreter is being used:

```bash
python --version
which python
```

The Python version should be `3.10.13`, and the executable path should point to the `mvgc-v4-3.10` environment under `~/.pyenv/versions/`.

Upgrade pip and build tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 7. Install Dependencies

Install the full dependency set:

```bash
pip install -r requirements_mvgc.txt
```

If installation of `torch-geometric` or `transformers` fails due to local wheel compatibility, the main CPU-based controlled experiments and the hyperparameter sensitivity experiments can still be reproduced with the core dependencies:

```bash
pip install pandas numpy scikit-learn scipy matplotlib joblib torch
```

The original PyTorch-Geometric graph baseline training in `mvcg.py` requires additional dependencies, including `torch-geometric`, `transformers`, and `sentencepiece`.

Check core package installation:

```bash
python - <<'PY'
import pandas
import numpy
import sklearn
import scipy
import matplotlib
import torch

print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)
print("torch:", torch.__version__)
PY
```

Optionally check PyTorch-Geometric:

```bash
python - <<'PY'
try:
    import torch_geometric
    print("torch_geometric:", torch_geometric.__version__)
except Exception as e:
    print("torch_geometric is not available:", e)
PY
```

---

## 8. Check the Dataset

The main dataset is located at:

```text
dataset/voicephishing_data.csv
```

Check the number of rows, columns, and label distribution:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("dataset/voicephishing_data.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nLabel distribution:")
print(df["label"].value_counts())
PY
```

The paper uses 2,927 transcript-level samples with phishing and non-phishing labels.

---

## 9. Reset Output Directories

To reproduce all results from scratch, remove the previous experiment outputs and create a clean directory:

```bash
rm -rf outputs/v3_required_experiments
mkdir -p outputs/v3_required_experiments
```

The directory name includes `v3` because the experimental protocol was finalized in the v3 stage. The v4 package preserves the same core experimental protocol while adding an executable hyperparameter sensitivity pipeline.

---

## 10. Run Controlled Experiments

Run the main controlled experiment script:

```bash
python mvgc_v3_fast_experiments.py \
  --data dataset/voicephishing_data.csv \
  --config_dir config \
  --output_dir outputs/v3_required_experiments
```

This script generates results for:

- text-only controls
- structural-view controls
- entity-view controls
- HRKG-only controls
- text-HRKG late fusion
- lexical-HRKG early fusion
- extractor silver audit
- HRKG-noise robustness
- scenario-signature blocking
- partial-transcript detection
- feature matrices
- seed-specific split definitions

Expected output files include:

```text
outputs/v3_required_experiments/transcript_level_control_metrics.csv
outputs/v3_required_experiments/transcript_level_control_summary.csv
outputs/v3_required_experiments/cluster_blocked_control_metrics.csv
outputs/v3_required_experiments/cluster_blocked_control_summary.csv
outputs/v3_required_experiments/noise_robustness_metrics.csv
outputs/v3_required_experiments/noise_robustness_summary.csv
outputs/v3_required_experiments/partial_transcript_metrics.csv
outputs/v3_required_experiments/partial_transcript_summary.csv
outputs/v3_required_experiments/extractor_silver_detail.csv
outputs/v3_required_experiments/extractor_silver_summary.csv
outputs/v3_required_experiments/hrkg_feature_matrix.csv
outputs/v3_required_experiments/lexical_feature_matrix.csv
outputs/v3_required_experiments/split_transcript_seed_13.json
outputs/v3_required_experiments/split_transcript_seed_17.json
outputs/v3_required_experiments/split_transcript_seed_23.json
outputs/v3_required_experiments/split_transcript_seed_29.json
outputs/v3_required_experiments/split_transcript_seed_31.json
```

Inspect the main summaries:

```bash
cat outputs/v3_required_experiments/transcript_level_control_summary.csv
cat outputs/v3_required_experiments/extractor_silver_summary.csv
cat outputs/v3_required_experiments/noise_robustness_summary.csv
cat outputs/v3_required_experiments/cluster_blocked_control_summary.csv
cat outputs/v3_required_experiments/partial_transcript_summary.csv
```

---

## 11. Run Full MVGC Multi-Seed Neural Retraining

The paper reports Full MVGC using mean and standard deviation over five seeds. To reproduce these results, run:

```bash
python mvgc_full_neural_multiseed.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments
```

This script uses the feature matrices and split files generated in the previous step.

Expected output files:

```text
outputs/v3_required_experiments/full_mvgc_neural_multiseed_metrics.csv
outputs/v3_required_experiments/full_mvgc_neural_multiseed_summary.csv
outputs/v3_required_experiments/full_mvgc_neural_predictions_seed_13.csv
outputs/v3_required_experiments/full_mvgc_neural_predictions_seed_17.csv
outputs/v3_required_experiments/full_mvgc_neural_predictions_seed_23.csv
outputs/v3_required_experiments/full_mvgc_neural_predictions_seed_29.csv
outputs/v3_required_experiments/full_mvgc_neural_predictions_seed_31.csv
```

The paper reports the following representative Full MVGC results:

```text
Accuracy  = 0.9973 ± 0.0019
Precision = 0.9944 ± 0.0084
Recall    = 0.9943 ± 0.0085
F1        = 0.9943 ± 0.0040
```

Check the generated summary:

```bash
cat outputs/v3_required_experiments/full_mvgc_neural_multiseed_summary.csv
```

---

## 12. Run Hyperparameter Sensitivity Experiments

Version v4 includes an executable hyperparameter sensitivity pipeline. Unlike a static plotting script, `mvgc_hyperparameter_sensitivity.py` actually evaluates the model under hyperparameter grids and writes experimental metrics before Figure 3 is generated.

The sensitivity sweep covers three groups:

1. `lambda1`: the cross-view contrastive alignment weight
2. `tau`: the InfoNCE temperature
3. `entity_radius`: the radius used by the entity-centric HRKG view

### 12.1 Default sensitivity experiment

Run:

```bash
python mvgc_hyperparameter_sensitivity.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments \
  --figure_dir figures
```

This command expects the feature matrices and split files generated by `mvgc_v3_fast_experiments.py`. Therefore, run Section 9 first.

Expected output files:

```text
outputs/v3_required_experiments/hyperparameter_sensitivity_metrics.csv
outputs/v3_required_experiments/hyperparameter_sensitivity_summary.csv
outputs/v3_required_experiments/hyperparameter_sensitivity_manifest.json
figures/fig3_hyperparameter_sensitivity.csv
figures/fig3_hyperparameter_sensitivity.pdf
figures/fig3_hyperparameter_sensitivity.png
```

The metrics file stores seed-level or run-level results, while the summary file stores the mean and standard deviation for each hyperparameter setting. The manifest file records the grid, seeds, input paths, and run configuration.

### 12.2 Larger multi-seed sensitivity sweep

For a more stable but slower sensitivity analysis, run:

```bash
python mvgc_hyperparameter_sensitivity.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments \
  --figure_dir figures \
  --seeds 13,17,23 \
  --epochs 20
```

You can also adjust grid values explicitly:

```bash
python mvgc_hyperparameter_sensitivity.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments \
  --figure_dir figures \
  --lambda1_grid 0.0,0.05,0.1,0.2,0.4 \
  --tau_grid 0.05,0.1,0.2,0.5,1.0 \
  --entity_radius_grid 1,2,3,4 \
  --seeds 13,17,23 \
  --epochs 20
```

### 12.3 Inspect sensitivity outputs

After the run, inspect the outputs:

```bash
cat outputs/v3_required_experiments/hyperparameter_sensitivity_summary.csv
cat outputs/v3_required_experiments/hyperparameter_sensitivity_manifest.json
ls figures/fig3_hyperparameter_sensitivity.pdf
```

The generated `figures/fig3_hyperparameter_sensitivity.pdf` is the Figure 3 artifact used by the paper.

---

## 13. Regenerate Figure 3 from Existing Sensitivity Results

If sensitivity experiment results already exist and you only want to regenerate the figure, run:

```bash
python generate_paper_figures_v3.py
```

In v4, this script reads the experimental sensitivity CSV rather than relying on hard-coded sensitivity values.

Expected files:

```text
figures/fig3_hyperparameter_sensitivity.pdf
figures/fig3_hyperparameter_sensitivity.png
figures/fig3_hyperparameter_sensitivity.csv
```

The paper uses the following figure files:

```text
figures/vpd_diagram.pdf                         Figure 1
figures/results.pdf                             Figure 2
figures/fig3_hyperparameter_sensitivity.pdf     Figure 3
```

Check that they exist:

```bash
ls figures/vpd_diagram.pdf
ls figures/results.pdf
ls figures/fig3_hyperparameter_sensitivity.pdf
```

---

## 14. Print All Main Result Summaries

Use the following command to display all major result summaries together:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

out = Path("outputs/v3_required_experiments")

files = [
    "transcript_level_control_summary.csv",
    "full_mvgc_neural_multiseed_summary.csv",
    "extractor_silver_summary.csv",
    "noise_robustness_summary.csv",
    "cluster_blocked_control_summary.csv",
    "partial_transcript_summary.csv",
    "hyperparameter_sensitivity_summary.csv",
]

for f in files:
    p = out / f
    print("\n" + "=" * 80)
    print(f)
    print("=" * 80)
    if p.exists():
        print(pd.read_csv(p).to_string(index=False))
    else:
        print("Missing:", p)
PY
```

These files correspond to the controlled evaluations and sensitivity analyses reported in the paper.

---

## 15. Optional: Retrain Original PyTorch-Geometric Baselines

The paper preserves the original graph-centric Figure 2 as `figures/results.pdf`. If you want to retrain the original PyTorch-Geometric models, use `mvcg.py`.

A default run:

```bash
python mvcg.py --epochs 10
```

Individual model runs:

```bash
python mvcg.py --gnn_type RGCN --epochs 10
python mvcg.py --gnn_type HGT --epochs 10
python mvcg.py --gnn_type HAN --epochs 10
python mvcg.py --gnn_type GeneralConv --epochs 10
python mvcg.py --gnn_type FiLMConv --epochs 10
python mvcg.py --gnn_type HAHE --epochs 10
python mvcg.py --gnn_type QUAD --epochs 10
python mvcg.py --gnn_type LightHGNN --epochs 10
python mvcg.py --gnn_type OnDeviceHRGNN --epochs 10 --distillation_epochs 10
python mvcg.py --gnn_type StarE --epochs 10
python mvcg.py --gnn_type CMVHRKG --epochs 10
```

This optional retraining path requires PyTorch-Geometric and related dependencies. Results may vary slightly depending on hardware, package versions, cached language models, and random seeds.

---

## 16. Mapping Paper Results to Repository Artifacts

| Paper item | Command | Output artifact |
|---|---|---|
| Dataset statistics | Dataset check snippet | `dataset/voicephishing_data.csv` |
| Figure 1 | Preserved source figure | `figures/vpd_diagram.pdf` |
| Figure 2 | Preserved result figure | `figures/results.pdf` |
| Text-only / HRKG-only / fusion controls | `mvgc_v3_fast_experiments.py` | `transcript_level_control_summary.csv` |
| Full MVGC mean ± std | `mvgc_full_neural_multiseed.py` | `full_mvgc_neural_multiseed_summary.csv` |
| Extractor audit | `mvgc_v3_fast_experiments.py` | `extractor_silver_summary.csv` |
| HRKG-noise robustness | `mvgc_v3_fast_experiments.py` | `noise_robustness_summary.csv` |
| Scenario-signature blocking | `mvgc_v3_fast_experiments.py` | `cluster_blocked_control_summary.csv` |
| Partial-transcript detection | `mvgc_v3_fast_experiments.py` | `partial_transcript_summary.csv` |
| Hyperparameter sensitivity | `mvgc_hyperparameter_sensitivity.py` | `hyperparameter_sensitivity_summary.csv` |
| Figure 3 | `mvgc_hyperparameter_sensitivity.py` or `generate_paper_figures_v3.py` | `figures/fig3_hyperparameter_sensitivity.pdf` |

---

## 17. One-Shot Reproduction Script

The following command sequence reproduces the core experimental results from a clean setup:

```bash
# Prepare repository
mkdir -p ~/work/mvgc
cd ~/work/mvgc
unzip mvgc-code-v4-hyperparam-final.zip -d mvgc-v4
cd mvgc-v4

# Create Python environment
pyenv install 3.10.13
pyenv virtualenv 3.10.13 mvgc-v4-3.10
pyenv local mvgc-v4-3.10

# Check Python
python --version
which python

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_mvgc.txt

# Reset outputs
rm -rf outputs/v3_required_experiments
mkdir -p outputs/v3_required_experiments

# Run controlled experiments
python mvgc_v3_fast_experiments.py \
  --data dataset/voicephishing_data.csv \
  --config_dir config \
  --output_dir outputs/v3_required_experiments

# Run Full MVGC multi-seed neural retraining
python mvgc_full_neural_multiseed.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments

# Run actual hyperparameter sensitivity experiments and regenerate Figure 3
python mvgc_hyperparameter_sensitivity.py \
  --data dataset/voicephishing_data.csv \
  --feature_dir outputs/v3_required_experiments \
  --output_dir outputs/v3_required_experiments \
  --figure_dir figures

# Inspect main results
cat outputs/v3_required_experiments/transcript_level_control_summary.csv
cat outputs/v3_required_experiments/full_mvgc_neural_multiseed_summary.csv
cat outputs/v3_required_experiments/extractor_silver_summary.csv
cat outputs/v3_required_experiments/noise_robustness_summary.csv
cat outputs/v3_required_experiments/cluster_blocked_control_summary.csv
cat outputs/v3_required_experiments/partial_transcript_summary.csv
cat outputs/v3_required_experiments/hyperparameter_sensitivity_summary.csv

# Check figures
ls figures/vpd_diagram.pdf
ls figures/results.pdf
ls figures/fig3_hyperparameter_sensitivity.pdf
```

---

## 18. Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{mvgc2026,
  title   = {Multi-View Hyper-Relational Knowledge Graph Contrastive Learning for Voice Phishing Detection},
  author  = {Sungsoo Kim},
  year    = {2026}
}
```

Please update the citation entry once the final bibliographic information becomes available.

---

## 19. Notes on Data and Privacy

Voice phishing transcripts may contain sensitive personal or conversational information. If raw transcripts cannot be redistributed due to privacy or licensing constraints, users should release or use anonymized dataset artifacts, hashed split identifiers, extracted features, and reproducibility scripts in accordance with applicable data governance policies.

The project is intended for research and reproducibility. Deployment in real-world warning systems should include privacy protection, human oversight, calibration at low false-positive operating points, and fairness checks across speaker and dialectal subgroups.
