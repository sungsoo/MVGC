"""Generate paper Figure 3 while preserving Figure 1 from v2.

Figure 1 is not redrawn. The original v2 architecture diagram is used unchanged
as figures/vpd_diagram.pdf. Figure 3 is generated from empirical sensitivity
results produced by mvgc_hyperparameter_sensitivity.py. If the empirical CSV is
not present, this script stops with an instruction rather than silently using
hard-coded values.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def make_fig1():
    src = OUT / "vpd_diagram.pdf"
    if not src.exists():
        raise FileNotFoundError("Expected original v2 Figure 1 at figures/vpd_diagram.pdf")
    print(f"Figure 1 preserved unchanged from v2: {src}")


def load_empirical_sensitivity() -> pd.DataFrame:
    candidates = [
        OUT / "fig3_hyperparameter_sensitivity.csv",
        ROOT / "outputs" / "v3_required_experiments" / "hyperparameter_sensitivity_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"panel", "x", "validation_f1"}.issubset(df.columns):
                return df
            if {"sweep", "param_value", "best_validation_f1_mean"}.issubset(df.columns):
                return df.rename(columns={
                    "sweep": "panel",
                    "param_value": "x",
                    "best_validation_f1_mean": "validation_f1",
                    "best_validation_f1_std": "validation_f1_std",
                })
    raise FileNotFoundError(
        "Empirical sensitivity results were not found. Run:\n"
        "  python mvgc_hyperparameter_sensitivity.py --feature_dir outputs/v3_required_experiments "
        "--output_dir outputs/v3_required_experiments --figure_dir figures\n"
        "before regenerating Figure 3."
    )


def make_fig3():
    df = load_empirical_sensitivity()
    panels = [
        ("lambda1", r"Contrastive weight $\lambda_1$", "(a)"),
        ("tau", r"Temperature $\tau$", "(b)"),
        ("radius", "Entity radius", "(c)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 1.75), sharey=True)
    for ax, (panel, xlabel, label) in zip(axes, panels):
        sub = df[df["panel"] == panel].copy().sort_values("x")
        if sub.empty:
            raise ValueError(f"No sensitivity rows for panel={panel}")
        x = sub["x"].astype(float).to_numpy()
        y = sub["validation_f1"].astype(float).to_numpy()
        yerr = sub["validation_f1_std"].fillna(0).astype(float).to_numpy() if "validation_f1_std" in sub else None
        if "default_value" in sub:
            default = float(sub["default_value"].iloc[0])
        else:
            default = {"lambda1": 0.5, "tau": 0.2, "radius": 2}[panel]
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.3, ms=3.3, capsize=2)
        ax.axvline(default, ls="--", lw=0.9)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_title(label, loc="left", fontsize=7.4, weight="bold", pad=1.5)
        ax.set_xlabel(xlabel, fontsize=7.2, labelpad=1)
        ax.tick_params(axis="both", labelsize=6.8, pad=1)
        ax.text(default, max(0.0, y.min() - 0.0002), "default", rotation=90, va="bottom", ha="right", fontsize=5.8)
    ymin = max(0.0, df["validation_f1"].astype(float).min() - 0.004)
    ymax = min(1.0, df["validation_f1"].astype(float).max() + 0.002)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("Validation F1", fontsize=7.2, labelpad=1)
    fig.suptitle("Empirical hyperparameter sensitivity from validation experiments", fontsize=8.3, y=1.02)
    fig.tight_layout(w_pad=0.6, pad=0.15)
    fig.savefig(OUT / "fig3_hyperparameter_sensitivity.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig3_hyperparameter_sensitivity.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig3()
    print(f"Figure 3 regenerated from empirical sensitivity results in {OUT}.")
