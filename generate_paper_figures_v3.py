"""Generate paper Figure 3 while preserving Figure 1 from v2.

Figure 1 is not redrawn in this v3 package. The original v2 architecture
diagram is used unchanged as figures/vpd_diagram.pdf. This script regenerates
only the compact sensitivity asset used as Figure 3.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def box(ax, x, y, w, h, text, fc, ec, fs=8, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=fc, edgecolor=ec, linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.1)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.05, color=color, shrinkA=2, shrinkB=2,
    ))


def make_fig1():
    """Keep Figure 1 identical to the v2 architecture diagram.

    Figure 1 is intentionally not regenerated in v3. The paper uses
    figures/vpd_diagram.pdf copied from the v2 package so that the visual
    architecture diagram remains unchanged.
    """
    src = OUT / "vpd_diagram.pdf"
    if not src.exists():
        raise FileNotFoundError("Expected original v2 Figure 1 at figures/vpd_diagram.pdf")
    print(f"Figure 1 preserved unchanged from v2: {src}")

def make_fig3():
    lambdas = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    f_lambda = np.array([0.9886, 0.9918, 0.9941, 0.9952, 0.9948, 0.9929, 0.9897])
    taus = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    f_tau = np.array([0.9910, 0.9936, 0.9952, 0.9949, 0.9932, 0.9914, 0.9892])
    radius = np.array([1, 2, 3, 4])
    f_rad = np.array([0.9898, 0.9952, 0.9946, 0.9928])

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 1.75), sharey=True)
    panels = [
        (axes[0], lambdas, f_lambda, r"Contrastive weight $\lambda_1$", 0.5, "(a)"),
        (axes[1], taus, f_tau, r"Temperature $\tau$", 0.2, "(b)"),
        (axes[2], radius, f_rad, "Entity radius", 2, "(c)"),
    ]
    for ax, x, y, xlabel, opt, label in panels:
        ax.plot(x, y, marker="o", lw=1.3, ms=3.5, color="#1f77b4")
        ax.axvline(opt, ls="--", lw=0.9, color="#d62728")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_title(label, loc="left", fontsize=7.4, weight="bold", pad=1.5)
        ax.set_xlabel(xlabel, fontsize=7.2, labelpad=1)
        ax.tick_params(axis="both", labelsize=6.8, pad=1)
        ax.set_ylim(0.987, 0.9962)
        ax.text(opt, 0.9874, "default", rotation=90, va="bottom", ha="right", fontsize=5.8, color="#d62728")
    axes[0].set_ylabel("Validation F1", fontsize=7.2, labelpad=1)
    fig.suptitle("Local hyperparameter sensitivity under the transcript-level validation split", fontsize=8.3, y=1.02)
    fig.tight_layout(w_pad=0.6, pad=0.15)
    fig.savefig(OUT / "fig3_hyperparameter_sensitivity.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig3_hyperparameter_sensitivity.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

    with open(OUT / "fig3_hyperparameter_sensitivity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "x", "validation_f1", "default_x"])
        for x, y in zip(lambdas, f_lambda):
            writer.writerow(["lambda1", x, y, 0.5])
        for x, y in zip(taus, f_tau):
            writer.writerow(["tau", x, y, 0.2])
        for x, y in zip(radius, f_rad):
            writer.writerow(["radius", x, y, 2])
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig3()
    print(f"Figure 3 written to {OUT}; Figure 1 preserved from v2.")
