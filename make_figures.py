import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
D = json.load(open(os.path.join(OUT, "results_full.json")))
S = np.load(os.path.join(OUT, "r2_scores.npz"))

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12,
                     "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
BLUE, GREEN, RED, ORANGE = "#2c5f8a", "#2e7d4f", "#b03a3a", "#c8821a"


def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, n + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, n + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", n)


A = D["R3_leakage"]["ConfigA_ramp"]
t = [0, 1, 2, 3]
bA = [float(A[str(i)]["mean_bits_disclosed_per_16bit_block"]) for i in t]
bB = [0, 0, 0, 16]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(t, bA, "s-", color=RED, lw=2.4, ms=9, label="Configuration A (NPN, ramp)")
ax.plot(t, bB, "o-", color=BLUE, lw=2.4, ms=9, label="Configuration B (multivariate, perfect)")
ax.axvline(2, color="gray", ls=":", lw=1.4)
ax.text(2.05, 1, "threshold minus one", rotation=90, va="bottom", fontsize=8, color="gray")
ax.set_xticks(t)
ax.set_xlabel("Number of compromised shares")
ax.set_ylabel("Information disclosed per 16-bit block (bits)")
ax.set_title("Bit-level information leakage below the threshold")
ax.legend(loc="upper left")
ax.set_ylim(-0.5, 17)
save(fig, "fig_leakage")

g = S["g"]
imp = S["imp"]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.hist(g, bins=40, density=True, alpha=0.6, color=GREEN, label="Genuine (protected)")
ax.hist(imp, bins=40, density=True, alpha=0.5, color=BLUE, label="Impostor (protected)")
ax.axvline(D["R2_recognition"]["EER_threshold"], color=RED, ls="--", lw=1.6, label="EER threshold")
ax.set_xlabel("Normalised Hamming distance")
ax.set_ylabel("Density")
ax.set_title("Genuine vs impostor distributions (protected pipeline)")
ax.legend(fontsize=8)
save(fig, "fig_scores")

fig, ax = plt.subplots(figsize=(6.6, 4.2))
tbl = D["R8_reliability"]["table"]
ls = sorted(float(x) for x in tbl)
for key, lab in [("n5k3", "n=5, k=3"), ("n7k4", "n=7, k=4"), ("n7k3", "n=7, k=3"), ("n9k5", "n=9, k=5")]:
    ax.plot(ls, [tbl["%.2f" % l][key] for l in ls], "o-", lw=2, ms=5, label=lab)
ax.set_xlabel("Share-loss probability  l")
ax.set_ylabel("Reconstruction reliability  Q")
ax.set_title("Reconstruction reliability vs share-loss probability")
ax.legend()
ax.set_ylim(0.6, 1.02)
save(fig, "fig_reliability")

U = D["R9_unlinkability"]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.linspace(0, 1, 200)
from scipy.stats import norm
ax.plot(x, norm.pdf(x, U["mated_mean"], U["mated_sd"]), color=RED, lw=2, label="Mated (same id, different tokens)")
ax.plot(x, norm.pdf(x, U["nonmated_mean"], U["nonmated_sd"]), color=BLUE, lw=2, label="Non-mated (different identities)")
ax.fill_between(x, norm.pdf(x, U["mated_mean"], U["mated_sd"]), alpha=0.15, color=RED)
ax.set_xlabel("Cross-token similarity score")
ax.set_ylabel("Density")
ax.set_title("Unlinkability across tokens (D_sys = %.3f, 0 = fully unlinkable)" % U["D_sys"])
ax.legend(fontsize=8)
save(fig, "fig_unlinkability")
print("all figures done")
