from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def read_rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def grouped_prompt_means(rows, value, extra_filter=None):
    cells = {}
    for row in rows:
        if extra_filter and not extra_filter(row):
            continue
        key = (row["domain"], int(row["top_k"]), row["temperature_ratio"], int(row["prompt_id"]))
        cells.setdefault(key, []).append(float(row[value]))
    result = {}
    for (domain, k, tr, prompt), values in cells.items():
        result.setdefault((domain, k, tr), []).append((prompt, float(np.mean(values))))
    return result


def cluster_ci(cluster_values, seed=20260924, reps=2000):
    arr = np.asarray(cluster_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arr), size=(reps, len(arr)))
    boot = arr[draws].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def summary_table(ident, local):
    gaps = grouped_prompt_means(ident, "head_mass_gap")
    local_moment = grouped_prompt_means(local, "moment4_tail_tv")
    local_sample = grouped_prompt_means(local, "sample6_tail_tv")
    local_cert = grouped_prompt_means(local, "moment4_certified")
    rows = []
    for key, prompts in sorted(gaps.items()):
        domain, k, tr = key
        mean, lo, hi = cluster_ci([v for _, v in prompts], seed=20260924 + k + int(float(tr)*100))
        row = {"domain": domain, "top_k": k, "temperature_ratio": tr,
               "head_mass_gap_prompt_mean": mean, "cluster_bootstrap_95_low": lo,
               "cluster_bootstrap_95_high": hi}
        if float(tr) in (.9, 1.1):
            local_key = (domain, k, tr)
            prompt_values = local_moment.get(local_key, [])
            sample_values = local_sample.get(local_key, [])
            cert_values = local_cert.get(local_key, [])
            by_prompt = {p: v for p, v in prompt_values}
            sample_by_prompt = {p: v for p, v in sample_values}
            cert_by_prompt = {p: v for p, v in cert_values}
            moment_all = [float(r["moment4_tail_tv"]) for r in local
                          if r["domain"] == domain and int(r["top_k"]) == k and r["temperature_ratio"] == tr]
            sample_all = [float(r["sample6_tail_tv"]) for r in local
                          if r["domain"] == domain and int(r["top_k"]) == k and r["temperature_ratio"] == tr]
            row.update({
                "moment4_tv_median": float(np.median(moment_all)),
                "moment4_tv_p90": float(np.quantile(moment_all, .9)),
                "moment4_tv_max": float(np.max(moment_all)),
                "moment4_certificate_coverage": float(np.mean([
                    int(r["moment4_certified"]) for r in local
                    if r["domain"] == domain and int(r["top_k"]) == k and r["temperature_ratio"] == tr])),
                "sample6_tv_median": float(np.median(sample_all)),
                "sample6_tv_p90": float(np.quantile(sample_all, .9)),
                "moment4_prompt_cluster_count": len(by_prompt),
                "sample6_prompt_cluster_count": len(sample_by_prompt),
                "certificate_cluster_count": len(cert_by_prompt),
            })
        rows.append(row)
    with (RESULTS / "cache_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_identifiability(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True)
    colors = {"ordinary": BLUE, "arithmetic": ORANGE}
    all_ratios = sorted({float(r["temperature_ratio"]) for r in rows})
    for ax, k in zip(axes, (8, 32)):
        for domain in ("ordinary", "arithmetic"):
            xvals, means, lows, highs = [], [], [], []
            for tr in all_ratios:
                values = grouped_prompt_means(rows, "head_mass_gap", lambda r: r["domain"] == domain and int(r["top_k"]) == k)
                ps = values.get((domain, k, f"{tr:.6g}"), [])
                if not ps:
                    continue
                mean, lo, hi = cluster_ci([v for _, v in ps], seed=20260924 + k + int(tr*100))
                xvals.append(tr); means.append(mean); lows.append(mean-lo); highs.append(hi-mean)
            ax.errorbar(xvals, means, yerr=np.asarray([lows, highs]), color=colors[domain],
                        marker="o", markersize=2.8, linewidth=1.1, capsize=1.5,
                        label=domain.capitalize())
        ax.axhline(.05, color="#666666", linestyle="--", linewidth=.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([.5, .75, 1., 1.25, 1.5, 1.75, 2.])
        ax.set_xticklabels(["0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0"])
        ax.set_title(f"Top-{k} plus tail mass")
        ax.set_xlabel("Temperature ratio T/T0")
        ax.grid(axis="y", alpha=.22, linewidth=.5)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Legal interval width in head mass")
    fig.tight_layout(pad=.6, w_pad=1.0)
    fig.savefig(FIGURES / "fig1_identifiability.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_cache(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.3), sharey=True)
    colors = [BLUE, ORANGE]
    for row_i, domain in enumerate(("ordinary", "arithmetic")):
        for col_i, k in enumerate((8, 32)):
            ax = axes[row_i, col_i]
            data, positions, labels = [], [], []
            for tr_i, tr in enumerate(("0.9", "1.1")):
                subset = [r for r in rows if r["domain"] == domain and int(r["top_k"]) == k and r["temperature_ratio"] == tr]
                # The Taylor cache is deterministic for a context position; keep one row per context.
                moment = {}
                for r in subset:
                    moment[(r["prompt_id"], r["context_position"])] = float(r["moment4_tail_tv"])
                sample = [float(r["sample6_tail_tv"]) for r in subset]
                data.extend([list(moment.values()), sample])
                positions.extend([tr_i*3+1, tr_i*3+2])
                labels.extend([f"{tr}:M4", f"{tr}:S6"])
            boxes = ax.boxplot(data, positions=positions, widths=.65, showfliers=True,
                               patch_artist=True, medianprops={"color": "black", "linewidth": .8},
                               flierprops={"marker": ".", "markersize": 1.8, "alpha": .3})
            for i, box in enumerate(boxes["boxes"]):
                box.set_facecolor(colors[i % 2]); box.set_alpha(.42)
                box.set_edgecolor(colors[i % 2]); box.set_linewidth(.8)
            ax.set_yscale("log")
            ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_title(f"{domain.capitalize()}, top-{k}")
            ax.grid(axis="y", alpha=.22, linewidth=.5)
    axes[0, 0].set_ylabel("Coarse top-k/other TV error")
    axes[1, 0].set_ylabel("Coarse top-k/other TV error")
    fig.tight_layout(pad=.7)
    fig.savefig(FIGURES / "fig2_cache_error.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    ident = read_rows("identifiability_rows.csv")
    local = read_rows("local_cache_rows.csv")
    summary_table(ident, local)
    figure_identifiability(ident)
    figure_cache(local)
    print("Wrote fig1_identifiability.pdf, fig2_cache_error.pdf, and results/cache_summary.csv")


if __name__ == "__main__":
    main()
