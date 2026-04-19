"""
03_make_figures.py

Generate all figures for the LaTeX report from the real user-level
A/B test data. Saves PDF (vector, for the report) and PNG (preview).

Figures
  fig1_ga4_events      -- GA4 event totals by arm (macro validation)
  fig2_primary         -- full-workflow completion with 95% CIs
  fig3_stage_funnel    -- per-stage reach rates
  fig4_workflow_depth  -- workflow depth distribution
  fig5_linear_path     -- linear-path-score distribution
  fig6_tab_duration    -- total/avg tab duration boxplots
  fig7_logreg_forest   -- logistic regression ORs with bootstrap CIs
  fig8_conversion_funnel -- arrived -> non-bounce -> 2+ stages -> full
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
COLOR_A, COLOR_B = "#4361ee", "#06d6a0"

# 核心修改：获取脚本所在的绝对目录
SCRIPT_DIR = Path(__file__).parent.absolute()
# 基于脚本目录拼接输入/输出路径
FIGDIR = SCRIPT_DIR / "figures"  # 图片保存到脚本目录下的figures文件夹
DF_PATH = SCRIPT_DIR / "analysis_df.csv"
RES_PATH = SCRIPT_DIR / "results.json"

# 创建figures目录（如果不存在）
FIGDIR.mkdir(parents=True, exist_ok=True)

# 读取数据（改为基于脚本目录的路径）
df  = pd.read_csv(DF_PATH)
res = json.load(open(RES_PATH, "r"))
A, B = df[df["ab_version"] == "A"], df[df["ab_version"] == "B"]


def save(fig, name: str) -> None:
    # 保存路径改为脚本目录下的figures文件夹
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


#  fig1 GA4 events -
def fig1_ga4_events() -> None:
    """Macro validation using raw GA4 event counts."""
    ga4 = {
        "button_click":     ("A", 314), "button_click_B":     ("B", 229),
        "tab_switch":       ("A", 199), "tab_switch_B":       ("B", 141),
        "tab_duration":     ("A", 176), "tab_duration_B":     ("B", 132),
        "page_view":        ("A",  38), "page_view_B":        ("B",  31),
        "scroll":           ("A",  45), "scroll_B":           ("B",  38),
        "user_engagement":  ("A",  33), "user_engagement_B":  ("B",  26),
    }
    labels = ["button_click", "tab_switch", "tab_duration",
              "page_view", "scroll", "user_engagement"]
    a_vals = [ga4[k][1] for k in labels]
    b_vals = [ga4[k + "_B"][1] for k in labels]

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, a_vals, w, color=COLOR_A, label="Variant A (n=37)",
           edgecolor="white")
    ax.bar(x + w/2, b_vals, w, color=COLOR_B, label="Variant B (n=30)",
           edgecolor="white")
    for i, (va, vb) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - w/2, va + 4, str(va), ha="center", fontsize=9)
        ax.text(i + w/2, vb + 4, str(vb), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Total GA4 event count")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Figure 1.  Raw GA4 event totals by variant\n"
                 "(2026-04-09 to 2026-04-17, n=67 unique users)")
    save(fig, "fig1_ga4_events")


#  fig2 primary 
def fig2_primary() -> None:
    p1, p2 = res["primary"]["p_A"], res["primary"]["p_B"]
    n1, n2 = res["n_A"], res["n_B"]
    x1, x2 = res["primary"]["x_A"], res["primary"]["x_B"]
    se1 = math.sqrt(p1 * (1 - p1) / n1)
    se2 = math.sqrt(p2 * (1 - p2) / n2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    bars = ax1.bar(["Variant A", "Variant B"], [p1, p2],
                   yerr=[1.96 * se1, 1.96 * se2], capsize=8,
                   color=[COLOR_A, COLOR_B], edgecolor="white", width=0.55)
    for bar, val, x, n in zip(bars, [p1, p2], [x1, x2], [n1, n2]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                 f"{val*100:.1f}%\n({x}/{n})",
                 ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Full-workflow completion rate")
    ax1.set_title("Completion rate (primary outcome)")

    diff = res["primary"]["diff"] * 100
    lo, hi = res["primary"]["ci"][0] * 100, res["primary"]["ci"][1] * 100
    ax2.errorbar([diff], [0.5], xerr=[[diff - lo], [hi - diff]],
                 fmt="o", color="#1a1a2e", capsize=6, markersize=10,
                 elinewidth=2)
    ax2.axvline(0, color="grey", lw=1, ls="--")
    ax2.set_yticks([])
    ax2.set_xlim(min(-2, lo - 5), max(hi + 5, 85))
    ax2.set_xlabel("Absolute difference  (B − A, percentage points)")
    ax2.set_title(
        f"Δ = {diff:+.1f} pp,   95% CI [{lo:+.1f}, {hi:+.1f}]\n"
        f"z = {res['primary']['z']:.2f},  p = {res['primary']['p_value']:.2e}\n"
        f"Fisher OR = {res['primary']['fisher_OR']:.1f},  "
        f"Cohen's h = {res['primary']['cohens_h']:.2f}"
    )

    fig.suptitle("Figure 2.  Primary outcome — full-workflow completion",
                 fontsize=12, y=1.04)
    save(fig, "fig2_primary")


#  fig3 stage funnel
def fig3_stage_funnel() -> None:
    stages = ["EDA", "Cleaning", "Feature Eng."]
    keys = ["reached_eda", "reached_cleaning", "reached_feature_eng"]

    p_a = [res["secondary"][k]["p_A"] for k in keys]
    p_b = [res["secondary"][k]["p_B"] for k in keys]
    p_holm = [res["secondary"][k]["p_holm"] for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(stages))
    w = 0.38
    ax.bar(x - w/2, p_a, w, color=COLOR_A, label="Variant A",
           edgecolor="white")
    ax.bar(x + w/2, p_b, w, color=COLOR_B, label="Variant B",
           edgecolor="white")
    for i, (va, vb) in enumerate(zip(p_a, p_b)):
        ax.text(i - w/2, va + 0.02, f"{va*100:.0f}%", ha="center",
                fontweight="bold")
        ax.text(i + w/2, vb + 0.02, f"{vb*100:.0f}%", ha="center",
                fontweight="bold")
    ax.set_xticks(x)
    # Combine stage name + holm p-value in two-line tick label
    tick_labels = []
    for stage, p in zip(stages, p_holm):
        star = "n.s." if p > 0.05 else ("*" if p > 0.01 else
                                        ("**" if p > 0.001 else "***"))
        tick_labels.append(f"{stage}\n(Holm p = {p:.3g}  {star})")
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Share of users who reached the stage")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Figure 3.  Stage-by-stage reach rates\n"
                 "(Variant B advantage concentrates on Cleaning and Feature Engineering)")
    save(fig, "fig3_stage_funnel")


#  fig4 depth 
def fig4_workflow_depth() -> None:
    # depth 0..3 counts
    max_d = int(df["workflow_depth"].max())
    vals_a = [int(((A["workflow_depth"] == d)).sum()) for d in range(max_d + 1)]
    vals_b = [int(((B["workflow_depth"] == d)).sum()) for d in range(max_d + 1)]

    s = res["secondary"]["workflow_depth"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(max_d + 1)
    w = 0.4
    ax1.bar(x - w/2, vals_a, w, color=COLOR_A, edgecolor="white",
            label="A")
    ax1.bar(x + w/2, vals_b, w, color=COLOR_B, edgecolor="white",
            label="B")
    for i in range(max_d + 1):
        ax1.text(i - w/2, vals_a[i] + 0.4, str(vals_a[i]), ha="center",
                 fontsize=9)
        ax1.text(i + w/2, vals_b[i] + 0.4, str(vals_b[i]), ha="center",
                 fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([str(d) for d in range(max_d+1)])
    ax1.set_xlabel("Workflow depth (# stages reached)")
    ax1.set_ylabel("Number of users")
    ax1.set_title("Workflow-depth distribution")
    ax1.legend(frameon=False)

    # means with 95% CI
    means = [s["mean_A"], s["mean_B"]]
    se = [s["sd_A"] / math.sqrt(s["n_A"]), s["sd_B"] / math.sqrt(s["n_B"])]
    ax2.bar(["A", "B"], means, yerr=[1.96 * e for e in se], capsize=8,
            color=[COLOR_A, COLOR_B], edgecolor="white", width=0.55)
    for i, m in enumerate(means):
        ax2.text(i, m + 0.12, f"{m:.2f}", ha="center", fontweight="bold")
    ax2.set_ylabel("Mean workflow depth")
    ax2.set_title("Means with 95% CIs")
    ax2.set_ylim(0, max(means) * 1.3)

    fig.suptitle(
        f"Figure 4.  Workflow depth   "
        f"(Welch p = {s['p_welch']:.2e};  Cohen's d = {s['cohens_d']:.2f};  "
        f"Holm p = {s['p_holm']:.2e})",
        fontsize=12, y=1.02)
    save(fig, "fig4_workflow_depth")


#  fig5 linear path
def fig5_linear_path() -> None:
    # values 1..4
    vals = sorted(df["linear_path_score"].unique())
    counts_a = [int((A["linear_path_score"] == v).sum()) for v in vals]
    counts_b = [int((B["linear_path_score"] == v).sum()) for v in vals]
    s = res["secondary"]["linear_path_score"]

    # side-by-side bars, plus a stacked-pct to read shift
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(vals))
    w = 0.4
    ax1.bar(x - w/2, counts_a, w, color=COLOR_A, label="A",
            edgecolor="white")
    ax1.bar(x + w/2, counts_b, w, color=COLOR_B, label="B",
            edgecolor="white")
    for i in range(len(vals)):
        ax1.text(i - w/2, counts_a[i] + 0.4, str(counts_a[i]),
                 ha="center", fontsize=9)
        ax1.text(i + w/2, counts_b[i] + 0.4, str(counts_b[i]),
                 ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([str(v) for v in vals])
    ax1.set_xlabel("Linear path score (1=chaotic … 4=textbook order)")
    ax1.set_ylabel("Number of users")
    ax1.set_title("Linear-path-score distribution")
    ax1.legend(frameon=False)

    # stacked percentages
    pct_a = [c / sum(counts_a) * 100 for c in counts_a]
    pct_b = [c / sum(counts_b) * 100 for c in counts_b]
    colors = ["#d62828", "#f77f00", "#06d6a0", "#118ab2"]
    bottom_a = bottom_b = 0
    for i, v in enumerate(vals):
        ax2.barh("A", pct_a[i], left=bottom_a, color=colors[i],
                 label=str(v), edgecolor="white", height=0.5)
        if pct_a[i] > 4:
            ax2.text(bottom_a + pct_a[i]/2, 0, f"{pct_a[i]:.0f}%",
                     ha="center", va="center", color="white",
                     fontweight="bold", fontsize=9)
        ax2.barh("B", pct_b[i], left=bottom_b, color=colors[i],
                 edgecolor="white", height=0.5)
        if pct_b[i] > 4:
            ax2.text(bottom_b + pct_b[i]/2, 1, f"{pct_b[i]:.0f}%",
                     ha="center", va="center", color="white",
                     fontweight="bold", fontsize=9)
        bottom_a += pct_a[i]; bottom_b += pct_b[i]
    ax2.set_xlabel("Share of users (%)")
    ax2.set_xlim(0, 100)
    ax2.legend(title="Path score", bbox_to_anchor=(1.02, 1),
               loc="upper left", frameon=False)
    ax2.set_title("Distribution share")

    fig.suptitle(
        f"Figure 5.  Linear path score   "
        f"(Welch p = {s['p_welch']:.2e};  Cliff's δ = {s['cliffs_delta']:.2f};  "
        f"Holm p = {s['p_holm']:.2e})",
        fontsize=12, y=1.02)
    save(fig, "fig5_linear_path")


# ---------------------------------------------------------- fig6 tab duration
def fig6_tab_duration() -> None:
    s_total = res["secondary"]["total_tab_duration_sec"]
    s_avg   = res["secondary"]["avg_tab_duration_sec"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # total tab duration - violin + box
    a1 = A["total_tab_duration_sec"].to_numpy()
    b1 = B["total_tab_duration_sec"].to_numpy()
    parts = ax1.violinplot([a1, b1], showmeans=False, showmedians=False,
                           showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], [COLOR_A, COLOR_B]):
        pc.set_facecolor(c); pc.set_edgecolor("white"); pc.set_alpha(0.55)
    bp = ax1.boxplot([a1, b1], widths=0.18, patch_artist=True,
                     boxprops=dict(facecolor="white", edgecolor="#333"),
                     medianprops=dict(color="#d62828", lw=2))
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(["Variant A", "Variant B"])
    ax1.set_ylabel("Total time on tabs (seconds)")
    ax1.set_title(f"Total tab duration\n"
                  f"Welch p={s_total['p_welch']:.3f}, "
                  f"Holm p={s_total['p_holm']:.3f}")

    # avg tab duration
    a2 = A["avg_tab_duration_sec"].to_numpy()
    b2 = B["avg_tab_duration_sec"].to_numpy()
    parts = ax2.violinplot([a2, b2], showmeans=False, showmedians=False,
                           showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], [COLOR_A, COLOR_B]):
        pc.set_facecolor(c); pc.set_edgecolor("white"); pc.set_alpha(0.55)
    ax2.boxplot([a2, b2], widths=0.18, patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="#333"),
                medianprops=dict(color="#d62828", lw=2))
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(["Variant A", "Variant B"])
    ax2.set_ylabel("Mean time per tab (seconds)")
    ax2.set_title(f"Avg tab duration\n"
                  f"Welch p={s_avg['p_welch']:.3f}, "
                  f"Holm p={s_avg['p_holm']:.3f}")

    fig.suptitle("Figure 6.  Engagement depth — tab duration",
                 fontsize=12, y=1.02)
    save(fig, "fig6_tab_duration")


# fig7 forest 
def fig7_logreg_forest() -> None:
    ors = res["logreg"]["odds_ratios"]
    pretty = {
        "variant_B":     "Variant B (vs A)",
        "button_clicks": "Button clicks (z-scored)",
        "tab_switches":  "Tab switches (z-scored)",
        "scroll_count":  "Scroll count (z-scored)",
    }
    cols = list(ors.keys())
    pt = np.array([ors[c]["OR"] for c in cols])
    lo = np.array([ors[c]["CI_lo"] for c in cols])
    hi = np.array([ors[c]["CI_hi"] for c in cols])

    # sort ascending for readability
    order = np.argsort(pt)
    pt, lo, hi = pt[order], lo[order], hi[order]
    labels = [pretty[cols[i]] for i in order]

    # clip bootstrap CI arms to cap them visually at 100 (the upper CI
    # for variant_B can be astronomical on this small n); plotting log
    # scale handles dispersion fine as long as the CI isn't negative.
    xerr_lo = np.maximum(pt - lo, 0)
    xerr_hi = np.maximum(hi - pt, 0)

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4),
                                  gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(labels))
    is_variant = ["Variant B" in l for l in labels]
    colors = [COLOR_B if v else "#475569" for v in is_variant]

    # cap the upper CI display at 1000 for visual sanity (we report the
    # true number in the text panel on the right and in the title).
    cap = 1000.0
    hi_disp = np.minimum(hi, cap)
    xerr_hi_disp = np.maximum(hi_disp - pt, 0)

    for i in range(len(labels)):
        ax.errorbar(pt[i], y[i], xerr=[[xerr_lo[i]], [xerr_hi_disp[i]]],
                    fmt="o", capsize=4, ecolor=colors[i], color=colors[i],
                    markersize=9, markerfacecolor=colors[i],
                    markeredgecolor=colors[i], lw=1.8)
        # arrow if the true upper CI is past the cap
        if hi[i] > cap:
            ax.annotate("", xy=(cap * 1.4, y[i]), xytext=(cap, y[i]),
                        arrowprops=dict(arrowstyle="->", color=colors[i],
                                        lw=1.8))
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio  (95% bootstrap CI; arrows = CI past axis)")
    ax.set_xlim(0.3, cap * 2)

    # Right-hand text panel with exact ORs and CIs
    axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(-0.5, len(labels) - 0.5)
    axt.text(0.0, len(labels) - 0.4, "Predictor", fontweight="bold",
             fontsize=10)
    axt.text(0.55, len(labels) - 0.4, "OR  [95% CI]", fontweight="bold",
             fontsize=10)
    for i in range(len(labels)):
        axt.text(0.0, y[i], labels[i], va="center", fontsize=9.5)
        axt.text(0.55, y[i],
                 f"{pt[i]:6.2f}  [{lo[i]:5.2f}, {hi[i]:7.2f}]",
                 va="center", family="monospace", fontsize=9.5,
                 color="#1e293b")

    or_b = ors["variant_B"]
    fig.suptitle(
        f"Figure 7.  Logistic regression — predictors of full-workflow completion\n"
        f"Adjusted OR for Variant B = {or_b['OR']:.1f}  "
        f"(95% bootstrap CI [{or_b['CI_lo']:.1f}, {or_b['CI_hi']:.1f}])",
        fontsize=12, y=1.04
    )
    save(fig, "fig7_logreg_forest")


#  fig8 funnel 
def fig8_conversion_funnel() -> None:
    # 4 stages: arrived -> >=1 stage reached -> >=2 stages -> all 3
    stages = ["Arrived", "Reached\n>=1 stage",
              "Reached\n>=2 stages", "Completed\nfull workflow"]

    def counts(g):
        n = len(g)
        s1 = int(((g["reached_eda"] | g["reached_cleaning"]
                   | g["reached_feature_eng"]) > 0).sum())
        s2 = int(g["completed_two_or_more"].sum())
        s3 = int(g["completed_full_workflow"].sum())
        return [n, s1, s2, s3]

    cA, cB = counts(A), counts(B)
    pctA = [c / cA[0] * 100 for c in cA]
    pctB = [c / cB[0] * 100 for c in cB]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(stages))
    w = 0.38
    ax.bar(x - w/2, cA, w, color=COLOR_A, label="Variant A",
           edgecolor="white")
    ax.bar(x + w/2, cB, w, color=COLOR_B, label="Variant B",
           edgecolor="white")
    for i in range(len(stages)):
        ax.text(i - w/2, cA[i] + 1, str(cA[i]), ha="center",
                fontweight="bold")
        ax.text(i + w/2, cB[i] + 1, str(cB[i]), ha="center",
                fontweight="bold")
        if i > 0:
            ax.text(i - w/2, -3.5, f"{pctA[i]:.0f}%", ha="center",
                    color=COLOR_A, fontweight="bold", fontsize=9)
            ax.text(i + w/2, -3.5, f"{pctB[i]:.0f}%", ha="center",
                    color=COLOR_B, fontweight="bold", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(-8, max(cA + cB) + 8)
    ax.set_ylabel("Number of users")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Figure 8.  Conversion funnel by variant\n"
                 "(percentages relative to total arrivals in that arm)")
    save(fig, "fig8_conversion_funnel")


def main():
    fig1_ga4_events()
    fig2_primary()
    fig3_stage_funnel()
    fig4_workflow_depth()
    fig5_linear_path()
    fig6_tab_duration()
    fig7_logreg_forest()
    fig8_conversion_funnel()


if __name__ == "__main__":
    main()