"""
02_statistical_analysis.py

Statistical analysis for the real A/B test data collected via GA4 on
the Interactive Data Workbench (Shiny for Python) between
2026-04-09 and 2026-04-17.

  (a) Randomization / activity-parity check (button_clicks, scroll_count,
      tab_switches)  these proxy exposure rather than demographics
      but the test is still informative.
  (b) Two-proportion z-test + Fisher exact for full-workflow completion.
  (c) Welch's t-test + Mann-Whitney U for continuous secondary outcomes.
  (d) Effect sizes: Cohen's h (proportions), Cohen's d (means),
      Cliff's delta (non-parametric location shift).
  (e) Holm-Bonferroni adjustment over the family of secondary tests.
  (f) Logistic regression of primary outcome on variant + activity
      covariates with 800-resample bootstrap CIs.
  (g) Post-hoc power analysis for the primary outcome.
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path  # 新增：用于处理路径

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# 核心修改：获取脚本所在的绝对目录
SCRIPT_DIR = Path(__file__).parent.absolute()
# 基于脚本目录拼接输入/输出路径
DATA = SCRIPT_DIR / "analysis_df.csv"
OUT = SCRIPT_DIR / "results.json"
ALPHA = 0.05


# - helpers 
def two_prop_ztest(x1: int, n1: int, x2: int, n2: int):
    """Pooled two-proportion z-test. Returns (z, p, diff, 95% CI)."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else float("inf")
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_d = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    ci = (diff - 1.96 * se_d, diff + 1.96 * se_d)
    return z, p, diff, ci


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (x2.mean() - x1.mean()) / sp if sp > 0 else float("nan")


def cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]."""
    n1, n2 = len(x1), len(x2)
    gt = lt = 0
    for a in x2:
        for b in x1:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    return (gt - lt) / (n1 * n2) if n1 * n2 > 0 else float("nan")


def welch_ci_diff(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.05):
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    diff = m2 - m1
    return diff, (diff - tcrit * se, diff + tcrit * se), df


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def power_two_prop(p1: float, p2: float, n1: int, n2: int,
                   alpha: float = 0.05) -> float:
    h = abs(cohens_h(p1, p2))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = h * math.sqrt((n1 * n2) / (n1 + n2)) - z_alpha
    return float(stats.norm.cdf(z_beta))


# main 
def main() -> None:
    df = pd.read_csv(DATA)
    A = df[df["ab_version"] == "A"]
    B = df[df["ab_version"] == "B"]
    n_A, n_B = len(A), len(B)
    res: dict = {"alpha": ALPHA, "n_A": n_A, "n_B": n_B,
                 "n_total": n_A + n_B}

    #  (a) activity-parity check 
    # No demographics available; test whether passive activity proxies
    # look similar across arms (not a demographic balance check  see §).
    res["balance"] = {}
    for col in ["button_clicks", "tab_switches", "scroll_count"]:
        t, p = stats.ttest_ind(A[col], B[col], equal_var=False)
        res["balance"][col] = {
            "t": float(t), "p": float(p),
            "mean_A": float(A[col].mean()),
            "mean_B": float(B[col].mean()),
        }

    #  (b) PRIMARY: full-workflow completion 
    x1 = int(A["completed_full_workflow"].sum())
    x2 = int(B["completed_full_workflow"].sum())
    z, p_prim, diff, ci = two_prop_ztest(x1, n_A, x2, n_B)
    fisher_or, fisher_p = stats.fisher_exact(
        [[x2, n_B - x2], [x1, n_A - x1]]
    )
    p1, p2 = x1 / n_A, x2 / n_B
    res["primary"] = {
        "outcome": "completed_full_workflow",
        "definition": "reached EDA AND Cleaning AND Feature Engineering",
        "p_A": p1, "p_B": p2, "x_A": x1, "x_B": x2,
        "diff": diff, "ci": list(ci), "z": z, "p_value": p_prim,
        "fisher_OR": float(fisher_or), "fisher_p": float(fisher_p),
        "rel_lift_pct": (p2 - p1) / p1 * 100 if p1 > 0 else float("inf"),
        "cohens_h": cohens_h(p1, p2),
        "post_hoc_power": power_two_prop(p1, p2, n_A, n_B),
    }

    #  (c) secondary continuous/ordinal outcomes 
    cont_outcomes = {
        "workflow_depth":         "Workflow depth (0-3)",
        "linear_path_score":      "Linear path score (1-4)",
        "session_duration_sec":   "Session duration (s)",
        "total_tab_duration_sec": "Total tab duration (s)",
        "avg_tab_duration_sec":   "Avg tab duration (s)",
    }
    sec = {}
    raw_pvals = []
    fam_keys = []

    for col, label in cont_outcomes.items():
        a = A[col].to_numpy(dtype=float)
        b = B[col].to_numpy(dtype=float)
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        cd = cliffs_delta(a, b)
        diff_m, ci_m, df_w = welch_ci_diff(a, b)
        sec[col] = {
            "label": label,
            "n_A": len(a), "n_B": len(b),
            "mean_A": float(a.mean()), "mean_B": float(b.mean()),
            "sd_A": float(a.std(ddof=1)), "sd_B": float(b.std(ddof=1)),
            "median_A": float(np.median(a)),
            "median_B": float(np.median(b)),
            "t": float(t), "df_welch": float(df_w), "p_welch": float(p_t),
            "U": float(u), "p_mwu": float(p_u),
            "cohens_d": float(d), "cliffs_delta": float(cd),
            "diff": float(diff_m), "ci": [float(ci_m[0]), float(ci_m[1])],
        }
        raw_pvals.append(p_t); fam_keys.append(col)

    # completed >=2 of 3 stages (partial success)
    x1b = int(A["completed_two_or_more"].sum())
    x2b = int(B["completed_two_or_more"].sum())
    zb, pb, diff_b, ci_b = two_prop_ztest(x1b, n_A, x2b, n_B)
    _, fisher_p_b = stats.fisher_exact(
        [[x2b, n_B - x2b], [x1b, n_A - x1b]]
    )
    sec["completed_two_or_more"] = {
        "label": "Reached >=2 stages",
        "p_A": x1b / n_A, "p_B": x2b / n_B,
        "x_A": x1b, "x_B": x2b,
        "diff": diff_b, "ci": list(ci_b),
        "z": zb, "p_welch": float(pb),
        "fisher_p": float(fisher_p_b), "p_mwu": float(pb),
        "cohens_h": cohens_h(x1b / n_A, x2b / n_B),
    }
    raw_pvals.append(pb); fam_keys.append("completed_two_or_more")

    # individual tab reach rates
    for col, label in [
        ("reached_eda",         "Reached EDA"),
        ("reached_cleaning",    "Reached Cleaning"),
        ("reached_feature_eng", "Reached Feature Engineering"),
    ]:
        x1c = int(A[col].sum()); x2c = int(B[col].sum())
        zc, pc, dc, ci_c = two_prop_ztest(x1c, n_A, x2c, n_B)
        _, fisher_p_c = stats.fisher_exact(
            [[x2c, n_B - x2c], [x1c, n_A - x1c]]
        )
        sec[col] = {
            "label": label,
            "p_A": x1c / n_A, "p_B": x2c / n_B,
            "x_A": x1c, "x_B": x2c,
            "diff": dc, "ci": list(ci_c),
            "z": zc, "p_welch": float(pc),
            "fisher_p": float(fisher_p_c), "p_mwu": float(pc),
            "cohens_h": cohens_h(x1c / n_A, x2c / n_B),
        }
        raw_pvals.append(pc); fam_keys.append(col)

    #  (d) Holm-Bonferroni over the whole secondary family 
    adj = holm_bonferroni(raw_pvals)
    for k, p_a in zip(fam_keys, adj):
        sec[k]["p_holm"] = float(p_a)
    res["secondary"] = sec

    #  (e) logistic regression 
    feat_df = pd.DataFrame({
        "variant_B":    (df["ab_version"] == "B").astype(int),
        "button_clicks": df["button_clicks"].astype(float),
        "tab_switches":  df["tab_switches"].astype(float),
        "scroll_count":  df["scroll_count"].astype(float),
    })
    for c in ["button_clicks", "tab_switches", "scroll_count"]:
        mu, sd = feat_df[c].mean(), feat_df[c].std()
        feat_df[c] = (feat_df[c] - mu) / (sd if sd > 0 else 1.0)

    X = feat_df.to_numpy(dtype=float)
    y = df["completed_full_workflow"].to_numpy()
    cols = feat_df.columns.tolist()

    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X, y)
    coefs = dict(zip(cols, model.coef_.flatten().tolist()))
    intercept = float(model.intercept_[0])

    rng = np.random.default_rng(7)
    boot_coefs = []
    n = len(y)
    for _ in range(800):
        idx = rng.integers(0, n, size=n)
        try:
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            m.fit(X[idx], y[idx])
            boot_coefs.append(m.coef_.flatten())
        except Exception:
            continue
    boot_coefs = np.array(boot_coefs)

    ors = {}
    for j, c in enumerate(cols):
        pt = math.exp(coefs[c])
        lo = math.exp(np.percentile(boot_coefs[:, j], 2.5))
        hi = math.exp(np.percentile(boot_coefs[:, j], 97.5))
        ors[c] = {"OR": pt, "CI_lo": lo, "CI_hi": hi, "beta": coefs[c]}

    res["logreg"] = {
        "intercept": intercept,
        "coefs": coefs,
        "odds_ratios": ors,
        "n_boot": int(len(boot_coefs)),
        "features": cols,
        "accuracy": float(model.score(X, y)),
    }

    #  descriptive table 
    desc = df.groupby("ab_version").agg(
        n=("user_id", "count"),
        completion_full=("completed_full_workflow", "mean"),
        completion_2plus=("completed_two_or_more", "mean"),
        workflow_depth=("workflow_depth", "mean"),
        linear_path=("linear_path_score", "mean"),
        session_sec=("session_duration_sec", "mean"),
        total_tab_sec=("total_tab_duration_sec", "mean"),
        avg_tab_sec=("avg_tab_duration_sec", "mean"),
        button_clicks=("button_clicks", "mean"),
        tab_switches=("tab_switches", "mean"),
        guided_clicks=("guided_clicks", "mean"),
    ).round(3)
    res["desc_table"] = desc.to_dict()

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"Wrote {OUT}")
    print(json.dumps({"primary": res["primary"]}, indent=2, default=float))
    print("\nSecondary family (Holm-adjusted):")
    for k in fam_keys:
        print(f"  {k:<28s}  p_raw={sec[k]['p_welch']:.4g}   "
              f"p_holm={sec[k]['p_holm']:.4g}")
    print("\nLogistic regression ORs:")
    for c, v in ors.items():
        print(f"  {c:<16s}  OR={v['OR']:.2f}  "
              f"[{v['CI_lo']:.2f}, {v['CI_hi']:.2f}]")


if __name__ == "__main__":
    main()