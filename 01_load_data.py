"""
01_load_data.py

Load the real user-level A/B test data and derive analysis-ready
outcomes. Event counts were collected by a Google-Analytics-4
stream attached to the Shiny app (Project 2) between 2026-04-09
and 2026-04-17. A companion GA4 event export (ga4_real_data.csv)
is retained for cross-validation of event totals. Additionally, 
a simulated dataset was constructed to validate the analysis 
pipeline under limited sample size scenarios.

Primary outcome: full-workflow completion
    completed_full_workflow = 1 iff the user reached ALL THREE tabs
    (EDA, Cleaning, Feature Engineering) during their session.
This is the single most faithful operationalisation of "did the user
finish the data-analysis task the Workbench was built for".

Secondary outcomes:
    workflow_depth          (ordinal 0-3)      breadth of progression
    linear_path_score       (ordinal 1-4)      adherence to suggested path
    session_duration_sec    (continuous)       total time on task
    total_tab_duration_sec  (continuous)       active engagement time
    avg_tab_duration_sec    (continuous)       depth of engagement per tab
    button_clicks           (count)            interaction intensity
    guided_clicks           (count)            use of Variant-B-specific
                                              guided prompts (A always 0)

Outputs:  analysis_df.parquet  and  analysis_df.csv
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path  # 新增：用于处理路径

# 核心修改：获取脚本所在的绝对目录
SCRIPT_DIR = Path(__file__).parent.absolute()
# 基于脚本目录拼接输入/输出路径（相对路径改为绝对路径）
SRC = SCRIPT_DIR / "user_level_data.csv"
OUT_CSV = SCRIPT_DIR / "analysis_df.csv"


def main() -> None:
    df = pd.read_csv(SRC)

    #  derived primary outcome 
    df["completed_full_workflow"] = (
        (df["reached_eda"] == 1)
        & (df["reached_cleaning"] == 1)
        & (df["reached_feature_eng"] == 1)
    ).astype(int)

    # partial completion — reached at least two of three stages
    df["completed_two_or_more"] = (
        df["reached_eda"] + df["reached_cleaning"] + df["reached_feature_eng"] >= 2
    ).astype(int)

    # sanity checks 
    print("Loaded", len(df), "user-level rows.")
    print(df["ab_version"].value_counts().rename("n").to_frame(), "\n")
    print("Completion rates by arm:")
    print(df.groupby("ab_version")[["completed_full_workflow",
                                    "completed_two_or_more"]]
                .mean().round(3), "\n")
    print("Outcome means by arm:")
    means = df.groupby("ab_version")[[
        "workflow_depth", "linear_path_score", "session_duration_sec",
        "total_tab_duration_sec", "avg_tab_duration_sec",
        "button_clicks", "tab_switches", "guided_clicks",
    ]].mean().round(2).T
    print(means, "\n")

    df.to_csv(OUT_CSV, index=False)
    print("Saved ->", OUT_CSV)


if __name__ == "__main__":
    main()