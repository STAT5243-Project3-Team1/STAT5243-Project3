# STAT5243 Project 3

## Overview
This project investigates whether a guided redesign improves user success in an interactive data workbench using a randomized A/B test. The redesign introduces inline prompts that guide users through a structured workflow (EDA → Cleaning → Feature Engineering).

---

## Key Results

- Variant B significantly increased full workflow completion rates  
- Two-proportion test: **p < 0.001**  
- Large effect size (**Cohen’s h ≈ 1.17**)  
- Logistic regression indicates a strong treatment effect (**OR ≈ 36.9**)  
- Improvements are consistent across multiple engagement metrics  

**Conclusion:** Guided UX substantially improves user success in multi-step analytical workflows.

---

## Data

- Source: Google Analytics 4 (GA4) event-level data  
- Unit: user-level aggregated dataset  
- Sample size: ~60+ users  
- Includes engagement metrics, workflow progression, and interaction logs  

---

## Methods

- A/B test with randomized assignment  
- Primary outcome: full workflow completion  

Statistical methods:
- Two-proportion z-test  
- Welch’s t-test  
- Mann–Whitney U test  
- Holm-adjusted multiple testing correction  
- Logistic regression for robustness analysis  

---

## Project Structure

- `01_load_data.py` — data loading and preprocessing  
- `02_statistical_analysis.py` — hypothesis testing and analysis  
- `03_make_figures.py` — visualization  
- `report.tex` — final report  

---

## Figures

Generated via `03_make_figures.py`, including:
- Conversion funnel  
- Workflow depth distribution  
- Tab duration comparisons  
- Logistic regression forest plot  

---

## Reproducibility

Run the full pipeline:

```bash
python3 01_load_data.py
python3 02_statistical_analysis.py
python3 03_make_figures.py
pdflatex report.tex
