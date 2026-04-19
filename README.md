# STAT5243 Project 3

## Overview
This project investigates whether a guided redesign improves user success in an interactive data workbench using an A/B test.

## Structure
- `01_load_data.py`: data loading
- `02_statistical_analysis.py`: hypothesis testing and analysis
- `03_make_figures.py`: visualization
- `report.tex`: final report

## Reproducibility
```bash
python3 01_load_data.py
python3 02_statistical_analysis.py
python3 03_make_figures.py
pdflatex report.tex

## Key Results

- Variant B significantly improved full workflow completion rate  
- Two-proportion test: p < 0.001  
- Large effect size (Cohen’s h ≈ 1.17)  
- Logistic regression shows strong positive effect of Variant B (OR ≈ 36.9)

Overall, guided UX substantially increases user success.

## Data

- Source: Google Analytics 4 (GA4) event-level data  
- Unit: user-level aggregated dataset  
- Size: ~60+ users, session-level tracking  
- Includes engagement metrics, workflow progression, and interaction logs

## Methods

- A/B test with randomized assignment  
- Primary outcome: full workflow completion  
- Statistical tests:
  - Two-proportion z-test
  - Welch’s t-test
  - Mann–Whitney U test
- Multiple testing controlled via Holm adjustment  
- Logistic regression for robustness analysis

## Figures

All figures are generated via `03_make_figures.py`, including:
- Conversion funnel
- Workflow depth
- Tab duration distributions
- Logistic regression forest plot

## Repository

Full project available at:  
https://github.com/STAT5243-Project3-Team1/STAT5243-Project3
