# Survey-GenAI-Italy
Data and code associated with the paper "Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context"

## 📋 About
This repository contains the survey data, analysis code, and results associated with our comprehensive empirical study of GenAI chatbot adoption, usage patterns, and literacy in Italy.

**Survey period**: May–August 2025  
**Sample size**: 1,906 Italian-speaking adults

## 📖 Citation

```bibtex
@article{savoldi2025generativeaipracticesliteracy,
      title={Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context}, 
      author={Beatrice Savoldi and Giuseppe Attanasio and Olga Gorodetskaya and Marta Marchiori Manerba and Elisa Bassignana and Silvia Casola and Matteo Negri and Tommaso Caselli and Luisa Bentivogli and Alan Ramponi and Arianna Muti and Nicoletta Balbo and Debora Nozza},
      year={2025},
      eprint={2512.03671},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.03671}, 
}
```

## Repository Structure

- `requirements.txt` — Python dependencies required to run the notebooks and scripts.
- `survey_full.tsv` — Raw survey data that is processed by our processing pipeline.
- `survey_clean.tsv` — Intermediate survey dataset version after cleaning and validation.
- `survey_clean_var.tsv` — Main survey dataset with key derived variables.
- `figures/` — Static figures exported from the analysis notebooks.
- `notebooks/` — Jupyter notebooks used for data preparation, analysis, and visualization (see detailed descriptions below).
- `results/` — Key summary tables and regression outputs referenced in the manuscript.
- `scripts/` — Standalone Python scripts.
- `src/` — Python modules with shared data transformation functions.
- `survey_and_translation/` — Supplementary reference material for the survey instrument and translations.

## Getting Started

1. **Clone the repository**
	```bash
	git clone <repository-url>
	cd rita_survey
	```
2. **Create and activate a virtual environment** (recommended)

	One example using python's venv is below, but feel free to use your favorite package manager (virtualenv, conda, uv, etc.):
	
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
4. **Set up notebook extensions (optional)** — If you plan to run notebooks, enable any preferred Jupyter extensions locally.

## Workflow Overview

1. Inspect the raw and cleaned survey data in `survey_clean.tsv` and `survey_full.tsv` using the descriptive information in `survey_clean_var.tsv`.
2. Run the notebooks in numerical order to reproduce preprocessing, modeling, robustness checks, and visualizations.
3. Consult the outputs in `results/` and `figures/` for tables and graphics cited in the paper.
4. Use scripts in `scripts/` for command-line automation of selected steps.

## Notebooks and Scripts

The notebooks should be run in the following order:

- `0_preliminary_filters.ipynb` — Applies inclusion criteria, removes invalid responses, and documents the resulting analytic sample.
- `1_data_analysis.ipynb` — Exploratory data analysis with descriptive statistics, distributions, and initial insights.
- `2_data_preparation.ipynb` — Feature engineering and creation of derived variables for final analysis, vizualization, checks and statistical modeling.
- `3_robustness_check.ipynb` — Implements sensitivity analyses; reruns key models under alternative specifications to verify stability. 
- `4_modeling.ipynb` — Fits logistic and OLS regression models, computes average marginal effects, and exports results to `results/`.
- `5_data_viz.ipynb` — Generates publication-ready figures and stores them in `figures/`.

Additional exploratory notebooks and scripts:

- `opentext_fields.ipynb` — Preprocesses free-text survey responses, including cleaning, tokenization, and language-specific handling.
- `stopwords_it.txt` — Companion resource listing domain-specific stopwords used in the text analysis workflow.
- `src/test_multicollinearity.py` — Diagnoses multicollinearity among covariates using variance inflation factors and correlation diagnostics.

Each notebook is designed to be run top-to-bottom. Before execution, ensure the working directory is set to the repository root so relative paths resolve correctly.

## Data and Outputs

- `results/*.tsv` — Regression summaries, odds ratios, average marginal effects, decomposition analyses, and demographic tables aligned with manuscript tables.
- `results/*.tex` — LaTeX-formatted tables for direct inclusion in the manuscript.
- `figures/` — Publication-ready visualizations in PNG and PDF formats.

## Scripts

- `scripts/anonymize.py` — Utility for anonymizing survey data (already applied to datasets in this repository).

Run scripts from the repository root to ensure relative file paths resolve correctly.

## Reproducing Key Tables and Figures

1. Execute notebooks `0_preliminary_filters.ipynb` through `5_data_viz.ipynb` sequentially.
2. Regression outputs and summary tables will populate `results/`.
3. Publication-ready figures will appear in `figures/`.