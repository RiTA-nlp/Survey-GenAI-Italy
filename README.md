## Code and data for the paper "Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context"

This repository hosts the datasets, exploratory analyses, and reproducible code that accompany the research paper referenced above. All materials in this branch are anonymized and self-contained so that reviewers can reproduce the main findings without needing external references.

## Repository Structure

- `requirements.txt` — Python dependencies required to run the notebooks and scripts.
- `survey_v3_full.tsv` - Raw survey data that is processed by our processing pipeline.
- `survey_v3_clean.tsv` — Main survey dataset after cleaning and validation.
- `survey_variables.tsv` — Codebook describing survey questions and derived variables.
- `figures/` — Static figures exported from the analysis notebooks.
- `notebooks/` — Jupyter notebooks used for data preparation, analysis, and visualization (see detailed descriptions below).
- `outputs/` — Intermediate artifacts such as topic model assignments produced during text analysis.
- `results/` — Key summary tables referenced in the manuscript.
- `scripts/` — Standalone Python scripts supporting the analysis pipeline.
- `survey_and_translation/` — Supplementary reference material for the survey instrument and translations.

## Getting Started

1. **Clone the repository**
	```bash
	git clone <repository-url>
	cd rita_survey
	```
2. **Create and activate a virtual environment** (recommended)
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

1. Inspect the raw and cleaned survey data in `survey_v3_clean.tsv` using the descriptive information in `survey_variables.tsv`.
2. Run the notebooks in numerical order to reproduce preprocessing, modeling, robustness checks, and visualizations.
3. Consult the outputs in `results/` and `figures/` for tables and graphics cited in the paper.
4. Use scripts in `scripts/` for command-line automation of selected steps (e.g., mapping occupational categories).

## Notebooks

- `0_preliminary_filters.ipynb` — Applies inclusion criteria, removes invalid responses, and documents the resulting analytic sample.
- `1_data_analysis.ipynb` — Contains the main descriptive statistics, regression models, and hypothesis tests reported in the paper.
- `2_robustenss_check.ipynb` — Implements sensitivity analyses; reruns key models under alternative specifications to verify stability. *(Filename retains the original spelling.)*
- `3_data_viz.ipynb` — Generates publication-ready figures and stores them in `figures/`.
- `multicollinearity.ipynb` — Diagnoses multicollinearity among covariates using variance inflation factors and correlation diagnostics.
- `opentext_fields.ipynb` — Preprocesses free-text survey responses, including cleaning, tokenization, and language-specific handling.
- `stopwords_it.txt` — Companion resource listing domain-specific stopwords used in the text analysis workflow.

Each notebook is designed to be run top-to-bottom. Before execution, ensure the working directory is set to the repository root so relative paths resolve correctly. Notebook outputs in `outputs/` and `results/` are cached to speed up review; rerunning notebooks will overwrite these files.

## Data and Outputs

- `outputs/opentext*/` — Topic model labels and assignments for open-ended survey questions.
- `results/*.tsv` — Aggregated regression summaries, average marginal effects, and descriptive tables aligned with manuscript tables.

## Scripts

- `scripts/map_jobs.py` — Harmonizes occupation-related responses, mapping free-text entries to standardized categories leveraged in the main analysis. Run from the repository root to ensure relative file paths resolve.

## Reproducing Key Tables and Figures

1. Execute `0_preliminary_filters.ipynb` through `3_data_viz.ipynb` sequentially.
2. Exported tables will populate `results/`; figures will appear in `figures/`.
3. Cross-check outputs against the manuscript to verify alignment with reported numbers.