# LLM-CoRe

This repository contains the equation-search code and data required to reproduce the experiments in the LLM-CoRe manuscript.

## Environment

The tested environment uses Python 3.12. Install the pinned dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Equation search

Copy the environment-variable template and add your API key:

```bash
cp .env.sample .env
```

Then run:

```bash
./search.sh
```

Search results are written to `logs/search/<exp_name>/`. This directory contains `args.json`, `info.log`, per-generation records, checkpoints, and the final `best_model.py`. Files under `logs/` are runtime outputs and are excluded from version control.

## Main-text figures

The notebooks in `scripts/` generate the data panels in main-text Figures 2--4. Run them from the repository root. Outputs are saved to `logs/scripts/<notebook-name>/`, with filenames that correspond directly to the manuscript panels:

- `figure2_deterministic.ipynb`: `Fig2a-c-e`, `Fig2b-d-f-1`, and `Fig2b-d-f-2`;
- `figure2_stochastic.ipynb`: `Fig2g-h`;
- `figure3_co2_attribution.ipynb`: `Fig3a` and `Fig3b`;
- `figure3_interactions.ipynb`: `Fig3c`, `Fig3d`, and `Fig3e`;
- `figure3_event_leads.ipynb`: `Fig3f`;
- `figure3_event_diagnostics.ipynb`: `Fig3g` and `Fig3h`;
- `figure4_counterfactual.ipynb`: `Fig4a`, `Fig4b`, `Fig4c`, and `Fig4d`;
- `figure4_phase_space.ipynb`: `Fig4e`, `Fig4f`, `Fig4g`, and `Fig4h`.

## Data directory

- `data/*.csv`: observational and external-forcing time series;
- `data/baseline/<baseline-name>/*.csv`: exported predictions from ORCA-DL, UniCM, and Physics-DESN;
- `data/cache/default_args.json`: default model-loading arguments used by the plotting notebooks;
- `data/cache/figure2_deterministic.pkl.xz`: prediction details used by the deterministic panels in Figure 2;
- `data/cache/counterfactual.csv`: statistics and confidence intervals used by the counterfactual experiments in Figure 4.
