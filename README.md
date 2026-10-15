# LLM-CoRe

Code and source data for the LLM-CoRe manuscript.

## Environment

The tested environment uses Python 3.12. Create the local environment and install the pinned dependencies with:

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

`nd2py` is installed directly from a pinned Git commit. It is not installed in editable mode.

## Equation search

Add the required API credentials to `.env`, then run:

```bash
./search.sh
```

The script contains the search configuration reported in the manuscript: the `only_co2` setting, 200 individuals, 300 iterations, eight generated programs per prompt, five-fold cross-validation, and the reported objective weights. Extra command-line arguments can be appended for diagnostic runs. For example, the following performs a one-individual, one-iteration run without an LLM call:

```bash
./search.sh --exp_name smoke-test --no_llm \
  --seed_individuals XRO --use_seed_individuals_only \
  --population_size 1 --n_iter 1 --num_workers 1 --no-parallel \
  --max_month 2 --a_std 0 --a_skewness 0 --a_spectrum 0
```

Search outputs are written below `logs/search/`.

## Main-text figures

The notebooks in `scripts/` reproduce the data panels used in Figures 2--4. Run them from the repository root with the `Python (LLM-CoRe)` kernel, or execute one non-interactively with:

```bash
venv/bin/python -m jupyter nbconvert --to notebook --execute \
  scripts/figure2_deterministic.ipynb \
  --output figure2_deterministic.ipynb --output-dir scripts \
  --ExecutePreprocessor.timeout=1800
```

Each notebook writes figures to `logs/scripts/<notebook-name>/`. The pairwise interaction ablation in `figure3_interactions.ipynb` is the slowest step (about 20 minutes on the test machine) and caches its predictions in that output directory. Figure 1 is a schematic assembled in presentation software and has no source plotting notebook.

- `figure2_deterministic.ipynb`: `Fig2a-c-e` plus `Fig2b-d-f-1` and `Fig2b-d-f-2`, which jointly form panels b, d, and f.
- `figure2_stochastic.ipynb`: `Fig2g-h`.
- `figure3_co2_attribution.ipynb`, `figure3_interactions.ipynb`, `figure3_event_leads.ipynb`, and `figure3_event_diagnostics.ipynb`: Figure 3.
- `figure4_counterfactual.ipynb` and `figure4_phase_space.ipynb`: Figure 4.

Output filenames follow the manuscript panels: for example, `Fig3a.{svg,png,pdf}` is panel 3a and `Fig4a-c.{svg,png,pdf}` contains panels 4a--c. A filename such as `Fig2a-c-e` contains the non-adjacent panels listed in its name.

The notebooks use compact exported metrics for the counterfactual panels, baseline results in `data/baseline/<model>/`, and a compressed prediction-detail file for Figure 2. This avoids distributing the legacy multi-gigabyte simulation cache while retaining the experiment code needed to regenerate the expensive interaction analysis.
