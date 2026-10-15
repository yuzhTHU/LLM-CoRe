#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

exec "$ROOT/venv/bin/python" search.py \
  --name reported-search \
  --setting only_co2 \
  --llm_model gpt-4o-mini \
  --train_period 1979-01 2024-12 \
  --test_period 1979-01 2024-12 \
  --no-use_z20_as_ohc \
  --no-use_l3yo_cv \
  --use_nfold_cv \
  --max_month 30 \
  --damp 0 \
  --population_size 200 \
  --n_iter 300 \
  --programs_per_prompt 2 \
  --generate_per_prompt 8 \
  --elitism_k 10 \
  --expire_iter 1 \
  --temperature_init 0.1 \
  --temperature_period 100 \
  --depth_range 2 6 \
  --tournament_size 20 \
  --p_crossover 0.90 \
  --p_subtree_mutation 0.05 \
  --p_hoist_mutation 0.05 \
  --score_func tanh \
  --a_skill_cv_sum 0.0 \
  --a_ENSO_skill_cv 1.0 \
  --a_ATL3_skill_cv 0.0 \
  --a_SASD_skill_cv 0.0 \
  --a_Total_skill_cv 10.0 \
  --a_length 0.3 \
  --a_params 0.0 \
  --a_std 0.3 \
  --a_skewness 0.3 \
  --a_spectrum 0.3 \
  --timeout 120 \
  "$@"
