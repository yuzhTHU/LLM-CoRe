import re
import os
import time
import json
import gzip
import pickle
import logging
import inspect
import textwrap
import traceback
import nd2py as nd
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.rich import tqdm
from argparse import Namespace
from numpy.random import default_rng
from sklearn.base import BaseEstimator, RegressorMixin
from typing import List, Dict, Tuple, Optional, Generator
from nd2py.search.llmsr import render_python, render_markdown
from .assistant import Assistant
from .core import Individual, Triplet
from .tools import GPOperator, LLMProvider, FitnessEvaluator, GPLearnGenerator
from .utils import jsonalize, tag2ansi, my_uuid, my_hash, log_exception, GenWrapper

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

_logger = logging.getLogger(__name__)


class AIScientist(BaseEstimator, RegressorMixin):
    """LLM enhanced Genetic Programming for regression tasks."""
    def __init__(
        self,
        args: Namespace,
        assistant: Assistant,
        variables: nd.Variable,
        binary: List[nd.Symbol] = [nd.Add, nd.Sub, nd.Mul, nd.Div],
        unary: List[nd.Symbol] = [nd.Sin, nd.Cos, nd.Exp, nd.Log, nd.Sqrt, nd.Tan],
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        seed_individuals: List[Individual] = [],
    ):  
        self.args = args
        self.assistant = assistant
        self.generator = GPLearnGenerator(args, assistant, variables, binary, unary, edge_list, num_nodes)
        self.llm = LLMProvider(args, assistant)
        self.gp = GPOperator(args, assistant, self.generator)
        self.evaluator = FitnessEvaluator(args, assistant, edge_list, num_nodes)
        self.seed_individuals = seed_individuals
        self.variables = variables
        self._rng = np.random.default_rng(seed := args.random_state + my_hash("AIScientist"))
        _logger.debug(f"AIScientist initialized with RNG seed: {seed}, initial state: {self._rng.bit_generator.state}")
        self.records = []
        self.population = []
        self.best_model = None
        self.speed_timer = nd.utils.Timer()
        self.named_timer = nd.utils.NamedTimer()
        self.price_timer = nd.utils.ParallelTimer(unit="$")
        self.token_timer = nd.utils.ParallelTimer(unit="token")

    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Fit the model to the data."""
        save_path = Path(self.args.save_path) if self.args.save_path else None
        self.best_model = None
        self.speed_timer.clear(reset_last_add_time=True)
        self.named_timer.clear(reset_last_add_time=True)
        self.token_timer.clear(reset_last_add_time=True)
        self.price_timer.clear(reset_last_add_time=True)
        for n_iter in range(0, 1 + self.args.n_iter):
            ## Eolve
            try:
                if n_iter == 0:
                    self.population = self.init_population(df_train, df_test)
                else:
                    self.population = self.evolve(self.population, df_train, df_test, n_iter=n_iter)
            except Exception as e:
                _logger.error(f"Error during {n_iter}-th iteration since: {log_exception(e)}")
                if self.args.debug: raise e from e
            self.speed_timer.add()
            ## See if explanation of best model is added
            if (
                (self.best_model is not None) and 
                ((best_uuid := self.best_model.uuid) in (uuid_list := [i.uuid for i in self.population])) and
                (self.best_model.explanation != self.population[idx := uuid_list.index(best_uuid)].explanation)
            ):
                self.best_model.explanation = deepcopy(self.population[idx].explanation)
                _logger.note(
                    f"Explanation added for best model {best_uuid} "
                    f"at iteration {self.best_model.n_iter}:"
                    f"\n{render_markdown(self.best_model.explanation)}"
                )
            ## Update Best Model
            best = max(self.population, key=lambda indiv: indiv.fitness)
            if _update_best := (self.best_model is None or best.fitness > self.best_model.fitness):
                self.best_model = deepcopy(best)
            elif best.fitness < self.best_model.fitness:
                _logger.warning(
                    f"Best model fitness decreased "
                    f"from {self.best_model.fitness:.4f} (at {self.best_model.n_iter}) " 
                    f"to {best.fitness:.4f} at iteration {n_iter}"
                )
            ## Save Best Model of Current (not all historical) Population
            if save_path and _update_best:
                save_file = save_path / 'best_model' / f'best_model_{n_iter}.py'
                save_file.parent.mkdir(parents=True, exist_ok=True)
                log = self.get_log_for_indiv(best)
                detail = "\n".join(f"{k}={v}" for k, v in log.items())
                save_file.write_text(re.sub(r"\033\[[\d;]+m", "", tag2ansi(
                    f"'''\n{detail}\n'''\n{best.program}"
                )))
            ## Save Checkpoint
            if save_path and self.args.save_checkpoint_per_iter and n_iter % self.args.save_checkpoint_per_iter == 0:
                save_file = save_path / 'checkpoint' / f'population_{n_iter}.pkl.gz'
                save_file.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = [indiv.json_dict() for indiv in self.population]
                with gzip.open(save_file, "wb") as f:
                    pickle.dump(checkpoint, f)
            ## Early stopping
            if '_patience' not in locals() or _update_best:
                _patience = 100
            else:
                _patience -= 1
            ## Save Records
            record = dict(
                n_iter=n_iter,
                time=self.speed_timer.time,
                best=best.json_dict(),
                population_size=len(self.population),
                speed=self.speed_timer.speed,
                time_detail=self.named_timer.named_time,
                token_usage=self.token_timer.named_count,
                money_usage=self.price_timer.named_count,
            )
            record = jsonalize(record)
            self.records.append(record)
            if save_path:
                save_file = save_path / 'records.jsonl'
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
            ## Print Logs
            if (
                (_update_best) or
                (_patience <= 0) or
                (n_iter == self.args.n_iter) or
                (self.args.log_per_iter and (n_iter % self.args.log_per_iter == 0)) or
                (self.args.log_per_sec and ('_last_log_time' not in locals() or self.speed_timer.time - _last_log_time > self.args.log_per_sec))
            ):
                log = {
                    'Iter': n_iter,
                    'Fitness': f'{self.best_model.fitness:.6f}',
                    'Population Size': len(self.population),
                    'Speed': self.speed_timer.to_str(mode="pace"),
                    'Time Usage': self.named_timer.to_str(mode="time", mode_of_detail="pace", mode_of_percent="by_time"),
                    'Token Usage': '[gold bold]' + self.token_timer.to_str(mode="count", mode_of_detail="speed", mode_of_percent="by_count"),
                    'Money Usage': '[gold bold]' + self.price_timer.to_str(mode="count", mode_of_detail="speed", mode_of_percent="by_count"),
                    'Patience': f'{_patience}/100 (since {self.best_model.n_iter} iteration)'
                }
                if _update_best:
                    log |= self.get_log_for_indiv(self.best_model)
                    msg = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                    _logger.note(tag2ansi(msg))
                else:
                    msg = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                    _logger.info(tag2ansi(msg))
                _last_log_time = self.speed_timer.time
            ## Early stopping check
            if _patience <= 0:
                _logger.info(f"Early stopping at iteration {n_iter} due to no improvement in the last 100 iterations.")
                break

    def evolve(
        self, population: List[Individual], df_train: pd.DataFrame, df_test: pd.DataFrame, n_iter=None
    ) -> List[Individual]:
        children = []
        ## Elitism: Keep the top-k individuals
        loader = self.get_elitism(population, n_iter=n_iter)
        loader = tqdm(loader, total=self.args.elitism_k, desc="Geting elitism...")
        for child in loader:
            children.append(child)
        ## LLM Mutation: Generate new individuals using LLM
        if self.args.llm_model is not None:
            size = self.args.generate_per_prompt
            loader = self.llm.llm_mutation(population, size, n_iter=n_iter)
            loader = GenWrapper(loader)
            # pbar = tqdm(total=size, desc="Generating new individuals using LLM...")
            for child in tqdm(loader, total=size, desc="Generating new individuals using LLM..."):
                if child is not None:
                    children.append(child)
                # pbar.update(1)
            # pbar.close()
            usage_list = loader.return_value
            self.update_usage(usage_list)
                
        ## GP Mutatioin: Generate new individuals using Genetic Programming
        size = max(self.args.population_size - len(children), 0)
        loader = self.gp.gp_mutation(population, size, n_iter)
        loader = tqdm(loader, total=size, desc="Generating new individuals using GP...")
        for child in loader:
            if child is not None:
                children.append(child)
            else:
                _logger.warning("GP mutation should not generate None individual!")
        ## Evaluate all children
        for child in tqdm(children, desc="Evaluating fitness for individuals..."):
            ## Evalute newly generated children
            if child.n_iter == n_iter:
                self.evaluator.set_fitness(child, df_test, df_fit=df_train)
            ## Ask LLM to explain its newly generated children
            if child.n_iter == n_iter and child.source == 'llm-mutation' and child.explanation is None and self.args.explain_children:
                usage = self.llm.set_explanation(child)
                self.update_usage([('explain-children', usage)])
            ## Log newly generated children with explanation
            if child.n_iter == n_iter and child.explanation is not None:
                log = self.get_log_for_indiv(child)
                detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                _logger.info(tag2ansi(f"Newly generated individual with explanation: {detail}"))
        return children

    def init_population(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> List[Individual]:
        children = []
        ## Initialize the seed individual
        children.extend(self.seed_individuals)
        ## Generate the rest of the population
        loader = range(len(children), self.args.population_size)
        if self.args.use_seed_individuals_only: loader = []
        loader = tqdm(loader, desc="Generating initial individuals using GP...")
        for _ in loader:
            coeff = self.generator.generate_eqtree(nettype=self.variables['t'].nettype)
            f_sst = self.generator.generate_eqtree(nettype=self.variables['x'].nettype)
            f_wwv = self.generator.generate_eqtree(nettype=self.variables['h'].nettype)
            triplet = Triplet(coeff=coeff, f_sst=f_sst, f_wwv=f_wwv)
            child = Individual(
                uuid=my_uuid(self._rng), 
                n_iter=0, 
                source='generator', 
                parents=[], 
                triplet=triplet,
                program=self.assistant.triplet_to_program(triplet),
                dynamics=self.assistant.triplet_to_dynamics(triplet),
            )
            children.append(child)
        ## Evaluate all children
        for child in tqdm(children, desc="Evaluating fitness for individuals..."):
            ## Evaluate fitness
            if child.n_iter == 0:
                self.evaluator.set_fitness(child, df_test, df_fit=df_train)
            ## Ask LLM to explain its newly generated children
            if child.n_iter == 0 and child.source == 'llm-mutation' and child.explanation is None and self.args.explain_children:
                usage = self.llm.set_explanation(child)
                self.update_usage([('explain-children', usage)])
            ## Log child with explanation
            if child.n_iter == 0 and child.explanation is not None:
                log = self.get_log_for_indiv(child)
                detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                _logger.debug(tag2ansi(f"Newly generated individual with explanation: {detail}"))
        return children

    def get_elitism(self, population: List[Individual], n_iter) -> Generator[Individual, None, None]:
        """Get the top-k individuals from the population."""
        sort = sorted(population, key=lambda x: (x.fitness, x.uuid), reverse=True)
        topk = sort[:self.args.elitism_k]
        for indiv in topk:
            if getattr(indiv, "expire_iter", -float('inf')) < n_iter + self.args.expire_iter:
                # 向上更新 indiv 的过期时间
                indiv.expire_iter = n_iter + self.args.expire_iter
        if self.args.keep_best:
            # 确保最优个体不过期
            topk[0].expire_iter = float('inf')
        for i in population:
            if getattr(i, "expire_iter", -float('inf')) >= n_iter:
                yield i

    def get_log_for_indiv(self, indiv: Individual) -> str:
        """Log the individual details."""
        format_percent = lambda x: f'[{'green' if x > 0 else ('yellow' if x == 0 else 'red')}]{x:+.2%}[reset]'

        baseline_skills = indiv.detail['skill'] / (indiv.detail['skill%'] + 1)
        if (invalid_mask := ~np.isfinite(baseline_skills)).any():
            baseline = self.evaluator.baseline_result['skill']
            if (is_equal := np.isclose(baseline_skills[~invalid_mask], baseline[~invalid_mask])).all():
                baseline_skills[invalid_mask] = baseline[invalid_mask]
            else:
                _logger.warning(
                    f"Individual {indiv.uuid} has invalid skill% values leading to non-finite baseline skills, "
                    f"but the valid parts do not match the evaluator's baseline skills."
                )
                baseline_skills[invalid_mask] = 0.0

        baseline_skills_cv = indiv.detail['skill_cv'] / (indiv.detail['skill_cv%'] + 1)
        if (invalid_mask := ~np.isfinite(baseline_skills_cv)).any():
            baseline = self.evaluator.baseline_result['skill_cv']
            if (is_equal := np.isclose(baseline_skills_cv[~invalid_mask], baseline[~invalid_mask])).all():
                baseline_skills_cv[invalid_mask] = baseline[invalid_mask]
            else:
                _logger.warning(
                    f"Individual {indiv.uuid} has invalid skill_cv% values leading to non-finite baseline skill_cv, "
                    f"but the valid parts do not match the evaluator's baseline skill_cv."
                )
                baseline_skills_cv[invalid_mask] = 0.0

        log = {
            "Info": f"{indiv.uuid} from [pink bold]{indiv.source}[reset]({', '.join(indiv.parents)}) at Iter [pink]{indiv.n_iter}[reset]",
            "Fitness": f"[pink bold]{indiv.fitness:.6f}[reset]",
            'Skills': (
                '; '.join(f'[pink bold]{a}_{b}={v}[reset] month' for (a, b), v in indiv.detail['skill'].items())
                + f';[pink bold]Total SST[reset]={indiv.detail["skill"]['x'].sum()} month'
                + f';[pink bold]Total Z20[reset]={indiv.detail["skill"]['h'].sum()} month'
                + f';[pink bold]Total[reset]={indiv.detail["skill"].sum()} month'
            ),
            'Skills%': (
                ';'.join(f'{a}_{b}={format_percent(v)}' for (a, b), v in indiv.detail['skill%'].items())
                + f';SST={format_percent(indiv.detail["skill"]["x"].sum() / baseline_skills['x'].sum() - 1)}'
                + f';Z20={format_percent(indiv.detail["skill"]["h"].sum() / baseline_skills['h'].sum() - 1)}'
                + f';Total={format_percent(indiv.detail["skill"].sum() / baseline_skills.sum() - 1)}'
            ),
            'Skills (CV)': (
                '; '.join(f'[pink bold]{a}_{b}={v}[reset] month' for (a, b), v in indiv.detail['skill_cv'].items())
                + f';[pink bold]Total SST[reset]={indiv.detail["skill_cv"]['x'].sum()} month'
                + f';[pink bold]Total Z20[reset]={indiv.detail["skill_cv"]['h'].sum()} month'
                + f';[pink bold]Total[reset]={indiv.detail["skill_cv"].sum()} month'
            ),
            'Skills% (CV)': (
                ';'.join(f'{a}_{b}={format_percent(v)}' for (a, b), v in indiv.detail['skill_cv%'].items())
                + f';SST={format_percent(indiv.detail["skill_cv"]["x"].sum() / baseline_skills_cv['x'].sum() - 1)}'
                + f';Z20={format_percent(indiv.detail["skill_cv"]["h"].sum() / baseline_skills_cv['h'].sum() - 1)}'
                + f';Total={format_percent(indiv.detail["skill_cv"].sum() / baseline_skills_cv.sum() - 1)}'
            ),
            'Correlation AUC%': format_percent(indiv.detail['correlation_auc%'].sum()),
            'Correlation AUC% (CV)': format_percent(indiv.detail['correlation_auc_cv%'].sum()),
            'Trajectory RMSE%': format_percent(indiv.detail['traj_rmse%']),
            'Formula Length%': format_percent(indiv.detail['length%']),
            'RMSE%': format_percent(indiv.detail['rmse%']),
            'Std%': format_percent(indiv.detail['std%']),
            'Skewness%': format_percent(indiv.detail['skewness%']),
            'Spectrum Density%': format_percent(indiv.detail['spectrum%']),
            'Lead Lag Corr%': format_percent(indiv.detail['lead_lag%']),
            'Event Prediction%': format_percent(indiv.detail['event_index%']),
            'Forecast Month%': format_percent(indiv.detail['forecast%']),
            "AIC%": format_percent(indiv.detail['aic%']),
            "BIC%": format_percent(indiv.detail['bic%']),
            "Params%": format_percent(indiv.detail['params%']),
            "Forecast Month": "; ".join(f"[pink]{k}={v:.2f}[reset] month" for k, v in indiv.detail['forecast'].items()),
            '\ncoeff': f"[pink]{indiv.triplet.coeff.to_str(number_format='.2e', omit_mul_sign=False)}[reset]",
            '\nf_sst': f"[pink]{indiv.triplet.f_sst.to_str(number_format='.2e', omit_mul_sign=False)}[reset]",
            '\nf_wwv': f"[pink]{indiv.triplet.f_wwv.to_str(number_format='.2e', omit_mul_sign=False)}[reset]",
            'dx/dt': indiv.dynamics.ddt_sst.to_str(number_format='.2e', omit_mul_sign=False),
            'dh/dt': indiv.dynamics.ddt_wwv.to_str(number_format='.2e', omit_mul_sign=False),
            'Program': f"\n{render_python(indiv.program)}",
        }
        if 'skill_after_2004' in indiv.detail:
            log['Skills (After 2004)'] = (
                '; '.join(f'[pink bold]{a}_{b}={v}[reset] month' for (a, b), v in indiv.detail['skill_after_2004'].items())
                + f';[pink bold]Total SST[reset]={indiv.detail["skill_after_2004"]['x'].sum()} month'
                + f';[pink bold]Total Z20[reset]={indiv.detail["skill_after_2004"]['h'].sum()} month'
                + f';[pink bold]Total[reset]={indiv.detail["skill_after_2004"].sum()} month'
            )
            log['Skills% (After 2004)'] = (
                ';'.join(f'{a}_{b}={format_percent(v)}' for (a, b), v in (indiv.detail['skill_after_2004'] / self.evaluator.baseline_result['skill_after_2004'] - 1).items())
                + f';SST={format_percent(indiv.detail["skill_after_2004"]["x"].sum() / self.evaluator.baseline_result['skill_after_2004']['x'].sum() - 1)}'
                + f';Z20={format_percent(indiv.detail["skill_after_2004"]["h"].sum() / self.evaluator.baseline_result['skill_after_2004']['h'].sum() - 1)}'
                + f';Total={format_percent(indiv.detail["skill_after_2004"].sum() / self.evaluator.baseline_result['skill_after_2004'].sum() - 1)}'
            )

        if indiv.explanation is not None:
            log['Explanation'] = f"\n{render_markdown(indiv.explanation)}"
        return log

    def update_usage(self, usage_list: List[Tuple[str, dict]]):
        for suffix, usage in usage_list:
            for k, v in usage['token'].items():
                if v != 0:
                    self.token_timer.add(f"{k}-{suffix}", v)
            for k, v in usage['price'].items():
                if v != 0:
                    self.price_timer.add(f"{k}-{suffix}", v)
