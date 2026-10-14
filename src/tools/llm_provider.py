import ast
import json
import logging
import numpy as np
from pathlib import Path
from argparse import Namespace
from collections import defaultdict
from typing import List, Generator, Dict
from nd2py.search.llmsr import render_markdown
from ..api import LLMAPI
from ..utils import my_uuid, my_hash, GenWrapper
from ..core import Individual
from ..assistant import Assistant

_logger = logging.getLogger(__name__)


def _softmax(logprob: np.ndarray, tau: float = 1.0) -> np.ndarray:
    prob = np.exp((logprob - np.max(logprob)) / tau)
    prob /= np.sum(prob)
    return prob


class LLMProvider:
    def __init__(self, args: Namespace, assistant: Assistant):
        self.args = args
        self.assistant = assistant
        self.llm_api = LLMAPI.load(args)
        _logger.info(f"LLM Provider initialized with model: {type(self.llm_api).__name__}({self.llm_api.model})")
        self._rng = np.random.default_rng(seed := args.random_state + my_hash(self.__class__.__name__))
        _logger.debug(f"LLMProvider initialized with RNG seed: {seed}, initial state: {self._rng.bit_generator.state}")

    def llm_mutation(
        self, population: List[Individual], size: int, n_iter: int
    ) -> Generator[Individual, dict, None]:
        """Generate new individuals using LLM."""
        ## Ensure this is a generator
        yield from []
        ## Track token & price usage
        usage_list = []
        ## Sample individuals from population
        parents = self.sample_individuals(population, n_iter)
        ## Set Explanation for parents if not exists
        if self.args.explain_parents:
            if self.args.replay_path is None:
                for parent in parents:
                    usage = self.set_explanation(parent)
                    usage_list.append(('explain-parents', usage))
            else:
                for parent in parents:
                    save_file = Path(self.args.replay_path) / f'explain_{parent.uuid}.json'
                    with open(save_file, "r", encoding="utf-8") as f:
                        save_data = json.load(f)
                    parent.explanation = save_data["contents"][0]
        ## Load programs from replay files
        if self.args.replay_path is not None:
            replay_file = Path(self.args.replay_path) / f'response_{n_iter}.json'
            with open(replay_file, "r", encoding="utf-8") as f:
                save_data = json.load(f)
            def replay_loader():
                yield from save_data["contents"]
                return save_data
            loader = replay_loader()
        ## Generate children using LLM
        else:
            messages = self.assistant.ask_for_generate(parents)
            loader = self.llm_api(messages, n=size)
        ## Yield children one by one
        loader = GenWrapper(loader)
        for markdown in loader:
            try:
                if (raw_program := self.assistant.markdown_to_program(markdown)) is None:
                    yield None
                elif (triplet := self.assistant.program_to_triplet(raw_program)) is None:
                    yield None
                else:
                    triplet = self.assistant.triplet_to_reduction(triplet)
                    child = Individual(
                        uuid=my_uuid(self._rng),
                        n_iter=n_iter,
                        source='llm-mutation',
                        parents=[indiv.uuid for indiv in parents],
                        triplet=triplet,
                        program=self.assistant.triplet_to_program(triplet),
                        dynamics=self.assistant.triplet_to_dynamics(triplet),
                    )
                    # 无法在这里为 child 生成 explanation，因为尚未评估其 fitness 和 detail
                    child.raw_program = raw_program
                    yield child
            except Exception as exc:
                _logger.warning("Discarding an invalid LLM candidate: %s", exc)
                yield None
        save_data = loader.return_value
        if self.args.save_path: 
            save_file = Path(self.args.save_path) / 'response' / f'response_{n_iter}.json'
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
        usage = save_data['usage']
        usage_list.append(('generate', usage))
        return usage_list

    def sample_individuals(self, population: List[Individual], n_iter) -> List[Individual]:
        ## 将 island 中的个体按照 fitness 分为若干 cluster
        clusters = defaultdict(list)
        for indiv in population:
            clusters[indiv.fitness].append(indiv)
        fitness = np.array(list(clusters.keys()))
        clusters = list(clusters.values())
        ## 按照 Boltzmann selection procedure 对 cluster 进行采样，fitness 大者优先
        if self.args.programs_per_prompt < len(clusters):
            tau = self.args.temperature_init * (1 - (n_iter % self.args.temperature_period) / self.args.temperature_period)
            prob = _softmax(fitness, tau=tau)
            if len(np.nonzero(prob)[0]) < self.args.programs_per_prompt:
                prob = (prob + 0.1) / (prob + 0.1).sum()
            clusters_idx = self._rng.choice(len(clusters), size=self.args.programs_per_prompt, p=prob, replace=False)
            clusters = [clusters[i] for i in clusters_idx]
        ## 对 cluster 中个体进行采样
        individuals = []
        for cluster in clusters:
            indiv = self._rng.choice(cluster)
            individuals.append(indiv)
        return individuals

    def set_explanation(self, indiv: Individual) -> dict:
        """Generate explanation for the individual if not exists."""
        # return {'token': {}, 'price': {}}
        if indiv.explanation is not None:
            return {'token': {}, 'price': {}}
        messages = self.assistant.ask_for_explain(indiv)
        loader = self.llm_api(messages, n=1)
        loader = GenWrapper(loader)
        for idx, explanation in enumerate(loader):
            if idx >= 1:
                raise RuntimeError("LLM explanation did not finish as expected.")
            indiv.explanation = explanation
        save_data = loader.return_value
        if self.args.save_path:
            save_file = Path(self.args.save_path) / 'response' / f'explain_{indiv.uuid}.json'
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
        return save_data['usage']
