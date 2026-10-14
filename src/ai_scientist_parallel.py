import os
import sys
import time
import queue
import signal
import logging
import traceback
import setproctitle
import nd2py as nd
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List, Dict
from logging.handlers import QueueHandler
from rich.live import Live
from rich.text import Text
from rich.console import Console, Group
from concurrent.futures import TimeoutError
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from .ai_scientist import AIScientist
from .core import Individual, Triplet
from .utils import my_uuid, tag2ansi, log_exception, RichLogHandler, MergingBarColumn, GenWrapper, NamedMpArray

_logger = logging.getLogger(__name__)


class AIScientistParallel(AIScientist):
    """LLM enhanced Genetic Programming for regression tasks (multi-process version)."""
    def __init__(self, args, *other_args, **kwargs):
        super().__init__(args, *other_args, **kwargs)

        if args.num_workers is None:
            raise ValueError("num_workers must be specified for AIScientistParallel.")
        elif args.num_workers == 0:
            self.args.num_workers = mp.cpu_count()
            _logger.info(f"Using ALL {args.num_workers} workers for parallel fitness evaluation.")
        else:
            _logger.info(f"Using {args.num_workers} workers for parallel fitness evaluation.")

    @staticmethod
    def monitor_wrapper(func):
        """ redirect src logger to log_queue and start monitor process to track progress """
        def wrapper(self, *args, **kwargs):
            ## Prepare logger and console for monitor process
            monitor_logger = logging.getLogger("monitor") # 不能是 src.monitor，因为所有 src 的日志都被重定向到 log_queue，需要由这个 logger 输出
            parent_logger = logging.getLogger('src')
            monitor_logger.handlers = parent_logger.handlers[:]
            monitor_logger.propagate = False
            console = Console(soft_wrap=True)
            for idx, handler in enumerate(monitor_logger.handlers):
                if type(handler) is logging.StreamHandler:
                    monitor_logger.handlers[idx] = RichLogHandler(console, handler.formatter)
                    monitor_logger.handlers[idx].setLevel(handler.level)
            ## Redirect src logger to log_queue
            log_queue = mp.Queue()
            backup = {}
            for name in ['src', 'nd2py']:
                src_logger = logging.getLogger(name)
                backup[f'{name}-handlers'] = src_logger.handlers[:]
                backup[f'{name}-propagate'] = src_logger.propagate
                src_logger.handlers = [QueueHandler(log_queue)]
                src_logger.propagate = False
            ## Start a monitor process
            status_counter = NamedMpArray([
                # 运行中的子进程
                'gp_producer', 'llm_producer', 'fitness_worker', 'fitness_worker_inner', 'monitor',
                # Produced during Init
                'seed_produced', 'generate_produced',
                # Produced during Evolve
                'elit_produced', 'llm_produced', 'llm_produced_f', 'gp_produced', 'gp_produced_f', 
                # Successly Consumed
                'consumed', 
                # Failed to consume
                'skip_consumed', 'timeout_consumed', 'error_consumed', 'nonfinite_consumed', 'eoferror_consumed',
            ])
            monitor_process = mp.Process(
                target=self.monitor,
                args=(status_counter, self.args.population_size, self.args.num_workers, 
                      log_queue, console, monitor_logger),
            )
            monitor_process.start()
            try:
                return func(self, *args, **kwargs, status_counter=status_counter)
            finally:
                ## 恢复日志配置并关闭 log_queue
                for name in ['src', 'nd2py']:
                    src_logger = logging.getLogger(name)
                    src_logger.handlers = backup[f'{name}-handlers']
                    src_logger.propagate = backup[f'{name}-propagate']
                log_queue.put(None)  # Signal the monitor to stop
                log_queue.close()
                log_queue.join_thread()
                monitor_process.join()
        return wrapper

    @staticmethod
    def exception_wrapper(func):
        """ 套在每个子进程的外面，用来捕获那些子进程内产生的错误并 print (父进程没法捕获子进程的错误) """
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt as e:
                proctitle = setproctitle.getproctitle()
                print(f"\n[{proctitle}] KeyboardInterrupt detected, terminating process...")
            except Exception as e:
                _logger.error(f"{log_exception(e)}")
                raise e from e
        return wrapper

    @monitor_wrapper
    def evolve(
        self, population: List[Individual], df_train: pd.DataFrame, df_test: pd.DataFrame, n_iter=None, status_counter=None
    ) -> List[Individual]:
        ## Prepare shared variables
        task_queue = mp.Queue() # 向 fitness_worker 分发任务
        result_queue = mp.Queue() # 从 fitness_worker 收集结果
        # self._register_child_queue(task_queue)
        # self._register_child_queue(result_queue)
        ## Elitism: Keep the top-k individuals
        for child in self.get_elitism(population, n_iter=n_iter):
            task_queue.put(child)
            status_counter.inc('elit_produced')
        self.named_timer.add("set_elitism")
        ## Prepare for the LLM and GP producers
        elit_size = status_counter['elit_produced']
        llm_size = self.args.generate_per_prompt
        if self.args.llm_model is None:
            llm_size = 0
        gp_size = self.args.population_size - elit_size - llm_size
        if gp_size < 0:
            _logger.warning(
                f"gp_size={gp_size} is less than 0, setting to 0. "
                "This may happen if the population size is too small or the number of generated individuals is too high. "
            )
            gp_size = 0
        _logger.info(
            f"Population: Elitism={elit_size}, LLM={llm_size}, GP={gp_size}, Total={self.args.population_size}"
        )
        self.named_timer.add('drop')
        ## Start the LLM and GP producers
        if llm_size > 0:
            llm_conn, llm_conn_child = mp.Pipe()
            llm_progress = mp.Process(
                target=self.llm_producer,
                args=(task_queue, self.llm, population, n_iter, 
                      status_counter, llm_conn_child, llm_size),
            )
            llm_progress.start()
            # self._register_child_process(llm_progress)
        if gp_size > 0:
            gp_conn, gp_conn_child = mp.Pipe()
            gp_progress = mp.Process(
                target=self.gp_producer,
                args=(task_queue, self.gp, population, n_iter,
                      status_counter, gp_conn_child, gp_size),
            )
            gp_progress.start()
            # self._register_child_process(gp_progress)
        self.named_timer.add("init_producers")
        ## Start the fitness workers
        workers = []
        for idx in range(self.args.num_workers):
            worker = mp.Process(
                target=self.fitness_worker,
                args=(task_queue, result_queue, self.evaluator, df_train, df_test, 
                      status_counter, idx, n_iter, self.args.timeout, self.args.debug, self.args.evaluate_before_and_after_2000),
            )
            worker.start()
            workers.append(worker)
            # self._register_child_process(worker)
        self.named_timer.add("init_workers")
        ## Wait for producers to finish
        if 'llm_progress' in locals():
            self.llm, usage_list, explanation_dict = llm_conn.recv()
            llm_progress.join()
            self.update_usage(usage_list)
        if 'gp_progress' in locals():
            self.gp = gp_conn.recv()
            gp_progress.join()
        self.named_timer.add("join_producers")
        ## Send poison pills to workers
        for _ in range(self.args.num_workers):
            task_queue.put(None)
        self.named_timer.add("put_poison_pills")
        ## Collect results from workers
        children = []
        finished_workers = 0
        total_workers = len(workers)
        while finished_workers < total_workers:
            child = result_queue.get()
            if child is None:
                finished_workers += 1
                continue
            children.append(child)
            ## Ask LLM to explain its newly generated children ## TODO
            # if child.n_iter == n_iter and child.source == 'llm-mutation' and child.explanation is None:
            #     usage = self.llm.set_explanation(child)
            #     for k, v in usage['token'].items():
            #         self.token_timer.add(k, v)
            #     for k, v in usage['price'].items():
            #         self.price_timer.add(k, v)
            ## Log child with explanation
            # if child.n_iter == n_iter and child.explanation is not None:
            #     log = self.get_log_for_indiv(child)
            #     detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
            #     _logger.debug(tag2ansi(f"Newly generated individual with explanation: {detail}"))
            ## Log best individual so far
            if self.args.log_subbest and ('best' not in locals() or child.fitness > best.fitness) and np.isfinite(child.fitness):
                best = child
                log = self.get_log_for_indiv(best)
                detail = " | ".join(f"[#FFAA00]{k}[reset]={v}" for k, v in log.items())
                _logger.info(tag2ansi(f"New best individual so far: {detail}"))
        self.named_timer.add("wait_for_workers")
        ## Add explanations from LLM producer
        if 'explanation_dict' in locals():
            uuid_list = [indiv.uuid for indiv in children]
            for uuid, explanation in explanation_dict.items():
                if uuid in uuid_list:
                    children[uuid_list.index(uuid)].explanation = explanation
                    _logger.debug(f"Add explanation for {uuid}")
                else:
                    _logger.warning(f"Cannot find individual {uuid} to add explanation")
        ## Wait for workers to finish
        for w in workers:
            w.join()
        self.named_timer.add("join_workers")
        ## Close queues
        task_queue.close()
        result_queue.close()
        task_queue.join_thread()
        result_queue.join_thread()
        children = sorted(children, key=lambda x: x.uuid)  # 保持顺序以保证可复现性
        return children

    @monitor_wrapper
    def init_population(self, df_train: pd.DataFrame, df_test: pd.DataFrame, status_counter=None) -> List[Individual]:
        ## Prepare shared variables
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        ## Initialize the seed individual
        for child in self.seed_individuals:
            task_queue.put(child)
            status_counter.inc('seed_produced')
        ## Generate the rest of the population
        loader = range(status_counter['seed_produced'], self.args.population_size)
        if self.args.use_seed_individuals_only: loader = []
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
            task_queue.put(child)
            status_counter.inc('generate_produced')
        ## Send poison pills to workers
        for _ in range(self.args.num_workers):
            task_queue.put(None)
        ## Create fitness workers
        workers = []
        for idx in range(self.args.num_workers):
            worker = mp.Process(
                target=self.fitness_worker,
                args=(task_queue, result_queue, self.evaluator, df_train, df_test,
                      status_counter, idx, 0, self.args.timeout, self.args.debug, self.args.evaluate_before_and_after_2000),
            )
            workers.append(worker)
            worker.start()
        ## Collect results from workers
        children = []
        finished_workers = 0
        total_workers = len(workers)
        while finished_workers < total_workers:
            child = result_queue.get()
            if child is None:
                finished_workers += 1
                continue
            children.append(child)
            ## Ask LLM to explain its newly generated children
            if child.n_iter == 0 and child.source == 'llm-mutation' and child.explanation is None and self.args.explain_children:
                usage = self.llm.set_explanation(child)
                for k, v in usage['token'].items():
                    self.token_timer.add(k, v)
                for k, v in usage['price'].items():
                    self.price_timer.add(k, v)
            ## Log child with explanation
            if child.n_iter == 0 and child.explanation is not None:
                log = self.get_log_for_indiv(child)
                detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                _logger.debug(tag2ansi(f"Newly generated individual with explanation: {detail}"))
            ## Log best individual so far
            if self.args.log_subbest and ('best' not in locals() or child.fitness > best.fitness) and np.isfinite(child.fitness):
                best = child
                log = self.get_log_for_indiv(best)
                detail = " | ".join(f"[#FFAA00]{k}[reset]={v}" for k, v in log.items())
                _logger.info(tag2ansi(f"New best individual so far: {detail}"))
                if hasattr(best, 'detail1'):
                    best.detail1, best.detail = best.detail, best.detail1
                    log = self.get_log_for_indiv(best)
                    detail = " | ".join(f"[#FFAA00]{k}[reset]={v}" for k, v in log.items())
                    _logger.info(tag2ansi(f"New best individual so far (until 1999-12): {detail}"))
                    best.detail1, best.detail = best.detail, best.detail1
                if hasattr(best, 'detail2'):
                    best.detail2, best.detail = best.detail, best.detail2
                    log = self.get_log_for_indiv(best)
                    detail = " | ".join(f"[#FFAA00]{k}[reset]={v}" for k, v in log.items())
                    _logger.info(tag2ansi(f"New best individual so far (since 2000-01): {detail}"))
                    best.detail2, best.detail = best.detail, best.detail2
        ## Wait for workers to finish
        for worker in workers:
            worker.join()
        self.named_timer.add("wait_for_workers")
        ## Close queues
        task_queue.close()
        result_queue.close()
        task_queue.join_thread()
        result_queue.join_thread()
        children = sorted(children, key=lambda x: x.uuid) # 保持顺序以保证可复现性
        return children

    @staticmethod
    @exception_wrapper
    def safe_set_fitness(status_counter, evaluator, indiv, df_train, df_test, conn, evaluate_before_and_after_2000):
        """ 在子进程中为 indiv 设置 fitness """
        status_counter.inc('fitness_worker_inner')
        try:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            proctitle = setproctitle.getproctitle()
            setproctitle.setproctitle(f"{proctitle} (Inner)")
            with nd.no_warn(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                evaluator.set_fitness(indiv, df_test, df_fit=df_train)
            # 可能是因为太大的对象在进程间传递，不扔掉这三个键会导致卡死
            if 'lead_lag' in indiv.detail: indiv.detail.pop('lead_lag')
            if 'GENERATION' in indiv.detail: indiv.detail.pop('GENERATION')
            if 'PREDICTION' in indiv.detail: indiv.detail.pop('PREDICTION')
            if 'PREDICTION_cv' in indiv.detail: indiv.detail.pop('PREDICTION_cv')
            if 'PREDICTION_after_2004' in indiv.detail: indiv.detail.pop('PREDICTION_after_2004')
            ## 换个数据集再算一次，取更差的 fitness 作为最终结果
            if evaluate_before_and_after_2000:
                indiv1 = Individual(indiv.uuid, indiv.n_iter, indiv.source, indiv.parents,
                                    indiv.program, indiv.triplet, indiv.dynamics)
                with nd.no_warn(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    evaluator.set_fitness(indiv1, df_test.loc[:'1999-12'], df_fit=df_train.loc[:'1999-12'])
                if 'lead_lag' in indiv1.detail: indiv1.detail.pop('lead_lag')
                if 'GENERATION' in indiv1.detail: indiv1.detail.pop('GENERATION')
                if 'PREDICTION' in indiv1.detail: indiv1.detail.pop('PREDICTION')
                if 'PREDICTION_cv' in indiv1.detail: indiv1.detail.pop('PREDICTION_cv')
                if 'PREDICTION_after_2004' in indiv1.detail: indiv1.detail.pop('PREDICTION_after_2004')
                indiv.fitness = min(indiv.fitness, indiv1.fitness)
                indiv.detail1 = indiv1.detail

                indiv2 = Individual(indiv.uuid, indiv.n_iter, indiv.source, indiv.parents,
                                    indiv.program, indiv.triplet, indiv.dynamics)
                with nd.no_warn(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    evaluator.set_fitness(indiv2, df_test.loc['2000-01':], df_fit=df_train.loc['2000-01':])
                if 'lead_lag' in indiv2.detail: indiv2.detail.pop('lead_lag')
                if 'GENERATION' in indiv2.detail: indiv2.detail.pop('GENERATION')
                if 'PREDICTION' in indiv2.detail: indiv2.detail.pop('PREDICTION')
                if 'PREDICTION_cv' in indiv2.detail: indiv2.detail.pop('PREDICTION_cv')
                if 'PREDICTION_after_2004' in indiv2.detail: indiv2.detail.pop('PREDICTION_after_2004')
                indiv.fitness = min(indiv.fitness, indiv2.fitness)
                indiv.detail2 = indiv2.detail
            conn.send(indiv)
        finally:
            status_counter.dec('fitness_worker_inner')

    @staticmethod
    @exception_wrapper
    def fitness_worker(
        task_queue,
        result_queue,
        evaluator,
        df_train, 
        df_test,
        status_counter,
        idx,
        n_iter,
        timeout,
        debug,
        evaluate_before_and_after_2000,
    ):
        """ 不断从 task_queue 中获取 indiv, 计算其 fitness 后放入 result_queue """
        status_counter.inc('fitness_worker')
        proctitle = setproctitle.getproctitle()
        setproctitle.setproctitle(f"{proctitle} (Worker{idx} - {n_iter})")
        while True:
            ## Get an individual to evaluate
            if (indiv := task_queue.get()) is None: 
                break # poison pill: exit signal
            ## Individual not for this iteration
            elif indiv.n_iter != n_iter:
                status_counter.inc('skip_consumed')
            ## Evaluate fitness with timeout
            else:
                ## Start inner process with timeout to evaluate fitness
                parent_conn, child_conn = mp.Pipe()
                inner_worker = mp.Process(
                    target=AIScientistParallel.safe_set_fitness, 
                    args=(status_counter, evaluator, indiv, df_train, df_test, child_conn, evaluate_before_and_after_2000),
                )
                inner_worker.start()
                inner_worker.join(timeout) # wait for at most `timeout` seconds
                ## Timeout exit
                if inner_worker.is_alive():
                    status_counter.inc('timeout_consumed')
                    inner_worker.terminate()
                    inner_worker.join()
                    indiv.fitness = -float("inf")
                    indiv.detail = indiv.detail or {"error": f"Timeout after {timeout}s"}
                    _logger.trace(f"set_fitness for indiv {indiv} took too long (> {timeout}s)")
                ## Abnormal exit
                elif inner_worker.exitcode != 0:
                    status_counter.inc('error_consumed')
                    indiv.fitness = -float("inf")
                    indiv.detail = indiv.detail or {"error": f"set_fitness exited with code {inner_worker.exitcode}."}
                    _logger.error(f"Inner Process exited with code {inner_worker.exitcode} when set_fitness for indiv {indiv}")
                    # if debug: raise RuntimeError(f"Inner Process exited with code {inner_worker.exitcode} when set_fitness for indiv {indiv}")
                ## Normal exit
                else:
                    try:
                        indiv = parent_conn.recv()
                        if not np.isfinite(indiv.fitness):
                            status_counter.inc('nonfinite_consumed')
                            indiv.fitness = -float("inf")
                            indiv.detail = indiv.detail or {"error": f"Non-finite fitness: {indiv.fitness}"}
                            _logger.trace(f"Fitness of indiv {indiv} is {indiv.fitness}, setting to -inf")
                        else:
                            status_counter.inc('consumed')
                    except EOFError:
                        status_counter.inc('eoferror_consumed')
                        indiv.fitness = -float("inf")
                        indiv.detail = indiv.detail or {"error": f"EOFError during fitness evaluation."}
                        _logger.error(f"Failed to receive fitness result for indiv {indiv}")
            result_queue.put(indiv)
        result_queue.put(None)
        status_counter.dec('fitness_worker')

    @staticmethod
    @exception_wrapper
    def llm_producer(
        task_queue,
        llm,
        population,
        n_iter,
        status_counter,
        conn,
        size,
    ):
        """ 从 llm 中获取新个体并放入 task_queue """
        status_counter.inc('llm_producer')
        proctitle = setproctitle.getproctitle()
        setproctitle.setproctitle(f"{proctitle} (LLM Producer - {n_iter})")
        explanation_dict = {indiv.uuid: indiv.explanation for indiv in population}
        with nd.no_warn():
            loader = llm.llm_mutation(population, size=size, n_iter=n_iter)
            loader = GenWrapper(loader)
            for child in loader:
                if child is None:
                    status_counter.inc('llm_produced_f')
                else:
                    task_queue.put(child)
                    status_counter.inc('llm_produced')
        for indiv in population:
            if explanation_dict[indiv.uuid] == indiv.explanation:
                explanation_dict.pop(indiv.uuid)
            else:
                explanation_dict[indiv.uuid] = indiv.explanation
        usage_list = loader.return_value
        conn.send((llm, usage_list, explanation_dict))
        status_counter.dec('llm_producer')

    @staticmethod
    @exception_wrapper
    def gp_producer(
        task_queue,
        gp,
        population,
        n_iter,
        status_counter,
        conn,
        size,
    ):
        """ 从 gp 中获取新个体并放入 task_queue """
        status_counter.inc('gp_producer')
        proctitle = setproctitle.getproctitle()
        setproctitle.setproctitle(f"{proctitle} (GP Producer - {n_iter})")
        with nd.no_warn():
            for child in gp.gp_mutation(population, size, n_iter):
                if child is None:
                    status_counter.inc('gp_produced_f')
                else:
                    task_queue.put(child)
                    status_counter.inc('gp_produced')
        conn.send(gp)
        status_counter.dec('gp_producer')

    @staticmethod
    @exception_wrapper
    def monitor(
        status_counter,
        total_tasks,
        num_workers,
        log_queue,
        console,
        logger,
    ):
        status_counter.inc('monitor')
        proctitle = setproctitle.getproctitle()
        setproctitle.setproctitle(f"{proctitle} (Monitor)")

        progress = Progress(
            TextColumn("{task.description}"),
            MergingBarColumn(width=40),
            TextColumn("[green]{task.fields[success]}✓[/]"),
            TextColumn("[bold red]{task.fields[failure]}✗[/]"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        produce_task = progress.add_task(
            "[bold white]🛠️ Produced", total=total_tasks, success=0, failure=0
        )
        consume_task = progress.add_task(
            "[bold white]⚙️ Consumed", total=total_tasks, success=0, failure=0
        )
        with Live(console=console, refresh_per_second=10) as live:
            stop = False
            while not stop:  # Poison pill: exit signal
                try:
                    if (log_msg := log_queue.get_nowait()) is None:
                        stop = True
                        status_counter.dec('monitor')
                    else:
                        logger.handle(log_msg)
                except queue.Empty:
                    pass
                status = status_counter.to_dict()
                progress.update(
                    produce_task,
                    success=(success := status['seed_produced'] + status['generate_produced'] + status['elit_produced'] + status['llm_produced'] + status['gp_produced']),
                    failure=(failure := status['gp_produced_f'] + status['llm_produced_f']),
                    completed=success + failure,
                )
                progress.update(
                    consume_task,
                    success=(success := status['consumed'] + status['skip_consumed']),
                    failure=(failure := status['timeout_consumed'] + status['error_consumed'] + status['nonfinite_consumed'] + status['eoferror_consumed']),
                    completed=success + failure,
                )
                # 子进程运行状态 Details
                workers_str = (
                    f"[bold white]👷 Workers[/bold white]   "
                    f"[cyan]GP[/]=[green]{status['gp_producer']}[/]  "
                    f"[cyan]LLM[/]=[green]{status['llm_producer']}[/]  "
                    f"[cyan]Fitness[/]=[green]{status['fitness_worker']}[/] [dim white](🧩 {status['fitness_worker_inner']})[/]  "
                    f"[cyan]Monitor[/]=[green]{status['monitor']}[/]"
                )
                # Produced Details
                x1 = status['seed_produced'] + status['generate_produced']
                x2 = status['elit_produced'] + status['llm_produced'] + status['llm_produced_f'] + status['gp_produced'] + status['gp_produced_f']
                sources_prefix = "[bold white]📦 Sources[/bold white]   "
                sources_content = ""
                if (x1 == 0) and (x2 == 0):
                    sources_content = "[dim yellow]Waiting for data...[/]"
                elif (x1 > 0) and (x2 == 0):
                    sources_content = (
                        f"[cyan]Seed[/]=[green]{status['seed_produced']}[/]  "
                        f"[cyan]Gen[/]=[green]{status['generate_produced']}[/]"
                    )
                elif (x1 == 0) and (x2 > 0):
                    gp_fail_style = "[bold red]" if status['gp_produced_f'] > 0 else "[dim white]"
                    llm_fail_style = "[bold red]" if status['llm_produced_f'] > 0 else "[dim white]"
                    sources_content = (
                        f"[cyan]Elit[/]=[green]{status['elit_produced']}[/]  "
                        f"[cyan]GP[/]=[green]{status['gp_produced']}[/] "
                        f"{gp_fail_style}(❌{status['gp_produced_f']})[/]  "
                        f"[cyan]LLM[/]=[green]{status['llm_produced']}[/] "
                        f"{llm_fail_style}(❌{status['llm_produced_f']})[/]"
                    )
                else:
                    sources_content = f"[yellow]Mixed State (Init: {x1}, Evolve: {x2})[/]"
                sources_str = f"{sources_prefix}{sources_content}"
                fmt_err = lambda val: f"[bold red]{val}[/]" if val > 0 else f"[dim white]{val}[/]"
                results_str = (
                    f"[bold white]📉 Results[/bold white]   "
                    f"[cyan]OK[/]=[green]{status['consumed']}[/]  "
                    f"[cyan]Skip[/]=[green]{status['skip_consumed']}[/]  "
                    f"[cyan]T/O[/]={fmt_err(status['timeout_consumed'])}  "
                    f"[cyan]Err[/]={fmt_err(status['error_consumed'])}  "
                    f"[cyan]Non-Finite[/]={fmt_err(status['nonfinite_consumed'])}  "
                    f"[cyan]EOF[/]={fmt_err(status['eoferror_consumed'])}"
                )
                # 每次循环创建一个新的 Group，包含当前的 Progress 对象和新生成的 Text 对象
                # Text.from_markup() 解析我们上面的颜色代码
                current_layout = Group(
                    progress,
                    Text.from_markup(workers_str),
                    Text.from_markup(sources_str),
                    Text.from_markup(results_str)
                )
                live.update(current_layout)
                time.sleep(0.1)
