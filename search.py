import re
import sys
import json
import yaml
import shlex
import random
import dotenv
import signal
import logging
import datetime
import nd2py as nd
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from nd2py.utils import init_logger
from setproctitle import setproctitle
from src.utils import add_negation_flags, tag2ansi, log_exception, my_uuid, my_hash
from src.core import Triplet, Individual
from src.assistant import Assistant
from src.ai_scientist import AIScientist
from src.ai_scientist_parallel import AIScientistParallel

import warnings
# warnings.simplefilter('error', RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="nd2py")

dotenv.load_dotenv()
_logger = logging.getLogger('src')


def main(args):
    ## Load Assistant
    assistant = Assistant(args)
    ## Load Data
    # 加载变量
    variables = assistant.load_variables()
    # 加载数据
    df_all = assistant.load_data(normalize_co2=args.normalize_co2)
    # 划分数据
    if True:
        train_start, train_end = args.train_period
        test_start, test_end = args.test_period
        train_start = pd.Period(train_start, freq='M') if train_start != 'all' else df_all.index[0]
        train_end = pd.Period(train_end, freq='M') if train_end != 'all' else df_all.index[-1]
        test_start = pd.Period(test_start, freq='M') if test_start != 'all' else df_all.index[0]
        test_end = pd.Period(test_end, freq='M') if test_end != 'all' else df_all.index[-1]
        df_train = df_all.loc[train_start:train_end]
        df_test = df_all.loc[test_start:test_end]
    elif False:
        df_train = df_all.loc[pd.Period(year=1979, month=1, freq='M'):pd.Period(year=2022, month=12, freq='M')]
        df_test = df_all.loc[pd.Period(year=1979, month=1, freq='M'):pd.Period(year=2022, month=12, freq='M')]
    else:
        df_train = df_all.loc[pd.Period(year=1979, month=1, freq='M'):pd.Period(year=1999, month=12, freq='M')]
        df_test = df_all.loc[pd.Period(year=2000, month=1, freq='M'):pd.Period(year=2022, month=12, freq='M')]
    # 准备拓扑
    edge_list, num_nodes = assistant.load_topology()
    # 打印日志
    _logger.info(
        f"Data loaded: {len(df_all)} samples, "
        f"where {len(df_train)}({len(df_train)/len(df_all):.2%}) "
        f"from {df_train.index[0]} to {df_train.index[-1]} for training, "
        f"and {len(df_test)}({len(df_test)/len(df_all):.2%}) "
        f"from {df_test.index[0]} to {df_test.index[-1]} for testing."
    )
    _logger.info(
        f"Variables: " + ", ".join(f'{var_name}({len(sub_df)})' for var_name, sub_df in df_all.T.groupby(level=0)) + "."
    )
    ## Load Seed Individuals
    seed_individuals = assistant.load_seed_individuals()
    for coeff in (iteration := iter(args.create_seed_individuals)):
        f_sst = next(iteration)
        f_wwv = next(iteration)
        triplet = Triplet(
            coeff=nd.parse(coeff, variables=variables),
            f_sst=nd.parse(f_sst, variables=variables),
            f_wwv=nd.parse(f_wwv, variables=variables),
        )
        indiv = Individual(
            uuid=my_uuid(np.random.default_rng(my_hash(f'coeff={coeff}, f_sst={f_sst}, f_wwv={f_wwv}'))),
            n_iter=0,
            source='indiv_to_evaluate',
            parents=[],
            triplet=triplet,
            program=assistant.triplet_to_program(triplet),
            dynamics=assistant.triplet_to_dynamics(triplet),
        )
        seed_individuals.append(indiv)


    ## Initialize Runner
    Runner = AIScientistParallel if args.parallel else AIScientist
    kwargs = dict(
        args=args,
        assistant=assistant,
        variables=variables,
        binary=[
            getattr(nd, name) for name in args.binary
            # nd.Add, nd.Sub, nd.Mul, nd.Div
        ],
        unary=[
            getattr(nd, name) for name in args.unary
            # nd.Sin, nd.Cos,
            # nd.Exp, nd.LogAbs,
            # nd.Sigmoid, nd.Tanh,
            # nd.Pow2, nd.Pow3, nd.SqrtAbs,
            # nd.Aggr, nd.Readout, nd.Sour, nd.Targ,
        ],
        edge_list=edge_list,
        num_nodes=num_nodes,
        seed_individuals=seed_individuals,
    )
    _logger.note(f"Initializing Runner with args: {kwargs}")
    est = Runner(**kwargs)
    # Set Clip Bounds
    est.evaluator.simulator.set_clip_bounds(df_all, clip_factor=args.clip_factor)
    # Set Ground Truth Results
    _logger.info(f"Calculating ground truth result on test data...")
    est.evaluator.set_ground_truth_result(df_test)
    # Set XRO Baseline Results
    _logger.info(f"Evaluating XRO on test data as baseline...")
    baseline_indiv = assistant.load_baseline_indiv()
    baseline_indiv.dynamics, mse = est.evaluator.fitter.fit_parameters(baseline_indiv.dynamics, df_train, nonlinear=True)
    est.evaluator.set_baseline_result(baseline_indiv.dynamics, df_test, use_tqdm=True)
    if True:
        baseline_indiv.fitness, baseline_indiv.detail = est.evaluator.get_fitness(est.evaluator.baseline_result)
        # est.evaluator.set_fitness(baseline_indiv, df_test, use_tqdm=True)
        baseline_log = est.get_log_for_indiv(baseline_indiv)
        baseline_detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in baseline_log.items())
        _logger.info(tag2ansi(f"Fit Baseline Individual with MSE={mse}. Details: {baseline_detail}"))

    # Setup Signal Handlers
    def signal_handler(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals

    ## Start Fitting
    _logger.info("Starting the fitting process...")
    try:
        # Run
        est.fit(df_train, df_test)
    except KeyboardInterrupt:
        _logger.note("KeyboardInterrupt detected, stopping the fitting process.")
    except Exception as e:
        _logger.error("\n\n" + log_exception(e))
        if args.debug: raise e from e
    finally:
        if est.best_model is not None:
            log = est.get_log_for_indiv(est.best_model)
            detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
            _logger.note(tag2ansi(f"Best individual: {detail}"))
            if args.save_path is not None:
                (Path(args.save_path) / 'best_model.py').write_text(est.best_model.program)
                _logger.note(f"Best model program saved to {Path(args.save_path) / 'best_model.py'}.")
        logs = []
        rank = 0
        curr = (None, None)
        for indiv in sorted(est.population, key=lambda x: (x.fitness, -x.n_iter), reverse=True):
            if rank >= args.log_topk:
                break
            new_sst = indiv.dynamics.ddt_sst.eval(df_test, edge_list=edge_list, num_nodes=num_nodes)
            new_wwv = indiv.dynamics.ddt_wwv.eval(df_test, edge_list=edge_list, num_nodes=num_nodes)
            cur_sst, cur_wwv = curr
            sst_is_close = (cur_sst is not None) and (new_sst is not None) and (np.isclose(new_sst, cur_sst, atol=1e-6).mean() > 0.99)
            wwv_is_close = (cur_wwv is not None) and (new_wwv is not None) and (np.isclose(new_wwv, cur_wwv, atol=1e-6).mean() > 0.99)
            if sst_is_close and wwv_is_close:
                logs.append(tag2ansi(f"Top-{rank} individual: {indiv.uuid} from [pink bold]{indiv.source}[reset]({', '.join(indiv.parents)}) at Iter [pink]{indiv.n_iter}[reset]"))
            else:
                rank += 1
                curr = (new_sst, new_wwv)
                log = est.get_log_for_indiv(indiv)
                detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
                logs.append(tag2ansi(f"Top-{rank} individual: {detail}"))
        
            # try:
            #     df_eval = df_all.loc[pd.Period(year=2021, month=1, freq='M'):pd.Period(year=2024, month=12, freq='M')]
            #     est.evaluator.set_fitness(indiv, df_eval, df_fit=None)
            #     log = est.get_log_for_indiv(indiv)
            #     detail = " | ".join(f"[#66CCFF]{k}[reset]={v}" for k, v in log.items())
            #     _logger.note(tag2ansi(f"Evaluated on df_eval: {detail}"))
            # except Exception as e:
            #     _logger.warning(f'Failed to evaluate indiv on df_eval since {e}')
        if len(logs):
            _logger.note("\n" + "\n".join(logs[::-1]))

        _logger.note(tag2ansi(f"Running process finished. Re-run: [green]{args.command}[reset]"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Run AIScientist to discover Global Climate Mode Dynamics")
    parser.add_argument("--name", type=str, default=f"AIScientist", help="Used to generate exp-name")
    parser.add_argument('--setting', type=str, choices=[path.stem for path in Path('./src/prompts').glob('*.md')], default='default', help='Which prompt/setting to use')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for logging and saving purposes')
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model to use for program generation")
    parser.add_argument("--no_llm", dest='llm_model', action='store_const', const=None)
    parser.add_argument("--llm_provider", type=str, default=None, help="在多个服务商提供了同名LLM时, 可以指定服务提供商名称")
    parser.add_argument("--seed_individuals", type=str, nargs="+", default=[], choices=['linear', 'XRO', 'temp_25_30', 'aot_co2_15_30', 'no_forcing_25_30', 'co2_29_29', 'co2_30_30', 'co2_320_230_368', 'co2_327_240_337', 'log_co2_12'], help="Names of seed individuals to load")
    parser.add_argument("--load_seed_individuals", type=str, nargs="+", default=[], help="Path of seed populations to load")
    parser.add_argument("--create_seed_individuals", type=str, nargs="+", default=[], help="Triplet of seed populations to create")
    parser.add_argument('--use_seed_individuals_only', action='store_true', default=False, help="Whether to use only seed individuals for evolution without generating new random individuals")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of parallel workers for fitness evaluation, 0 is the number of CPU cores")
    parser.add_argument("--parallel", action='store_true', default=True, help="Use multi-processing rather than single process")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout (in seconds) for each fitness evaluation, only valid when num-workers >= 1")
    parser.add_argument("--population_size", type=int, default=200, help="Number of candidate hypothesis during evolution")
    parser.add_argument("--n_iter", type=int, default=300, help="Number of iterations for the evolution process")
    parser.add_argument("--programs_per_prompt", type=int, default=2, help="Number of demo programs to generate per prompt")
    parser.add_argument("--generate_per_prompt", type=int, default=8, help="Number of programs to generate per prompt")
    parser.add_argument("--expire_iter", type=int, default=1, help="Keep elitism individuals for additional iterations")
    parser.add_argument("--keep_best", action="store_true", help="Whether to keep the best individuals during whole evolution")
    parser.add_argument("--random_state", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--score_func", type=str, default='tanh', choices=['linear', 'semi_linear', 'log', 'tanh'])
    parser.add_argument("--a_rmse", type=float, default=0.0, help="Weight for rmse%% in the fitness function")
    parser.add_argument("--a_trajrmse", type=float, default=0.0, help="Weight for trajrmse%% in the fitness function")
    parser.add_argument("--a_correlation_auc_sum", type=float, default=0.0, help="Weight for sum(correlation_auc%%) in the fitness function")
    parser.add_argument("--a_correlation_auc_cv_sum", type=float, default=0.0, help="Weight for sum(correlation_auc_cv%%) in the fitness function")
    parser.add_argument('--a_skill_cv_sum', type=float, default=1.0, help="Weight for sum(skill_cv%%) in the fitness function")
    parser.add_argument('--a_skill_cv_sum_inner', type=float, default=0.0, help="Weight for sum(skill_cv%%) in the fitness function")
    parser.add_argument('--a_skill_cv_mean', type=float, default=0.0, help="Weight for mean(skill_cv%%) in the fitness function")
    parser.add_argument('--a_ENSO_skill_cv', type=float, default=0.5, help="Weight for ENSO skill_cv%% in the fitness function")
    parser.add_argument('--a_ATL3_skill_cv', type=float, default=0.5, help="Weight for ATL3 skill_cv%% in the fitness function")
    parser.add_argument('--a_SASD_skill_cv', type=float, default=0.5, help="Weight for SASD skill_cv%% in the fitness function")
    parser.add_argument('--a_Total_skill_cv', type=float, default=0.0, help="Weight for Total skill_cv%% in the fitness function")
    parser.add_argument("--a_skill_sum_inner", type=float, default=0.0, help="Weight for sum(skill%%) in the fitness function")
    parser.add_argument("--a_skill_sum", type=float, default=0.0, help="Weight for sum(skill%%) in the fitness function")
    parser.add_argument("--a_skill_mean", type=float, default=0.0, help="Weight for mean(skill%%) in the fitness function")
    parser.add_argument("--a_ENSO_skill", type=float, default=0.0, help="Additional weight for ENSO skill%% in the fitness function")
    parser.add_argument("--a_IOB_skill", type=float, default=0.0, help="Additional weight for IOB skill%% in the fitness function")
    parser.add_argument("--a_SASD_skill", type=float, default=0.0, help="Additional weight for SASD skill%% in the fitness function")
    parser.add_argument("--a_Total_skill", type=float, default=0.0, help="Weight for Total skill%% in the fitness function")
    parser.add_argument('--a_skill_after_2004', type=float, default=0.0, help="Weight for Total skill%% after 2004 in the fitness function")
    parser.add_argument("--a_length", type=float, default=1.0, help="Weight for length%% in the fitness function")
    parser.add_argument("--a_params", type=float, default=1.0, help="Weight for number of parameters in the fitness function")
    parser.add_argument("--a_std", type=float, default=0.0, help="Weight for std%% in the fitness function")
    parser.add_argument("--a_skewness", type=float, default=0.0, help="Weight for skewness%% in the fitness function")
    parser.add_argument("--a_spectrum", type=float, default=0.0, help="Weight for spectrum%% in the fitness function")
    parser.add_argument("--a_leadlag", type=float, default=0.0, help="Weight for leadlag%% in the fitness function")
    parser.add_argument("--a_forecast", type=float, default=0.0, help="Weight for forecast%% in the fitness function")
    parser.add_argument("--a_contain_co2", type=float, default=0.0, help="Weight for whether the formula contains CO2 in the fitness function")
    parser.add_argument("--save_checkpoint_per_iter", type=int, default=1, help="Save checkpoint every N iterations")
    parser.add_argument('--max_params', type=int, default=5, help='Maximum number of nonlinear parameters allowed to fit')
    parser.add_argument("--save_dir", type=str, default="./logs/search", help="Directory to save logs and models")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging")
    parser.add_argument("--log_per_iter", type=int, default=1, help="Log information every N iterations")
    parser.add_argument("--log_per_sec", type=float, default=None, help="Log detailed info every N seconds")
    parser.add_argument("--elitism_k", type=int, default=10, help="Number of elite individuals to carry over to the next generation")
    parser.add_argument("--full_prob", type=float, default=0.5)
    parser.add_argument("--depth_range", type=int, nargs=2, default=[2, 6], help="Min and max depth of generated programs")
    parser.add_argument("--const_range", type=float, nargs=2, default=[-1.0, 1.0], help="Min and max value of constants in generated programs")
    parser.add_argument("--tournament_size", type=int, default=20, help="Number of individuals in each tournament selection")
    parser.add_argument("--p_crossover", type=float, default=0.90, help="Probability of crossover operation")
    parser.add_argument("--p_subtree_mutation", type=float, default=0.05, help="Probability of subtree mutation operation")
    parser.add_argument("--p_hoist_mutation", type=float, default=0.05, help="Probability of hoist mutation operation")
    parser.add_argument("--p_point_mutation", type=float, default=0.0, help="Probability of point mutation operation")
    parser.add_argument("--p_term_reduction", type=float, default=0.0, help="Probability of term reduction operation")
    parser.add_argument("--p_collapse_reduction", type=float, default=0.0, help="Probability of collapse reduction operation")
    parser.add_argument("--p_unwrap_reduction", type=float, default=0.0, help="Probability of unwrap reduction operation")
    parser.add_argument("--p_modal_reduction", type=float, default=0.0, help="Probability of modal reduction operation")
    parser.add_argument("--temperature_init", type=float, default=0.1, help="Initial temperature for LLM sampling")
    parser.add_argument("--temperature_period", type=int, default=100, help="Period for temperature decay in LLM sampling")
    parser.add_argument("--intergrator", type=str, default="euler", choices=["euler", "midpoint", "trapezoidal", "rk4"])
    parser.add_argument("--n_member", type=int, default=10, help="Number of ensemble members for noisy simulation")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of sub-time steps to simulate")
    parser.add_argument("--max_month", type=int, default=30, help="Maximum number of months to simulate")
    parser.add_argument("--clip_factor", type=float, default=4.5, help="Clipping factor for model outputs")
    parser.add_argument("--no_clip", action='store_const', const=None, dest='clip_factor', help="Disable clipping of model outputs")
    parser.add_argument("--update_t_within_step", action="store_true", help="Update time variable within each integration step", default=True)
    parser.add_argument('--replay_path', type=str, default=None, help="Path to the replay files for LLM generation, e.g., /path/to/response")
    parser.add_argument('--explain_parents', action='store_true', default=True, help="Whether to use LLM to explain parent individuals before generating children")
    parser.add_argument('--explain_children', action='store_true', default=False, help="Whether to use LLM to explain child individuals after generation")
    parser.add_argument('--coeff_max_len_for_GP', type=int, default=20, nargs='?', help="避免GP产生太长的公式")
    parser.add_argument('--f_sst_max_len_for_GP', type=int, default=25, nargs='?', help="避免GP产生太长的公式")
    parser.add_argument('--f_wwv_max_len_for_GP', type=int, default=25, nargs='?', help="避免GP产生太长的公式")
    parser.add_argument('--no_mode_vars_for_GP', action='store_true', default=False, help="在GP中不使用mode变量")
    parser.add_argument('--train_period', type=str, nargs=2, default=['all', 'all'], help="训练数据的时间范围, 格式如: 1979-01 2022-12")
    parser.add_argument('--test_period', type=str, nargs=2, default=['all', 'all'], help="测试数据的时间范围, 格式如: 1979-01 2022-12")
    parser.add_argument('--log_subbest', action='store_true', default=False, help="在迭代中途打印最优个体信息")
    parser.add_argument('--binary', type=str, nargs='+', default=['Add', 'Sub', 'Mul', 'Div'], help="Binary operators to use")
    parser.add_argument('--unary', type=str, nargs='+', default=['Sin', 'Cos', 'Exp', 'LogAbs', 'Sigmoid', 'Tanh', 'Pow2', 'Pow3', 'SqrtAbs', 'Aggr', 'Readout', 'Sour', 'Targ'], help="Unary operators to use")
    parser.add_argument('--log_topk', type=int, default=10, help="在结束时打印前K个最优个体信息")
    parser.add_argument('--use_z20_as_ohc', action='store_true', default=False, help="使用WWV作为OHC指标")
    parser.add_argument('--use_tpo_for_enso_ohc', action='store_true', default=True, help="使用赤道太平洋而非 Nino3.4 区域的OHC")
    parser.add_argument('--use_l3yo_cv', action='store_true', default=True, help="使用基于L3YO的交叉验证技能评分")
    parser.add_argument('--use_nfold_cv', action='store_true', default=False, help="使用基于N折的交叉验证技能评分")
    parser.add_argument('--evaluate_before_and_after_2000', action='store_true', default=False, help="分别在2000年前后评估个体的表现")
    parser.add_argument('--normalize_co2', action='store_true', default=False, help="是否对co2变量进行归一化处理")
    parser.add_argument('--detrend_quadratic', action='store_true', default=True, help="是否对SST和OHC变量去除二次趋势")
    parser.add_argument('--data_source', type=str, default='oras5', choices=['oras5', 'soda', 'godas'], help="默认使用 ORAS5 数据源")
    parser.add_argument('--damp', type=float, default=1e-1, help="damp parameter for scipy.sparse.linalg.lsqr when fitting linear coefficients")
    parser = add_negation_flags(parser)
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Set Random State
    if args.random_state is None:
        args.random_state = random.randint(0, 2**31 - 1)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))

    ## Init Logger
    init_logger(
        "src", 
        exp_name=args.exp_name, 
        log_file=save_path / "info.log", 
        info_level='debug' if args.debug else 'info',
        show_lineno_for_all_levels=args.debug,
    )
    nd2py_logger = logging.getLogger("nd2py")
    nd2py_logger.setLevel(_logger.level)
    nd2py_logger.handlers = _logger.handlers.copy()

    ## Warn Unknown Args
    if len(unknown) > 0:
        _logger.warning(f"Unknown arguments: {unknown}")
        
    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    # Start the main process
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
