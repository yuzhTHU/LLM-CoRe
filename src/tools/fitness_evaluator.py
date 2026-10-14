import logging
import nd2py as nd
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import skew
from scipy.signal import welch
from .rollout_simulator import RolloutSimulator
from .parameter_fitter import ParameterFitter
from ..utils import split_ab, df2dict
from ..core import Dynamics, Individual

_logger = logging.getLogger(__name__)


class FitnessEvaluator:
    def __init__(self, args, assistant, edge_list, num_nodes):
        self.args = args
        self.assistant = assistant
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.fitter = ParameterFitter(args, edge_list, num_nodes)
        self.simulator = RolloutSimulator(args, edge_list, num_nodes)
        self.ground_truth_result = None
        self.baseline_result = None

    def set_fitness(self, indiv: Individual, df: pd.DataFrame, df_fit: pd.DataFrame = None, use_tqdm=False, random_state=42):
        ## 拟合参数
        if df_fit is not None:
            if indiv.fitness is not None:
                _logger.warning("Refitting dynamics for individual during set_fitness.")
            dynamics = self.assistant.triplet_to_dynamics(indiv.triplet)
            dynamics, mse = self.fitter.fit_parameters(dynamics, df_fit)
            indiv.dynamics = dynamics
            if not np.isfinite(mse):
                indiv.fitness = -np.inf
                indiv.detail = {"error": f"Invalid MSE {mse}"}
                return
        ## 评估指标
        evaluation = self.get_evaluation(indiv.dynamics, df, use_tqdm=use_tqdm, random_state=random_state)
        indiv.fitness, indiv.detail = self.get_fitness(evaluation)

    def get_fitness(self, evaluation):
        evaluation = evaluation.copy()
        ground_truth = self.ground_truth_result
        baseline = self.baseline_result
        # 评估预测能力相对提升
        evaluation["skill%"] = evaluation['skill'] / baseline['skill'] - 1
        # 评估交叉验证能力相对提升
        evaluation["skill_cv%"] = evaluation['skill_cv'] / baseline['skill_cv'] - 1
        # 评估轨迹 RMSE 相对降低
        evaluation["traj_rmse%"] = 1 - evaluation["traj_rmse"].mean() / baseline["traj_rmse"].mean()
        # 评估单步 RMSE 相对降低
        evaluation["rmse%"] = 1 - evaluation["rmse"].mean() / baseline["rmse"].mean()
        # 评估公式长度相对降低
        evaluation["length%"] = 1 - evaluation['length'] / baseline['length']
        # 评估与 Ground Truth 月度 Std 之间 MAE 的相对降低
        mae1 = np.abs(ground_truth['std'] - evaluation['std']).mean()
        mae2 = np.abs(ground_truth['std'] - baseline['std']).mean()
        evaluation["std_mae"] = mae1
        evaluation["std%"] = 1 - mae1 / mae2
        # 评估与 Ground Truth 月度 Skewness 之间 MAE 的相对降低
        mae1 = np.abs(evaluation["skewness"] - ground_truth["skewness"]).mean()
        mae2 = np.abs(baseline["skewness"] - ground_truth["skewness"]).mean()
        evaluation["skewness_mae"] = mae1
        evaluation["skewness%"] = 1 - mae1 / mae2
        # 评估与 Ground Truth 频谱密度 MAE 的相对降低
        mae1 = np.abs(evaluation["spectrum"] - ground_truth["spectrum"]).mean()
        mae2 = np.abs(baseline["spectrum"] - ground_truth["spectrum"]).mean()
        evaluation["spectrum_mae"] = mae1
        evaluation["spectrum%"] = 1 - mae1 / mae2
        # 评估与 Ground Truth Lead-Lag 相关性的 MAE 相对降低
        mae1 = np.abs(evaluation["lead_lag"] - ground_truth["lead_lag"]).mean()
        mae2 = np.abs(baseline["lead_lag"] - ground_truth["lead_lag"]).mean()
        evaluation["lead_lag_mae"] = mae1
        evaluation["lead_lag%"] = 1 - mae1 / mae2
        # 评估与 Ground Truth 事件预测的 MAE 相对降低
        mae1 = np.abs(evaluation["event_index"] - ground_truth["event_index"]).mean()
        mae2 = np.abs(baseline["event_index"] - ground_truth["event_index"]).mean()
        evaluation["event_index_mae"] = mae1
        evaluation["event_index%"] = 1 - mae1 / mae2
        # 评估提前预测月份的相对降低
        evaluation["forecast%"] = evaluation["forecast"].mean() / baseline["forecast"].mean() - 1
        # 其他指标
        evaluation["aic%"] = 1 - evaluation["aic"] / baseline["aic"]
        evaluation["bic%"] = 1 - evaluation["bic"] / baseline["bic"]
        evaluation["params%"] = 1 - evaluation["params"] / baseline["params"]
        evaluation["correlation_auc%"] = evaluation["correlation_auc"] / baseline["correlation_auc"] - 1
        evaluation["correlation_auc_cv%"] = evaluation["correlation_auc_cv"] / baseline["correlation_auc_cv"] - 1

        ## 计算适应度
        if self.args.score_func == 'linear':
            score_func = ScoreFunc.linear
        elif self.args.score_func == 'semi_linear':
            score_func = ScoreFunc.semi_linear
        elif self.args.score_func == 'log':
            score_func = ScoreFunc.log
        elif self.args.score_func == 'tanh':
            score_func = ScoreFunc.tanh
        else:
            raise ValueError(f"Unknown score_func: {self.args.score_func}")        
        fitness = (
            self.args.a_rmse * score_func(evaluation["rmse%"]) +
            self.args.a_trajrmse * score_func(evaluation["traj_rmse%"]) +
            self.args.a_correlation_auc_sum * score_func(evaluation["correlation_auc%"]).sum() +
            self.args.a_correlation_auc_cv_sum * score_func(evaluation["correlation_auc_cv%"]).sum() +
            self.args.a_skill_cv_sum * score_func(evaluation["skill_cv%"]).sum() +
            self.args.a_skill_cv_sum_inner * score_func(evaluation["skill_cv%"].sum()) +
            self.args.a_skill_cv_mean * score_func(evaluation["skill_cv%"]).mean() +
            self.args.a_ENSO_skill_cv * score_func(evaluation["skill_cv%"][("x", "ENSO")]) +
            self.args.a_ATL3_skill_cv * score_func(evaluation["skill_cv%"][("x", "ATL3")]) +
            self.args.a_SASD_skill_cv * score_func(evaluation["skill_cv%"][("x", "SASD")]) +
            self.args.a_Total_skill_cv * score_func(evaluation["skill_cv"].sum() / baseline['skill_cv'].sum() - 1) +
            self.args.a_skill_sum * score_func(evaluation["skill%"]).sum() + 
            self.args.a_skill_sum_inner * score_func(evaluation["skill%"].sum()) + 
            self.args.a_skill_mean * score_func(evaluation["skill%"]).mean() + 
            self.args.a_ENSO_skill * score_func(evaluation["skill%"][("x", "ENSO")]) +
            self.args.a_IOB_skill * score_func(evaluation["skill%"][("x", "IOB")]) +
            self.args.a_SASD_skill * score_func(evaluation["skill%"][("x", "SASD")]) +
            self.args.a_Total_skill * score_func(evaluation["skill"].sum() / baseline['skill'].sum() - 1) +
            self.args.a_params * score_func(evaluation["params%"]) +
            self.args.a_length * score_func(evaluation["length%"]) +
            self.args.a_std * score_func(evaluation["std%"]) +
            self.args.a_skewness * score_func(evaluation["skewness%"]) +
            self.args.a_spectrum * score_func(evaluation["spectrum%"]) +
            self.args.a_leadlag * score_func(evaluation["lead_lag%"]) + 
            self.args.a_forecast * score_func(evaluation["forecast%"]) + 
            self.args.a_contain_co2 * (1 if evaluation["contain_co2"] else 0)
        )
        if self.args.a_skill_after_2004 > 0:
            fitness += self.args.a_skill_after_2004 * score_func(evaluation["skill_after_2004"].sum() / baseline['skill_after_2004'].sum() - 1)

        ## 返回适应度和更新后的详细评估结果
        return fitness, evaluation

    def set_ground_truth_result(self, df: pd.DataFrame):
        self.ground_truth_result = {
            "std": self.get_monthly_std(df=df),
            "skewness": self.get_monthly_skewness(df=df),
            "spectrum": self.get_spectrum_density(df=df),
            "lead_lag": self.get_mode_correlation(df=df),
            "event_index": self.get_event_prediction(df=df),
        }

    def set_baseline_result(
        self, f: Dynamics, df: pd.DataFrame, df_fit: pd.DataFrame = None, use_tqdm=False
    ) -> pd.DataFrame:
        self.baseline_result = self.get_evaluation(f, df, use_tqdm=use_tqdm)

    def get_evaluation(
        self, f: Dynamics, df: pd.DataFrame, use_tqdm=True, random_state=42
    ) -> pd.DataFrame:
        """ 评估 f 在 df 上的各项指标, 如果指定了 df_fit 则先重新拟合 dynamics & noisy_dynamics
        - skill
        - traj_rmse
        - rmse
        - length
        - std
        - skewness
        - spectrum_density
        - lead_lag_corr
        - event_prediction
        - forecast_month
        - PREDICTION (无噪声模拟结果)
        - GENERATION (有噪声模拟结果)
        """
        # df = df.sort_index(axis=1) # 不然 df.drop(...) 会警告性能降低
        dt = df['t'].values[1, 0] - df['t'].values[0, 0]
        ## 无噪声预测
        PREDICTION = self.simulator.get_simulation_with_forcing(
            f, df[['x', 'h', 't']], dt, df.drop(columns=['x', 'h', 't'], level=0),
            max_month=self.args.max_month, tqdm_desc='Noise-Free Simulation' if use_tqdm else None
        )
        ## 带噪声生成
        n_member = getattr(self.args, 'n_member', 10)
        f_red = self.fitter.fit_red_noise(f, df, n_member=n_member, n=self.args.n_steps, max_lag=2, random_state=random_state)
        if False:
            df_sample = df.iloc[[0]].sample(n_member, replace=True).copy() # 解决碎片化问题
        else:
            df_sample = df.sample(n_member, random_state=random_state).copy()
        # df_sample = df_sample.assign(member=range(len(df_sample))).set_index("member", append=True)
        max_month = getattr(self.args, 'generation_max_month', 120)
        GENERATION = self.simulator.get_simulation_with_forcing(
            f_red, df_sample[['x', 'h', 't']], dt, df.drop(columns=['x', 'h', 't'], level=0),
            max_month=max_month-1, tqdm_desc='Noisy Generation' if use_tqdm else None
        )
        GENERATION = GENERATION.assign(member=GENERATION.groupby(level=1).cumcount()).reset_index().set_index(['time', 'member', 'lead'])

        ## 计算各项指标
        evaluation = {
            "skill": self.get_skills(PREDICTION=PREDICTION),
            "traj_rmse": self.get_traj_rmse(PREDICTION=PREDICTION),
            "rmse": self.get_rmse(f=f, df=df),
            "length": self.get_length(f=f),
            "std": self.get_monthly_std(GENERATION=GENERATION),
            "skewness": self.get_monthly_skewness(GENERATION=GENERATION),
            "spectrum": self.get_spectrum_density(GENERATION=GENERATION),
            "lead_lag": self.get_mode_correlation(GENERATION=GENERATION),
            "event_index": self.get_event_prediction(PREDICTION=PREDICTION),
            "forecast": self.get_forecast_month(PREDICTION=PREDICTION),
            "correlation_auc": self.get_correlation_auc(PREDICTION=PREDICTION),
            "aic": self.get_aic(f=f, df=df),
            "bic": self.get_bic(f=f, df=df),
            "params": self.get_params(f=f),
            "contain_co2": ('u_co2' in str(f.ddt_sst)) or ('u_co2' in str(f.ddt_wwv)),
            "PREDICTION": PREDICTION,
            "GENERATION": GENERATION,
        }
        if self.args.use_l3yo_cv:
            N = getattr(self.args, 'lnyo_cv_N', 3)
            N_bias = getattr(self.args, 'lnyo_cv_N_bias', 0)
            evaluation['PREDICTION_cv'] = self.get_prediction_lnyo_cv(f=f, df=df, dt=dt, N=N, N_bias=N_bias)
            evaluation['skill_cv'] = self.get_skills(PREDICTION=evaluation['PREDICTION_cv'])
            evaluation['correlation_auc_cv'] = self.get_correlation_auc(PREDICTION=evaluation['PREDICTION_cv'])
        elif self.args.use_nfold_cv:
            N = getattr(self.args, 'nfold_cv_N', 5)
            evaluation['PREDICTION_cv'] = self.get_prediction_nfold_cv(f=f, df=df, dt=dt, N=N)
            evaluation['skill_cv'] = self.get_skills(PREDICTION=evaluation['PREDICTION_cv'])
            evaluation['correlation_auc_cv'] = self.get_correlation_auc(PREDICTION=evaluation['PREDICTION_cv'])
        else:
            evaluation['PREDICTION_cv'] = evaluation['PREDICTION']
            evaluation['skill_cv'] = evaluation['skill']  # 使用 In-Sample Hindcast 的结果占位防止报错
            evaluation['correlation_auc_cv'] = evaluation['correlation_auc']  # 使用 In-Sample Hindcast 的结果占位防止报错
        if self.args.a_skill_after_2004 > 0:
            df_after_2004 = df.loc[df.index.year >= 2004]
            f_fit_after_2004, mse_after_2004 = self.fitter.fit_parameters(f, df_after_2004, nonlinear=False)
            PREDICTION_after_2004 = self.simulator.get_simulation_with_forcing(
                f_fit_after_2004, df_after_2004[['x', 'h', 't']], dt, df.drop(columns=['x', 'h', 't'], level=0), 
                max_month=self.args.max_month, tqdm_desc='Noise-Free Simulation' if use_tqdm else None
            )
            evaluation['PREDICTION_after_2004'] = PREDICTION_after_2004
            evaluation['skill_after_2004'] = self.get_skills(PREDICTION=PREDICTION_after_2004)
            evaluation['correlation_auc_after_2004'] = self.get_correlation_auc(PREDICTION=PREDICTION_after_2004)
        return evaluation

    def get_prediction_lnyo_cv(self, f: Dynamics, df: pd.DataFrame, dt: float, N=3, N_bias=0):
        """ LNYO-CV (Leave N years out Cross Validation) for skill evaluation """
        assert N % 2 == 1
        half_N = N // 2

        PREDICTION = []
        for year in range(df.index.year.min(), df.index.year.max() + 1):
            df_train = df.loc[(df.index.year < year + N_bias - half_N) | (df.index.year > year + N_bias + half_N)]
            df_test = df.loc[df.index.year == year]
            f_fit, mse = self.fitter.fit_parameters(f, df_train, nonlinear=False)
            PREDICTION.append(self.simulator.get_simulation_with_forcing(
                f_fit, df_test[['x', 'h', 't']], dt, df.drop(columns=['x', 'h', 't'], level=0),
                max_month=self.args.max_month, tqdm_desc=None))
        PREDICTION = pd.concat(PREDICTION, axis=0)
        return PREDICTION

    def get_prediction_nfold_cv(self, f: Dynamics, df: pd.DataFrame, dt: float, N=3, splitby=None):
        """ N-Fold Cross Validation) for skill evaluation """
        if splitby is None:
            tmin = df.index.min().to_timestamp().value
            tmax = df.index.max().to_timestamp().value
            steps = np.linspace(tmin, tmax, N+1)
            bins = pd.to_datetime(steps)
        else:
            bins = pd.to_datetime([df.index.min(), splitby, df.index.max()])
        group_ids = pd.cut(df.index.to_timestamp(), bins=bins, labels=False, include_lowest=True)
        df_parts = [group for _, group in df.groupby(group_ids)]
        if not getattr(self, '_has_logged_get_skills_nfold_cv', False):
            _logger.info(f"Split data into {N} folds for N-Fold CV: {bins}")
            for i, part in enumerate(df_parts):
                _logger.info(f"Fold {i}: {part.index.min()} to {part.index.max()}")
            self._has_logged_get_skills_nfold_cv = True
        PREDICTION = []
        for df_part in df_parts:
            df_train = df.loc[~df.index.isin(df_part.index)]
            df_test = df_part
            f_fit, mse = self.fitter.fit_parameters(f, df_train, nonlinear=False)
            PREDICTION.append(self.simulator.get_simulation_with_forcing(
                f_fit, df_test[['x', 'h', 't']], dt, df_test.drop(columns=['x', 'h', 't'], level=0),
                max_month=self.args.max_month, tqdm_desc=None))
        PREDICTION = pd.concat(PREDICTION, axis=0)
        return PREDICTION

    def get_skills(self, PREDICTION, return_raw=False) -> pd.Series:
        skills = {}
        corrs = []
        for var in ['x', 'h']:
            for mode in PREDICTION[var].columns:
                corr = [(0, 1)]
                for lead, group in PREDICTION[(var, mode)].sort_index().groupby("lead"):
                    smooth = group.rolling(3, min_periods=1, center=True).mean()
                    if lead == 0:
                        df0 = smooth
                    else:
                        true = df0.iloc[lead:].values
                        pred = smooth.iloc[:-lead].values
                        r = np.corrcoef(true, pred)[0, 1]
                        corr.append((lead, r))
                corr = pd.DataFrame(corr, columns=["lead", "r"])
                corrs.append(corr.assign(var=var, mode=mode).set_index(['lead', 'mode', 'var'])['r'])
                flag = (corr.r > 0.5) & (np.cumsum(corr.r > 0.5) == 1 + np.arange(len(corr)))
                s = flag.to_numpy().nonzero()[0][-1]
                skills[(var, mode)] = int(corr.loc[s, "lead"])
        if return_raw:
            return pd.concat(corrs).unstack().unstack()
        return pd.Series(skills)

    def get_correlation_auc(self, PREDICTION) -> pd.Series:
        corrs = self.get_skills(PREDICTION, return_raw=True)
        return corrs.clip(0, None).sum(axis=0)

    def get_traj_rmse(self, PREDICTION, lambda_decay=1.0):
        PREDICTION = (
            PREDICTION.copy()
            .reset_index()
            .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
            .set_index(['time', 'lead', 'stop'])
        )
        rmse_dict = {}
        for var in ['x', 'h']:
            df = PREDICTION[var]
            for mode in df.columns:
                df_pivot = (
                   df
                    .pivot_table(index='lead', columns='stop', values=mode)
                    .sort_index(axis='rows')
                    .sort_index(axis='columns')
                )
                true = df_pivot.iloc[0].values
                pred = df_pivot.iloc[1:]
                rmse_list = (pred - true).pow(2).mean(axis=1).pow(0.5).values
                w = lambda_decay ** (len(rmse_list) - np.arange(len(rmse_list)))
                w /= w.sum()
                rmse = (rmse_list * w).sum()
                rmse_dict[(var, mode)] = rmse.item()
        return pd.Series(rmse_dict)

    def get_rmse(self, f: Dynamics, df: pd.DataFrame):
        A, b_sst, b_wwv = split_ab(df)
        X = df2dict(A)
        f_sst = f.ddt_sst.eval(X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        f_wwv = f.ddt_wwv.eval(X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        rmse_sst = np.mean((f_sst - b_sst)**2, axis=0)**0.5
        rmse_wwv = np.mean((f_wwv - b_wwv)**2, axis=0)**0.5
        return pd.DataFrame({'x': rmse_sst, 'h': rmse_wwv}).stack().swaplevel().loc[['x', 'h']]

    def get_length(self, f: Dynamics):
        return len(f.ddt_sst) + len(f.ddt_wwv)

    def get_event_prediction(self, PREDICTION=None, df=None):
        if PREDICTION is None: # Return the Ground-Truth
            assert df is not None
            return pd.Series({
                "2023 Dec": 2.0,
                "2015 Dec": 2.3,
                "1982 Dec": 2.2,
            })
        events = {}
        for key, idx, col in [
            ("2023 Dec", ('2022-12', 12), ('x', 'ENSO')),
            ("2015 Dec", ('2014-12', 12), ('x', 'ENSO')),
            ("1982 Dec", ('1981-12', 12), ('x', 'ENSO')),
        ]:
            if idx in PREDICTION.index and col in PREDICTION.columns:
                events[key] = PREDICTION.loc[idx, col].item()
            else:
                events[key] = np.nan
        return pd.Series(events)

    def get_monthly_std(self, GENERATION=None, df=None, member_average=True, var='x', mode='ENSO'):
        if (GENERATION is not None) and (df is not None):
            raise ValueError("Either GENERATION or df must be provided, but not both.")
        elif df is not None:
            # Return the Ground-Truth
            REAL = df[(var, mode)]
            REAL.index = pd.Index(REAL.index.strftime('%b'), name='month')
            return REAL.groupby(REAL.index, sort=False).std()
        elif GENERATION is not None:
            results = (
                GENERATION
                .reset_index()
                .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
                .pipe(lambda df: df.assign(month=df['stop'].dt.strftime('%b')))
                .set_index(['member', 'month']).sort_index()
                [(var, mode)]
                .groupby(['member', 'month']).std().unstack()
                .reindex(columns=[
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                ])
            )
            return results.mean(axis=0) if member_average else results
        else:
            raise ValueError("Either GENERATION or df must be provided.")

    def get_monthly_skewness(self, GENERATION=None, df=None, member_average=True, var='x', mode='ENSO'):
        if (GENERATION is not None) and (df is not None):
            raise ValueError("Either GENERATION or df must be provided, but not both.")
        elif df is not None:
            # Return the Ground-Truth
            REAL = df[(var, mode)]  # (time,)
            REAL.index = pd.Index(REAL.index.strftime('%b'), name='month')
            return REAL.groupby(REAL.index, sort=False).apply(skew)
        elif GENERATION is not None:
            results = (
                GENERATION  # (time, member)
                .reset_index()
                .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
                .pipe(lambda df: df.assign(month=df['stop'].dt.strftime('%b')))
                .set_index(['member', 'month'])
                [(var, mode)]
                .groupby(['member', 'month']).apply(skew).unstack()
                .reindex(columns=[
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                ])
            )
            return results.mean(axis=0) if member_average else results
        else:
            raise ValueError("Either GENERATION or df must be provided.")

    def get_spectrum_density(self, GENERATION=None, df=None, member_average=True, var='x', mode='ENSO'):
        if (GENERATION is not None) and (df is not None):
            raise ValueError("Either GENERATION or df must be provided, but not both.")
        elif df is not None:
            # Return the Ground-Truth
            REAL = df[(var, mode)]  # (time,)
            f, y = welch(REAL.values, fs=12, nperseg=120)
            f = np.round(f, 4)
            return pd.Series(y, index=f)
        elif GENERATION is not None:
            df_list = []
            for member, group in GENERATION[(var, mode)].groupby('member'):
                f, psd = welch(group.sort_index().values, fs=12, nperseg=120)  # °C² * yr
                f = np.round(f, 4)
                df_list.append(pd.DataFrame({'member': member, 'f': f, 'psd': psd}))
            results = pd.concat(df_list, axis=0).pivot_table(index='member', columns='f', values='psd')
            return results.mean(axis=0) if member_average else results
        else:
            raise ValueError("Either GENERATION or df must be provided.")

    def get_mode_correlation(self, GENERATION=None, df=None, member_average=True):
        if (GENERATION is not None) and (df is not None):
            raise ValueError("Either GENERATION or df must be provided, but not both.")
        elif df is not None:
            # Return the Ground-Truth
            source = []
            target = []
            pairs = []
            for mode_i in df[['x', 'h']].columns:
                for mode_j in df[['x', 'h']].columns:
                    # if mode_i == mode_j: continue
                    source.append(mode_i)
                    target.append(mode_j)
                    pairs.append((mode_i, mode_j))
            Xi = df[source].set_axis(pairs, axis='columns').rename_axis(columns='pairs')
            Xj = df[target].set_axis(pairs, axis='columns').rename_axis(columns='pairs')
            df_list = []
            for lead in range(-24, 24 + 1):
                corr = Xi.corrwith(Xj.shift(lead), method='pearson')
                df_list.append(corr.to_frame('corr').assign(lead=lead))
            return pd.concat(df_list, axis=0).reset_index().set_index(['pairs', 'lead'])['corr']
        elif GENERATION is not None:
            source = []
            target = []
            pairs = []
            for mode_i in GENERATION[['x', 'h']].columns:
                for mode_j in GENERATION[['x', 'h']].columns:
                    # if mode_i == mode_j: continue
                    source.append(mode_i)
                    target.append(mode_j)
                    pairs.append((mode_i, mode_j))
            Xi = (
                GENERATION[source]
                .set_axis(pairs, axis="columns")
                .rename_axis(columns="pairs")
                .reset_index(level=0, drop=True)
                .unstack(level=0)
            )
            Xj = (
                GENERATION[target]
                .set_axis(pairs, axis="columns")
                .rename_axis(columns="pairs")
                .reset_index(level=0, drop=True)
                .unstack(level=0)
            )
            df_list = []
            for lead in range(-24, 24 + 1):
                corr = Xi.corrwith(Xj.shift(lead), method='pearson')
                df_list.append(corr.to_frame('corr').assign(lead=lead))
            results = pd.concat(df_list, axis=0).reset_index().pivot(index="member", columns=["pairs", "lead"], values="corr")
            return results.mean(axis=0) if member_average else results
        else:
            raise ValueError("Either GENERATION or df must be provided.")

    def get_forecast_month(self, PREDICTION, return_raw=False, group_by_event=True, repeat_month=3, keep_all_events=False, var='x', mode='ENSO'):
        df_pivot = (
            PREDICTION
            .reset_index()
            .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
            .set_index(['time', 'lead', 'stop'])
            [var]
            .pivot_table(index='lead', columns='stop', values=mode)
            .sort_index(axis='rows')
            .sort_index(axis='columns')
        )
        # 离散化为 El Niño, La Niña, Neutral 事件
        df_event = pd.DataFrame(
            np.select(
                [df_pivot > 0.5, df_pivot < -0.5, (df_pivot >= -0.5) & (df_pivot <= 0.5)], 
                ['El Niño', 'La Niña', 'Neutral'], default='Unknown'
            ), index=df_pivot.index, columns=df_pivot.columns.strftime('%Y %b')
        )
        # 获取每个事件的最早预测时间
        is_correct = (df_event == df_event.iloc[0])
        for i in range(1, repeat_month): # 连续 N 个月预测正确
            is_correct &= is_correct.shift(-i)
        df_pred = pd.DataFrame([
            is_correct.where(is_correct) # 将 False 的位置设置为 Nan
            .apply(lambda col: col.last_valid_index()) # 提取最后一个非 NaN 的位置
            .astype(float)
            .fillna(0) # 如果全是 NaN，则填充为 0.0
            .rename("lead")
        ]).T.assign(event=df_event.iloc[0]) # index=stop, columns=[lead, event]
        # 只保留 El Niño & La Niña 事件
        if not keep_all_events:
            df_pred = df_pred[df_pred['event'].isin(['El Niño', 'La Niña'])]
        # 返回结果
        if return_raw:
            return df_pred
        elif group_by_event:
            return df_pred.groupby('event', sort=True)['lead'].mean()
        else:
            return df_pred['lead'].mean().item()

    def get_aic(self, f: Dynamics, df: pd.DataFrame):
        n = len(df)
        A, b_sst, b_wwv = split_ab(df)
        f_sst = f.ddt_sst.eval(A, edge_list=self.edge_list, num_nodes=self.num_nodes)
        f_wwv = f.ddt_wwv.eval(A, edge_list=self.edge_list, num_nodes=self.num_nodes)
        rss_sst = np.sum((f_sst - b_sst.values) ** 2)
        rss_wwv = np.sum((f_wwv - b_wwv.values) ** 2)
        k_sst = sum([np.sum(i.value != 0.0) for i in f.ddt_sst.iter_preorder() if isinstance(i, nd.Number)])
        k_wwv = sum([np.sum(i.value != 0.0) for i in f.ddt_wwv.iter_preorder() if isinstance(i, nd.Number)])
        aic_sst = n * np.log(rss_sst / n) + 2 * k_sst
        aic_wwv = n * np.log(rss_wwv / n) + 2 * k_wwv
        return aic_sst + aic_wwv

    def get_bic(self, f: Dynamics, df: pd.DataFrame):
        n = len(df)
        A, b_sst, b_wwv = split_ab(df)
        f_sst = f.ddt_sst.eval(A, edge_list=self.edge_list, num_nodes=self.num_nodes)
        f_wwv = f.ddt_wwv.eval(A, edge_list=self.edge_list, num_nodes=self.num_nodes)
        rss_sst = np.sum((f_sst - b_sst.values) ** 2)
        rss_wwv = np.sum((f_wwv - b_wwv.values) ** 2)
        k_sst = sum([np.sum(i.value != 0.0) for i in f.ddt_sst.iter_preorder() if isinstance(i, nd.Number)])
        k_wwv = sum([np.sum(i.value != 0.0) for i in f.ddt_wwv.iter_preorder() if isinstance(i, nd.Number)])
        bic_sst = n * np.log(rss_sst / n) + k_sst * np.log(n)
        bic_wwv = n * np.log(rss_wwv / n) + k_wwv * np.log(n)
        return bic_sst + bic_wwv

    def get_params(self, f: Dynamics):
        k_sst = sum([np.sum(i.value != 0.0) for i in f.ddt_sst.iter_preorder() if isinstance(i, nd.Number)])
        k_wwv = sum([np.sum(i.value != 0.0) for i in f.ddt_wwv.iter_preorder() if isinstance(i, nd.Number)])
        return k_sst + k_wwv

class ScoreFunc:
    @staticmethod
    def linear(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(np.isnan(x), -np.inf, x)  # Replace NaN with -inf
            return x
    
    @staticmethod
    def semi_linear(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(np.isnan(x), -np.inf, x)  # Replace NaN with -inf
            return np.where(x > 0, 0.5*x, x)
    
    @staticmethod
    def log(x, alpha=1.0):
        """ 使用对数压制正数部分的 x, alpha 越小压制越强 """
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(np.isnan(x), -np.inf, x)  # Replace NaN with -inf
            return np.where(x<=0, x, alpha*np.log1p(x/alpha))
    
    @staticmethod
    def tanh(x, alpha=1.0):
        """ 使用 tanh 压制正数部分的 x, 当 x 足够大时进一步增长毫无意义, 最大分数被限制为 alpha """
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(np.isnan(x), -np.inf, x)  # Replace NaN with -inf
            return np.where(x<=0, x, alpha*np.tanh(x/alpha))
