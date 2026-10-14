import logging
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class DataLoader(FrozenMixin, ABC):
    SUBCLASS_REGISTRY = {}

    def __init__(self, assistant):
        self.assistant = assistant

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls.SUBCLASS_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, assistant, name) -> 'DataLoader':
        sub_class = cls.SUBCLASS_REGISTRY.get(name, 'default')
        return sub_class(assistant)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


@DataLoader.register('default')
class DataLoader_v0(DataLoader):
    def load(self, rescale_ohc=True, normalize_co2=True, normalize_aot=True, normalize_tsi=True):
        data_source = getattr(self.assistant.args, 'data_source', 'oras5')
        assert data_source in ['oras5', 'soda', 'godas'], f"args.data_source should be 'oras5', 'soda', or 'godas', found {data_source}."
        # SST
        df_sst = pd.read_csv(f'data/{data_source}_sst.csv')
        df_sst = df_sst.assign(time=pd.PeriodIndex.from_fields(year=df_sst.pop('year'), month=df_sst.pop('month'), freq='M')).set_index('time')
        df_sst = df_sst[self.assistant.modes] # 规范化 modes 顺序, 并移除不需要的 modes
        # 对每一列去除逐月均值
        tmp = df_sst[df_sst.index < '2025-10']
        monthly_mean = tmp.groupby(tmp.index.month)[self.assistant.modes].mean()
        monthly_mean.index.name = 'month'
        df_sst = (df_sst[self.assistant.modes].set_index(df_sst.index.month.set_names('month'), append=True) - monthly_mean).reset_index(level='month', drop=True)
        # 对每一列去除二次趋势
        if getattr(self.assistant.args, 'detrend_quadratic', True): 
            for mode in self.assistant.modes:
                year = df_sst.index.year
                w = (year >= 1979) & (year <= 2022)
                t = np.arange(len(df_sst))
                coeffs = np.polyfit(t, df_sst[mode].values.astype(float), 2, w=w)
                _logger.debug(f"Detrending {mode} SST with coefficients: {coeffs}")
                trend = np.polyval(coeffs, t)
                df_sst[mode] = df_sst[mode] - trend
            _logger.debug(f"Finished detrending SST modes: {self.assistant.modes}")
        else:
            _logger.debug(f"Skipping quadratic detrending for SST modes: {self.assistant.modes}")


        # OHC (Z20)
        if self.assistant.args.use_z20_as_ohc:
            df_ohc = pd.read_csv(f'data/{data_source}_z20.csv')
        else:
            df_ohc = pd.read_csv(f'data/{data_source}_ohc.csv')
            if rescale_ohc:
                df_ohc = df_ohc.set_index(['year', 'month', 'time']).div(4.09e6 * 300).reset_index()  # J/m^2 to degC
        df_ohc = df_ohc.assign(time=pd.PeriodIndex.from_fields(year=df_ohc.pop('year'), month=df_ohc.pop('month'), freq='M')).set_index('time')
        if self.assistant.args.use_tpo_for_enso_ohc:
            df_ohc['ENSO'] = df_ohc['PO']
        df_ohc = df_ohc[self.assistant.modes]
        # 对每一列去除逐月均值
        tmp = df_ohc[df_ohc.index <= '2025-12']
        monthly_mean = tmp.groupby(tmp.index.month)[self.assistant.modes].mean()
        monthly_mean.index.name = 'month'
        df_ohc = (df_ohc[self.assistant.modes].set_index(df_ohc.index.month.set_names('month'), append=True) - monthly_mean).reset_index(level='month', drop=True)
        # 对每一列去除二次趋势
        if getattr(self.assistant.args, 'detrend_quadratic', True): 
            for mode in self.assistant.modes:
                year = df_ohc.index.year
                w = (year >= 1979) & (year <= 2022)
                t = np.arange(len(df_ohc))
                coeffs = np.polyfit(t, df_ohc[mode].values.astype(float), 2, w=w)
                _logger.debug(f"Detrending {mode} OHC with coefficients: {coeffs}")
                trend = np.polyval(coeffs, t)
                df_ohc[mode] = df_ohc[mode] - trend
            _logger.debug(f"Finished detrending OHC modes: {self.assistant.modes}")
        else:
            _logger.debug(f"Skipping quadratic detrending for OHC modes: {self.assistant.modes}")

        # CO2
        df_co2 = pd.read_csv('data/lmo_co2.csv')
        df_co2 = df_co2.assign(time=pd.PeriodIndex.from_fields(year=df_co2.pop('year'), month=df_co2.pop('month'), freq='M')).set_index('time')
        df_co2 = df_co2[['deseasonalized']].rename(columns={'deseasonalized': 'value'})
        if normalize_co2:
            _logger.info(
                f"Normalizing CO2 data by min-max scaling "
                f"(min={df_co2.min().value}, max={df_co2.max().value})"
            )
            df_co2 = (df_co2 - df_co2.min()) / (df_co2.max() - df_co2.min())
        # AOT
        df_aot = pd.read_csv('data/noaa_aot.csv')
        df_aot = df_aot.assign(time=pd.PeriodIndex.from_fields(year=df_aot.pop('year'), month=df_aot.pop('month'), freq='M')).set_index('time')
        df_aot = df_aot[['aot']].rename(columns={'aot': 'value'})
        if normalize_aot:
            _logger.info(
                f"Normalizing AOT data by min-max scaling "
                f"(min={df_aot.min().value}, max={df_aot.max().value})"
            )
            df_aot = (df_aot - df_aot.min()) / (df_aot.max() - df_aot.min())
        # TSI
        df_tsi = pd.read_csv('data/noaa_tsi.csv')
        df_tsi = df_tsi.assign(time=pd.PeriodIndex.from_fields(year=df_tsi.pop('year'), month=df_tsi.pop('month'), freq='M')).set_index('time')
        df_tsi = df_tsi[['TSI']].rename(columns={'TSI': 'value'})
        if normalize_tsi:
            _logger.info(
                f"Normalizing TSI data by min-max scaling "
                f"(min={df_tsi.min().value}, max={df_tsi.max().value})"
            )
            df_tsi = (df_tsi - df_tsi.min()) / (df_tsi.max() - df_tsi.min())
        # Global TEMP
        df_temp = pd.read_csv('data/noaa_globaltemp.csv')
        df_temp = df_temp.assign(time=pd.PeriodIndex.from_fields(year=df_temp.pop('year'), month=df_temp.pop('month'), freq='M')).set_index('time')
        df_temp = df_temp[['temp_anomaly']].rename(columns={'temp_anomaly': 'value'})
        if False: # 已经是 Anomalies 了，不适合再归一化
            _logger.info(
                f"Normalizing Global Tempreture data by min-max scaling "
                f"(min={df_temp.min().value}, max={df_temp.max().value})"
            )
            df_temp = (df_temp - df_temp.min()) / (df_temp.max() - df_temp.min())
        # Time
        t0 = pd.PeriodIndex.from_fields(year=[1979], month=[1], freq='M')
        w0 = 2 * np.pi / 12
        df_time = pd.Series(df_sst.index - t0, index=df_sst.index).apply(lambda x: w0 * x.n).to_frame(name="value")
        # One-hot Encoding for Modes
        mode_list = df_sst.columns
        df_modes = []
        for mode in mode_list:
            df = 0*df_sst.astype(int)
            df[mode] = 1
            df.columns = pd.MultiIndex.from_product([[mode], mode_list])
            df_modes.append(df)
        df_modes = pd.concat(df_modes, axis='columns')
        # 合并数据
        df_all = pd.concat([
            df_time.set_axis(pd.MultiIndex.from_product([['t'], df_time.columns]), axis='columns'),
            df_sst.set_axis(pd.MultiIndex.from_product([['x'], df_sst.columns]), axis='columns'),
            df_ohc.set_axis(pd.MultiIndex.from_product([['h'], df_ohc.columns]), axis='columns'),
            df_co2.set_axis(pd.MultiIndex.from_product([['u_co2'], df_co2.columns]), axis='columns'),
            df_aot.set_axis(pd.MultiIndex.from_product([['u_aot'], df_aot.columns]), axis='columns'),
            df_tsi.set_axis(pd.MultiIndex.from_product([['u_tsi'], df_tsi.columns]), axis='columns'),
            df_temp.set_axis(pd.MultiIndex.from_product([['u_temp'], df_temp.columns]), axis='columns'),
            df_modes,
        ], axis=1)
        df_all = df_all.sort_index(axis=0)
        ## Return
        return df_all



@DataLoader.register('miroc_es2l_1pctco2')
class DataLoader_miroc_es2l_1pctco2(DataLoader):
    def load(self, rescale_ohc=True, normalize_co2=True, normalize_aot=True, normalize_tsi=True):
        # SST
        # df_sst = pd.read_csv('data/cmip6_1pctco2_sst.csv')
        # df_sst = df_sst.assign(time=pd.PeriodIndex.from_fields(year=df_sst.pop('year'), month=df_sst.pop('month'), freq='M')).set_index('time')
        df1 = pd.read_csv("data/tos_Omon_MIROC-ES2L_1pctCO2_r1i1p1f2_gr1.csv")
        df1 = df1.assign(time=pd.PeriodIndex.from_fields(year=df1.pop('year'), month=df1.pop('month'), freq='M')).set_index('time')
        df2 = pd.read_csv("data/tos_Omon_MIROC-ES2L_1pctCO2-cdr_r1i1p1f2_gr1.csv")
        df2 = df2.assign(time=pd.PeriodIndex.from_fields(year=df2.pop('year'), month=df2.pop('month'), freq='M')).set_index('time')
        df_sst = pd.concat([df1.loc['1850':'1989'], df2.loc['1990':'2490']], axis=0)
        df_sst = df_sst[self.assistant.modes] # 规范化 modes 顺序, 并移除不需要的 modes
        # OHC (Z20)
        if self.assistant.args.use_z20_as_ohc:
            raise NotImplementedError()
        # df_ohc = pd.read_csv('data/cmip6_1pctco2_ohc.csv')
        # df_ohc = df_ohc.assign(time=pd.PeriodIndex.from_fields(year=df_ohc.pop('year'), month=df_ohc.pop('month'), freq='M')).set_index('time')
        df3 = pd.read_csv("data/thetao_Omon_MIROC-ES2L_1pctCO2_r1i1p1f2_gr1.csv")
        df3 = df3.assign(time=pd.PeriodIndex.from_fields(year=df3.pop('year'), month=df3.pop('month'), freq='M')).set_index('time')
        df4 = pd.read_csv("data/thetao_Omon_MIROC-ES2L_1pctCO2-cdr_r1i1p1f2_gr1.csv")
        df4 = df4.assign(time=pd.PeriodIndex.from_fields(year=df4.pop('year'), month=df4.pop('month'), freq='M')).set_index('time')
        df_ohc = pd.concat([df3.loc['1850':'1989'], df4.loc['1990':'2490']], axis=0)
        if rescale_ohc:
            df_ohc = df_ohc.div(4.09e6 * 300)  # J/m^2 to degC
        if self.assistant.args.use_tpo_for_enso_ohc:
            df_ohc['ENSO'] = df_ohc['PO']
        df_ohc = df_ohc[self.assistant.modes]
        # CO2
        df_co2 = pd.read_csv('data/co2_Omon_MIROC-ES2L_1pctCO2.csv')
        df_co2 = df_co2.assign(time=pd.PeriodIndex.from_fields(year=df_co2.pop('year'), month=df_co2.pop('month'), freq='M')).set_index('time')
        df_co2 = df_co2[['co2']].rename(columns={'co2': 'value'})
        if normalize_co2:
            df_co2_min = 314.44 # 在不同数据集之间保持固定不变
            df_co2_max = 427.45 # 在不同数据集之间保持固定不变
            _logger.info(
                f"Normalizing CO2 data by min-max scaling "
                f"(min={df_co2_min}, max={df_co2_max})"
            )
            df_co2 = (df_co2 - df_co2_min) / (df_co2_max - df_co2_min)
        # Time
        t0 = pd.PeriodIndex.from_fields(year=[1979], month=[1], freq='M')
        w0 = 2 * np.pi / 12
        df_time = pd.Series(df_sst.index - t0, index=df_sst.index).apply(lambda x: w0 * x.n).to_frame(name="value")
        # One-hot Encoding for Modes
        mode_list = df_sst.columns
        df_modes = []
        for mode in mode_list:
            df = 0*df_sst.astype(int)
            df[mode] = 1
            df.columns = pd.MultiIndex.from_product([[mode], mode_list])
            df_modes.append(df)
        df_modes = pd.concat(df_modes, axis='columns')
        # 合并数据
        df_all = pd.concat([
            df_time.set_axis(pd.MultiIndex.from_product([['t'], df_time.columns]), axis='columns'),
            df_sst.set_axis(pd.MultiIndex.from_product([['x'], df_sst.columns]), axis='columns'),
            df_ohc.set_axis(pd.MultiIndex.from_product([['h'], df_ohc.columns]), axis='columns'),
            df_co2.set_axis(pd.MultiIndex.from_product([['u_co2'], df_co2.columns]), axis='columns'),
            df_modes,
        ], axis=1)
        df_all = df_all.sort_index(axis=0)
        # 整理内存
        df_all = df_all.copy()
        ## Return
        return df_all


@DataLoader.register('1pctCO2')
class DataLoader_1pctCO2(DataLoader):
    def load(self, rescale_ohc=True, normalize_co2=True, normalize_aot=True, normalize_tsi=True):
        if not hasattr(self.assistant.args, 'data_source'):
            raise ValueError(
                "No data_source specified in args, set it to csv files like "
                "data/cmip6_1pctco2_1pctCO2-cdr_MIROC-ES2L_modes.csv"
            )
        
        df = pd.read_csv(self.assistant.args.data_source)

        def normalize(df, col):
            df = df.set_index(['year', 'month'])[col]
            # 对每一列去除逐月均值
            df = df - df.groupby('month').mean()
            # 对每一列去除二次趋势
            t = np.arange(len(df))
            coeffs = np.polyfit(t, df.values.astype(float), 2)
            _logger.debug(f"Detrending {col} with coefficients: {coeffs}")
            trend = np.polyval(coeffs, t)
            df = df - trend
            return df.values
        
        if not hasattr(self.assistant.args, 'normalize_sst_ohc') or not self.assistant.args.normalize_sst_ohc:
            pass
        elif self.assistant.args.normalize_sst_ohc == 'overall':
            for col in df.columns:
                if col.startswith(('SST_', 'OHC_')):
                    df[col] = normalize(df, col)
        elif self.assistant.args.normalize_sst_ohc == 'split':
            for col in df.columns:
                if col.startswith(('SST_', 'OHC_')):
                    for phase in ['ramp up', 'ramp down']:
                        mask = df['phase'] == phase
                        df.loc[mask, col] = normalize(df.loc[mask], col)
        else:
            raise ValueError(
                f"Invalid value for normalize_sst_ohc: {self.assistant.args.normalize_sst_ohc}. "
                f"Expected 'overall', 'split', or None."
            )

        # SST
        df_sst = df[[col for col in df.columns if col.startswith(('SST_'))]]
        df_sst.columns = [col.replace('SST_', '') for col in df_sst.columns]
        df_sst.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        df_sst.index.name = 'time'
        df_sst = df_sst[self.assistant.modes] # 规范化 modes 顺序, 并移除不需要的 modes

        # OHC (Z20)
        if self.assistant.args.use_z20_as_ohc:
            raise NotImplementedError()
        df_ohc = df[[col for col in df.columns if col.startswith(('OHC_'))]]
        df_ohc.columns = [col.replace('OHC_', '') for col in df_ohc.columns]
        df_ohc.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        df_ohc.index.name = 'time'
        if rescale_ohc:
            df_ohc = df_ohc.div(4.09e6 * 300)  # J/m^2 to degC
        if self.assistant.args.use_tpo_for_enso_ohc:
            df_ohc['ENSO'] = df_ohc['PO']
        df_ohc = df_ohc[self.assistant.modes]
        
        # CO2
        df_co2 = df[['co2']].rename(columns={'co2': 'value'})
        df_co2.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        if normalize_co2:
            df_co2_min = 314.44 # 在不同数据集之间保持固定不变
            df_co2_max = 427.45 # 在不同数据集之间保持固定不变
            _logger.info(
                f"Normalizing CO2 data by min-max scaling "
                f"(min={df_co2_min}, max={df_co2_max})"
            )
            df_co2 = (df_co2 - df_co2_min) / (df_co2_max - df_co2_min)
        # Time
        t0 = pd.PeriodIndex.from_fields(year=[1979], month=[1], freq='M')
        w0 = 2 * np.pi / 12
        df_time = pd.Series(df_sst.index - t0, index=df_sst.index).apply(lambda x: w0 * x.n).to_frame(name="value")
        # One-hot Encoding for Modes
        mode_list = df_sst.columns
        df_modes = []
        for mode in mode_list:
            df = 0*df_sst.astype(int)
            df[mode] = 1
            df.columns = pd.MultiIndex.from_product([[mode], mode_list])
            df_modes.append(df)
        df_modes = pd.concat(df_modes, axis='columns')
        # 合并数据
        df_all = pd.concat([
            df_time.set_axis(pd.MultiIndex.from_product([['t'], df_time.columns]), axis='columns'),
            df_sst.set_axis(pd.MultiIndex.from_product([['x'], df_sst.columns]), axis='columns'),
            df_ohc.set_axis(pd.MultiIndex.from_product([['h'], df_ohc.columns]), axis='columns'),
            df_co2.set_axis(pd.MultiIndex.from_product([['u_co2'], df_co2.columns]), axis='columns'),
            df_modes,
        ], axis=1)
        df_all = df_all.sort_index(axis=0)
        # 整理内存
        df_all = df_all.copy()
        df_all.index.name = 'time'
        ## Return
        return df_all



@DataLoader.register('ssp585')
class DataLoader_SSP585(DataLoader):
    def load(self, rescale_ohc=True, normalize_co2=True, normalize_aot=True, normalize_tsi=True):
        if not hasattr(self.assistant.args, 'data_source'):
            raise ValueError(
                "No data_source specified in args, set it to csv files like "
                "data/sssp5-8.5/cmip6_ssp585_E3SM-1-1_modes.csv"
            )
        
        data_source1 = Path(self.assistant.args.data_source)
        data_source2 = data_source1.parent.parent / 'ssp5-8.5-historical' / data_source1.name.replace('ssp585', 'historical')
        df1 = pd.read_csv(data_source1)
        df2 = pd.read_csv(data_source2)
        df = pd.concat([df2.assign(phase='historical'), df1.assign(phase='SSP5-8.5')], axis=0, ignore_index=True)

        def normalize(df, col):
            df = df.set_index(['year', 'month'])[col]
            # 对每一列去除逐月均值
            df = df - df.groupby('month').mean()
            # 对每一列去除二次趋势
            t = np.arange(len(df))
            coeffs = np.polyfit(t, df.values.astype(float), 2)
            _logger.debug(f"Detrending {col} with coefficients: {coeffs}")
            trend = np.polyval(coeffs, t)
            df = df - trend
            return df.values
        
        if not hasattr(self.assistant.args, 'normalize_sst_ohc') or not self.assistant.args.normalize_sst_ohc:
            pass
        elif self.assistant.args.normalize_sst_ohc == 'overall':
            for col in df.columns:
                if col.endswith(('_SST', '_OHC')):
                    df[col] = normalize(df, col)
        elif self.assistant.args.normalize_sst_ohc == 'split':
            for col in df.columns:
                if col.startswith(('SST_', 'OHC_')):
                    for phase in ['historical', 'SSP5-8.5']:
                        mask = df['phase'] == phase
                        df.loc[mask, col] = normalize(df.loc[mask], col)
        else:
            raise ValueError(
                f"Invalid value for normalize_sst_ohc: {self.assistant.args.normalize_sst_ohc}. "
                f"Expected 'overall', or None."
            )

        # SST
        df_sst = df[[col for col in df.columns if col.endswith(('_SST'))]]
        df_sst.columns = [col.replace('_SST', '') for col in df_sst.columns]
        df_sst.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        df_sst.index.name = 'time'
        df_sst = df_sst[self.assistant.modes] # 规范化 modes 顺序, 并移除不需要的 modes

        # OHC (Z20)
        if self.assistant.args.use_z20_as_ohc:
            raise NotImplementedError()
        df_ohc = df[[col for col in df.columns if col.endswith(('_OHC'))]]
        df_ohc.columns = [col.replace('_OHC', '') for col in df_ohc.columns]
        df_ohc.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        df_ohc.index.name = 'time'
        if rescale_ohc:
            df_ohc = df_ohc.div(4.09e6 * 300)  # J/m^2 to degC
        if self.assistant.args.use_tpo_for_enso_ohc:
            df_ohc['ENSO'] = df_ohc['PO']
        df_ohc = df_ohc[self.assistant.modes]
        
        # CO2
        df_co2 = df[['CO2']].rename(columns={'CO2': 'value'})
        df_co2.index = pd.PeriodIndex.from_fields(year=df['year'], month=df['month'], freq='M')
        df_co2['value'] = df_co2['value'].where(df_co2['value'].index.month == 1).interpolate(method='linear') # 只保留每年 1 月, 其他月份设为 NaN, 然后对其他月份做线性插值
        if normalize_co2:
            df_co2_min = 314.44 # 在不同数据集之间保持固定不变
            df_co2_max = 427.45 # 在不同数据集之间保持固定不变
            _logger.info(
                f"Normalizing CO2 data by min-max scaling "
                f"(min={df_co2_min}, max={df_co2_max})"
            )
            df_co2 = (df_co2 - df_co2_min) / (df_co2_max - df_co2_min)
        
        # Time
        t0 = pd.PeriodIndex.from_fields(year=[1979], month=[1], freq='M')
        w0 = 2 * np.pi / 12
        df_time = pd.Series(df_sst.index - t0, index=df_sst.index).apply(lambda x: w0 * x.n).to_frame(name="value")
        # One-hot Encoding for Modes
        mode_list = df_sst.columns
        df_modes = []
        for mode in mode_list:
            df = 0*df_sst.astype(int)
            df[mode] = 1
            df.columns = pd.MultiIndex.from_product([[mode], mode_list])
            df_modes.append(df)
        df_modes = pd.concat(df_modes, axis='columns')
        
        # 合并数据
        df_all = pd.concat([
            df_time.set_axis(pd.MultiIndex.from_product([['t'], df_time.columns]), axis='columns'),
            df_sst.set_axis(pd.MultiIndex.from_product([['x'], df_sst.columns]), axis='columns'),
            df_ohc.set_axis(pd.MultiIndex.from_product([['h'], df_ohc.columns]), axis='columns'),
            df_co2.set_axis(pd.MultiIndex.from_product([['u_co2'], df_co2.columns]), axis='columns'),
            df_modes,
        ], axis=1)
        df_all = df_all.sort_index(axis=0)
        # 整理内存
        df_all = df_all.copy()
        df_all.index.name = 'time'
        ## Return
        return df_all
