import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nd2py.utils.plot import get_fig, merge_axes

def split_axes(ax, nrows=20, ncols=20, hs_cm=0.0, vs_cm=0.0):
    fig = ax.figure
    pos = ax.get_position()  # Bbox in figure coordinates
    ax.remove()

    fig_w_cm, fig_h_cm = fig.get_size_inches() * 2.54  # inch -> cm
    total_w = pos.width * fig_w_cm
    total_h = pos.height * fig_h_cm

    # 子图尺寸
    cell_w = (total_w - (ncols-1)*hs_cm) / ncols
    cell_h = (total_h - (nrows-1)*vs_cm) / nrows

    axes = []
    for i in range(nrows):
        row_axes = []
        for j in range(ncols):
            left = pos.x0 + (cell_w + hs_cm)/fig_w_cm * j
            bottom = pos.y0 + (cell_h + vs_cm)/fig_h_cm * (nrows - i - 1)
            width = cell_w / fig_w_cm
            height = cell_h / fig_h_cm
            sub_ax = fig.add_axes([left, bottom, width, height])
            row_axes.append(sub_ax)
        axes.append(row_axes)
    return np.array(axes)


def plot_heatmap(PREDICTION, ax=None, cax=None, discretized=False, repeat_month=1, mode='ENSO', threshold=0.5):
    if discretized:
        interpolation='nearest'
        cmap = mcolors.ListedColormap(['#546de5', '#dfe6e9', '#c44569'])
        norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    else:
        interpolation='bilinear'
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=-3, vmax=3)
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_rdbu', ['#546de5', '#dfe6e9', '#c44569'])
        # norm = plt.Normalize(vmin=-2, vmax=2)
        
    # stop = PREDICTION.reset_index().apply(lambda row: row['time'] + pd.DateOffset(months=row['lead']), axis=1)
    # table = PREDICTION.assign(stop=stop).drop(columns=['time']).pivot_table(index='lead', columns='stop', values=mode)
    # table = table.sort_index(axis='rows').sort_index(axis='columns')
    # table.columns = table.columns.strftime('%Y %b')
    table = (
        PREDICTION
        .reset_index()
        .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
        .set_index(['time', 'lead', 'stop'])
        .pivot_table(index='lead', columns='stop', values=[('x', mode)])[('x', mode)]
    )
    discretized_table = pd.DataFrame(
        np.where(table.values > threshold, 1, np.where(table.values < -threshold, -1, np.where(np.isnan(table.values), np.nan, 0))),
        index=table.index, columns=table.columns,
    )
    if discretized:
        table = discretized_table

    # sns.heatmap(table, cmap=cmap, center=0, norm=norm, ax=ax, cbar=False)
    ax.imshow(table, cmap=cmap, norm=norm, interpolation=interpolation, aspect='auto', rasterized=True)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticklabels(table.index)
    ax.set_xticklabels(table.columns)
    ax.set_xlabel(None)
    ax.set_ylabel('Lead Month')
    ax.tick_params(axis='x', rotation=90)
    if repeat_month is not None and repeat_month > 0:
        same = discretized_table.iloc[1:] == discretized_table.iloc[0].values
        for i in range(1, repeat_month):
            same &= same.shift(-i)
        same &= discretized_table.iloc[0].isin([1, -1])
        # ax.imshow(same, cmap='Grays', alpha=0.2, interpolation='none', aspect='auto', zorder=10000, rasterized=True)
        # --- 替换原 ax.imshow(same, ...) 的部分 ---
        # 1. 准备坐标网格
        # 注意：imshow 的中心点在整数索引处
        rows, cols = same.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        # 2. 绘制斜线阴影 (Hatching)
        # levels=[0.5, 1.5] 表示只填充值为 1 (True) 的区域
        # colors='none' 确保不覆盖底部的颜色，只显示斜线
        # hatches=['///'] 定义斜线密度，'///' 比 '/' 更密
        cf = ax.contourf(X, Y, same, levels=[0.5, 1.5], hatches=['////'], colors='none', zorder=10000)
        cf.set_rasterized(True)

        # 3. 绘制平滑的轮廓线 (Outline)
        # 如果需要极度平滑，可以对 same 进行简单的插值，这里直接用 contour 默认会比 imshow 平滑
        # 如果追求更丝滑的线条，可以使用 scipy.ndimage.zoom 对 same 放大后再画，
        # 但通常直接使用 contour 并设置适当的 linewidth 即可。
        ax.contour(X, Y, same, levels=[0.5], colors='black', linewidths=0.8, alpha=0.8, zorder=10001)
        # ---------------------------------------

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    if discretized:
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['La Nina', 'Neutral', 'El Nino'])
    else:
        cbar.set_label('SST Anomaly (°C)', rotation=270, labelpad=15)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
