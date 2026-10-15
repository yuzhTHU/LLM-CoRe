# LLM-CoRe

本仓库包含 LLM-CoRe 手稿的方程搜索代码以及复现实验所需的数据。

## 环境配置

已测试的环境为 Python 3.12。可在仓库根目录创建虚拟环境并安装锁定版本的依赖：
```bash
pip install -r requirements.txt
```

## 方程搜索

复制环境变量模板并填写 API key：

```bash
cp .env.sample .env
```

随后运行：

```bash
./search.sh
```

搜索结果写入 `logs/search/<exp_name>/`。该目录包括 `args.json`、`info.log`、逐代记录、checkpoint 和最终的 `best_model.py`。`logs/` 属于运行产物，不纳入版本控制。

## 正文图片

`scripts/` 中的 notebook 生成正文图 2--4 的数据 panel。请从仓库根目录运行。输出保存在 `logs/scripts/<notebook-name>/`，文件名直接对应正文 panel：

- `figure2_deterministic.ipynb`：`Fig2a-c-e`、`Fig2b-d-f-1` 和 `Fig2b-d-f-2`；
- `figure2_stochastic.ipynb`：`Fig2g-h`；
- `figure3_co2_attribution.ipynb`：`Fig3a`、`Fig3b`；
- `figure3_interactions.ipynb`：`Fig3c`、`Fig3d`、`Fig3e`；
- `figure3_event_leads.ipynb`：`Fig3f`；
- `figure3_event_diagnostics.ipynb`：`Fig3g`、`Fig3h`；
- `figure4_counterfactual.ipynb`：`Fig4a`、`Fig4b`、`Fig4c`、`Fig4d`；
- `figure4_phase_space.ipynb`：`Fig4e`、`Fig4f`、`Fig4g`、`Fig4h`。

## 数据目录

- `data/*.csv`：观测和外强迫时间序列；
- `data/baseline/<baseline-name>/*.csv`：ORCA-DL、UniCM 和 Physics-DESN 的已导出预测结果；
- `data/cache/default_args.json`：绘图时加载模型所需的默认参数；
- `data/cache/figure2_deterministic.pkl.xz`：Figure 2 确定性预测 panel 使用的预测明细；
- `data/cache/counterfactual.csv`：Figure 4 反事实实验使用的统计量及其置信区间。
