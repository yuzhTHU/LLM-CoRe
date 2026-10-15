# LLM-CoRe

本仓库包含 LLM-CoRe 手稿的方程搜索代码、正文图 2--4 的绘图 notebook，以及复现实验所需的精简数据。

## 环境配置

已测试的环境为 Python 3.12。可在仓库根目录创建虚拟环境并安装锁定版本的依赖：

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

`nd2py` 由 `requirements.txt` 直接从锁定的 Git commit 安装，不需要以 editable 模式安装本地源码。

## 方程搜索

复制环境变量模板并填写 API key：

```bash
cp .env.sample .env
```

随后运行：

```bash
./search.sh
```

`search.sh` 固化了手稿中报告的主要搜索设置，包括 `only_co2` prompt、`gpt-4o-mini`、200 个个体、300 次迭代、每次 prompt 生成 8 个程序、五折交叉验证和目标函数权重。可以在命令末尾追加参数，用于指定实验名、随机种子或执行诊断。例如：

```bash
./search.sh --exp_name my-run --random_state 2026101401
```

搜索结果写入 `logs/search/<exp_name>/`。该目录包括 `args.json`、`info.log`、逐代记录、checkpoint 和最终的 `best_model.py`。`logs/` 属于运行产物，不纳入版本控制。

## 正文图片

`scripts/` 中的 notebook 生成正文图 2--4 的数据 panel。请从仓库根目录运行，并选择 `Python (LLM-CoRe)` kernel。也可以非交互执行，例如：

```bash
venv/bin/python -m jupyter nbconvert --to notebook --execute \
  scripts/figure2_deterministic.ipynb \
  --output figure2_deterministic.ipynb --output-dir scripts \
  --ExecutePreprocessor.timeout=1800
```

输出保存在 `logs/scripts/<notebook-name>/`，文件名直接对应正文 panel：

- `figure2_deterministic.ipynb`：`Fig2a-c-e` 和 `Fig2b-d-f`；
- `figure2_stochastic.ipynb`：`Fig2g-h`；
- `figure3_co2_attribution.ipynb`：`Fig3a`、`Fig3b`；
- `figure3_interactions.ipynb`：`Fig3c`、`Fig3d`、`Fig3e`；
- `figure3_event_leads.ipynb`：`Fig3f`；
- `figure3_event_diagnostics.ipynb`：`Fig3g`、`Fig3h`；
- `figure4_counterfactual.ipynb`：`Fig4a-c`、`Fig4d`；
- `figure4_phase_space.ipynb`：`Fig4e`、`Fig4f`、`Fig4g`、`Fig4h`。

每张图同时输出 SVG、PNG 和 PDF。`Fig2a-c-e` 表示同一文件包含不连续的 a、c、e 三个 panel；`Fig4a-c` 表示同一文件包含连续的 a--c 三个 panel。Figure 1 为演示文稿软件组装的示意图，因此没有对应的绘图 notebook。

`figure3_interactions.ipynb` 中的逐连接消融计算最耗时，在测试机器上约需 20 分钟。其预测结果会缓存在该 notebook 的输出目录，后续运行会复用缓存。

## 数据目录

- `data/*.csv`：观测和外强迫时间序列；
- `data/baseline/<baseline-name>/*.csv`：ORCA-DL、UniCM 和 Physics-DESN 的已导出预测结果；
- `data/counterfactual/*.csv`：图 4 反事实实验使用的精简统计量；
- `data/cache`：...

仓库不包含旧的多 GB 中间模拟缓存。保留的 notebook 包含重新生成必要结果的实验代码。
