# GeoReg3DAD

共享的几何异常检测方法，独立的 Real3D-AD / Anomaly-ShapeNet 配置，统一的
Windows/Linux 命令行入口。新实验使用 `run.py`；根目录的一次性历史脚本已清理。

## Real3D 官方输入协议（2026-09-29）

Real3D 的 `inspect/run/smoke` 现在默认使用 `--input-protocol real3dad-official`：
正常扫描读取 PCD，异常扫描读取 GT TXT 前三列，所有模板和扫描分别减去质心；
第四列标签只在预测后读取。五个历史 PCD–TXT 点数不一致样本也完整参与点级评价，
全量覆盖为 1,206 个物体 / 1,206 个点级扫描。此项对齐数据加载行为，不代表复现官方模型。
ShapeNet 保持原协议。重跑旧 PCD 协议必须显式指定 `--input-protocol legacy-pcd`；
旧输出不能以新协议续跑。下文 1,201 个点级扫描及历史 GT 排除说明适用于旧协议。

`tools/run_official_real3dad.py` 从已完成的历史四组实验冻结参数，重新进行模板与扫描配准，
比较 k128、k3、直接体素和固定 TFCE，统一 Top 1%。它保存全部 1,206 扫描的点分数，
并提供 `verify` 命令用 sklearn、原始标签和排序 Top 1% 独立复算。历史成绩差异同时包含
输入点集、减质心和新配准的影响，不应全部归因于补回五个扫描。

## 目录

```text
configs/
  config_real3dad.json      Real3D 参数及 baseline / balanced 预设
  config_shapenet.json      ShapeNet 参数及 k128_ablation 预设
georeg3dad/
  config.py                参数类型、验证、预设与覆盖
  geometry.py              特征、配准、模板、统一推理接口
  scoring.py               候选残差、权重、插值、物体聚合
  metrics.py               分块累计阈值的精确点级 AUROC/AP
  datasets.py              数据布局、标签对齐、文件名适配
  protocols/               类别覆盖与已审计 GT 排除记录
  runner.py                类别并行、内存调度、续跑、指标汇总
  runtime.py               线程限制、原子写入、跨平台进程与内存
  cli.py                   统一入口
run.py
tests/                     单元测试和子进程集成测试
tools/verify_refactor.py    真实数据的新旧实现数值对照
```

核心代码不依赖旧实验脚本，没有导入时替换函数或硬编码磁盘路径。
参数文件里不写个人数据路径；输入、输出由命令行指定。所有调参都先经配置验证。

## 环境

使用 Python 3.10 和 `requirements-portable.txt` 的固定依赖。历史 requirements
保留原样，避免改动旧归档。已有 `.venv-georeg` 可直接使用，无需重新安装。

新环境在 Linux 或 Windows 上均可执行：

```text
python -m venv .venv
python -m pip install -r requirements-portable.txt
```

第二条命令应使用新虚拟环境中的 Python：Linux 为 `.venv/bin/python`，Windows 为
`.venv\Scripts\python.exe`。也可以先激活环境。Linux 运行 Open3D 可能需要系统库
`libgl1`、`libgomp1`；推理不需要显示器或 CUDA。

以下示例假设当前目录是 `04_code`，`python` 指向上述环境。可选
`python -m pip install -e .`，安装后也可使用 `georeg3dad` 命令。

## 数据集配置

|配置|默认含义|
|---|---|
|`config_real3dad.json`|voxel=.2、k=128、p=0、penalty=1、radius=6、candidate_k=8、plane=1、normal=.5；最近验证集选中的配置，全量 AUC/AP 存在取舍|
|Real3D `--preset balanced`|上述配置的 plane=.5 对照|
|Real3D `--preset baseline`|原始 k=3、p=1、penalty=16、radius=8、plane/normal=.5|
|`config_shapenet.json`|voxel=.05、k=3、p=1、penalty=16、radius=8、plane/normal=.5；独立 ShapeNet 基线|
|ShapeNet `--preset k128_ablation`|仅复现 k=128、p=0 插值消融，不将它标为 ShapeNet 最优|

改变单个参数无需复制代码，例如 `--set matching.normal_weight=0.25`。
可多次使用 `--set`；只接受配置中支持的字段与合法数值。完整解析结果写入每次
实验的 `config.json`，包括从默认值补齐的参数。

数据根目录结构：

```text
Real3D-root/<category>/{train,test,gt}/
ShapeNet-root/{pcd,new_pcd}/<category>/{train,test,GT}/
```

ShapeNet 正常输入文件保留 `_positive` 原名；适配器只在样本标识和 CRC32 随机种子
中转换为 `_good`，不创建硬链接、不修改数据。随机种子始终根据 POSIX 格式的
`test/<canonical filename>` 计算，与盘符和目录分隔符无关。

## 运行

Windows（输出保存在 D 盘）：

```powershell
python run.py inspect --config configs/config_real3dad.json --data-root "D:/GeoReg3DAD-Data/datasets/Real3D-AD-PCD/Real3D-AD-PCD"
python run.py run --config configs/config_real3dad.json --data-root "D:/GeoReg3DAD-Data/datasets/Real3D-AD-PCD/Real3D-AD-PCD" --run-root "D:/GeoReg3DAD-Data/runs/my_real3dad_run" --workers 4 --threads 2
```

Linux（路径替换为服务器上的实际挂载目录）：

```bash
python run.py inspect --config configs/config_shapenet.json --data-root /data/Anomaly-ShapeNet-v2/dataset
python run.py run --config configs/config_shapenet.json --data-root /data/Anomaly-ShapeNet-v2/dataset --run-root /data/runs/my_shapenet_run --workers 4 --threads 2
```

`--data-root` 也可由 `GEOREG_DATA_ROOT` 环境变量提供。`--run-root` 必须显式指定，
防止将大量预测写入代码目录或系统盘。

ShapeNet 默认只评估官方 `pcd` 40 类；`--scope all` 包含 52 类，
`--scope new_pcd` 只评估扩展 12 类。类别使用完整标识，例如 `--categories pcd__bag0 pcd__bowl0`。
Real3D 使用 `--categories candybar gemstone`。`inspect` 校验类别、扫描及标签文件覆盖。

小规模运行使用 `smoke` 代替 `run`，每类取一个正常、一个点级标签有效的异常扫描，
默认选择两类，也可显式给出类别。它经过完整推理/标签/保存/评估流程，但不代表全量指标。

## 输出、并行与续跑

- `status.json`：当前阶段、正在运行及已完成类别。
- `config.json` / `dataset.json` / `protocol.json`：解析后的参数、输入哈希、环境及源码哈希。
- `results/<category>/scores/*.npz`：点分数与对齐标签；已知无效 GT 的标签数组为空。
- `results/<category>/cases.json`：变换、种子、配准统计、物体分数；模板变换也单独保存。
- `summary.json` / `REPORT.md`：完成后的类别宏平均；ShapeNet 包含单独的官方类别汇总。
- `verification.json`：覆盖、数组、输入/输出及源码一致性检查结果。

最多 `workers` 类别并行，每类 `threads` 个 OpenMP 线程，BLAS 固定一个线程。
类别调度同时检查估计内存和操作系统可用内存；不足时降低实际并发。
Windows 和 Linux 都使用独立子进程，失败会停止新增任务并记录日志。
文件写入采用原子替换，兼容 Windows 的临时文件占用；中断时清理本次创建的工作进程树。

输出目录需至少 5 GiB 可用空间；全量实验建议预留 15 GiB 以上。临时指标 memmap 在类别
成功后清理，历史缓存不删除。再次运行相同命令并增加 `--resume`，可复用哈希和覆盖
验证通过的完整类别；未完成类别重算。参数、输入或源码变化会拒绝续跑。
同时运行同一个输出目录会被 `.run.lock` 拒绝。强制关机留下锁时，确认旧进程已退出后
再移除该次运行的锁文件；不要复制旧配置到新输出目录冒充续跑。

GT 排除协议保留 Real3D 5 个、ShapeNet 5 个已知无效标注扫描，仍计入物体指标。
这些标注文件必须匹配归档 SHA-256，其他任何标签错误直接报错。
每类汇总有效扫描的全部点计算 P-AUROC/P-AP，再对类别等权平均；物体分数为最高
ceil(object_top_fraction×点数) 的点分数均值。旧默认比例为 1%，最新选定配置详见
[统一调参报告](../TUNING_RESULTS.md)。阈值和排除规则不参与调参。

## 验证与边界

```text
python -m unittest discover -s tests -v
```

CI 定义在 `.github/workflows/test.yml`，矩阵包含 Windows 与 Ubuntu、Python 3.10。
最初重构在 Windows 实测，本机 WSL 返回 `REGDB_E_CLASSNOTREG`。后续已经在 Linux
服务器执行双数据集搜索，记录见[统一调参报告](../TUNING_RESULTS.md)；这不等于未执行的
GitHub CI 已通过。跨平台路径、进程及内存处理已经实现。

真实数据对照命令见 `REFACTOR_VALIDATION.md`。同一模板和变换下的新旧评分可以严格
比较；新启动的 FGR/ICP 在不同系统或 Open3D 构建下仍可能产生不同变换，因此
不能承诺重新跑全量时逐位复现旧分数。源码重构不替代独立测试集或多种子验证。

`original/`、`recovered_shapenet/` 保留用于 `tools/verify_refactor.py` 的新旧数值对照。
旧 `analyze_*.py`、`tune_real3dad*.py`、迁移和一次性复现脚本已从代码目录删除，
历史源码集中保存在 [legacy_process_scripts.zip](D:/GeoReg3DAD-Data/reports/tuning_consolidation_20260928/legacy_process_scripts.zip)。
当前核心、测试及可复用搜索工具不依赖这些脚本。后续实验修改配置或使用 `--set`，
不再新增整份调参脚本副本。

## 最新全量与 ShapeNet 搜索（2026-09-24）

队列工具为 `tools/run_latest_campaign.py`，搜索工具为 `tools/tune_shapenet.py`。
搜索调用同一套核心方法，冻结已验证全量基线的模板和扫描配准，先在固定分组上
逐阶段选参，最后全量复评。具体范围、顺序和选择规则见
[统一调参报告](../TUNING_RESULTS.md)。旧缓存和已被后续方案替代的预测已按报告清单清理；
重放旧搜索前需重建对应缓存，不能只凭旧状态文件直接续跑。

点级指标改为一次全局排序、分块累计精确阈值计数，降低全量计算的峰值内存。
数值与 sklearn 对照测试覆盖并列分数和跨块边界；算法评分公式不变。
