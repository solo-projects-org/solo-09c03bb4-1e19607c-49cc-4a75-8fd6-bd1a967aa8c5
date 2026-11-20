# 公交出行比例两轮重抽分析（FastAPI + DuckDB）

对同一批抽样片区的两轮住户调查，按区计算"乘公交比例"的两轮差值，并用重抽权重
（replicate weights）按编号配对生成差值分布，报告其线性插值 2.5% / 97.5% 分位数。

系统同时提供两件事：

1. **配对重抽权重生成**：上传两轮原格式样本 CSV 与一份对照 CSV，填写重抽次数和
   种子，服务端按区有放回重抽片区、两轮共用抽中次数，产出两个可直接导入的
   Parquet 并打成一个 ZIP；
2. **重抽分析**：逐轮上传样本 CSV 与重抽权重 Parquet，做配对比例差与区间分析。

## 1. 数据口径

每轮上传一对文件：

| 文件 | 格式 | 列 |
|---|---|---|
| 样本 | CSV（含表头） | `sample_id, district, bus, main_weight` |
| 重抽权重 | Parquet | `sample_id, rep_id, weight` |

* `bus` 只能是 0/1；两类权重必须为有限非负数；`sample_id`、`district`、`rep_id` 非空。
* 重抽权重只能引用**本轮**样本 CSV 中的 `sample_id`（外键校验）。
* 每个样本在每次重抽中**恰有一条**权重（缺漏、重复均拒绝）；每次重抽的权重总和
  必须为正（全零重抽无法形成比例，拒绝）。
* 两轮的同编号 `rep_id` 表示同一次重抽；两轮编号集合不同时无法配对。

计算（选区 D，轮次 t）：

* 主权重比例 `p_t = Σ(main_weight × bus) / Σ(main_weight)`（仅 D 内样本）
* 比例差点估计 `p_2 − p_1`
* 第 k 次重抽比例 `p_t,k = Σ(weight_k × bus) / Σ(weight_k)`，配对差值
  `d_k = p_2,k − p_1,k`
* 区间端点：`d_k` 的 type-7 线性插值分位数（与 DuckDB `quantile_cont`、numpy 默认
  分位数同口径），取 0.025 与 0.975。

区间留空的三种情况（均返回中文原因）：

1. 任一轮主权重分母为零 → 点估计与区间均留空；
2. 两轮重抽编号集合不同 → 点估计保留，区间留空并列出各自独有的编号；
3. 某次重抽在该区的权重分母为零 → 点估计保留，区间留空并列出问题编号。

### 1.1 配对重抽权重生成（`POST /api/weight-jobs`）

上传三个**原格式 CSV（含表头）**并填写两个整数参数：

| 输入 | 列 |
|---|---|
| 第一轮样本 CSV | `sample_id, district, bus, main_weight` |
| 第二轮样本 CSV | `sample_id, district, bus, main_weight` |
| 对照 CSV | `round, sample_id, area_id`（轮次只取 1/2） |
| 表单 | `n_replicates`（1–999999 的重抽次数）、`seed`（整数随机种子） |

校验规则（任一不过即 422 并返回中文原因，**不触碰已发布的调查数据库**）：

* 样本 CSV 的列结构、`bus ∈ {0,1}`、主权重有限非负、样本号轮内唯一；
* 每个样本在对照 CSV 中恰归一个片区（同轮重复对照拒绝），对照行必须指向本轮
  存在的样本号，且每轮每个样本都必须有片区归属；
* 同一片区在同一轮只能属于一个区；两轮的**区集合一致**，且每个区内两轮的
  **片区集合完全相同**；
* 每轮各片区的主权重合计必须为正。

抽样与权重口径：

* 每次重抽，各区**各自**有放回地抽取与原片区数相同数量的片区；两轮共用同一套
  片区抽中次数（配对 bootstrap），因此同编号重抽在两轮间严格配对；
* 每户权重 = 主权重 × 所在片区本次抽中次数；未抽中片区也显式写出**零权重行**，
  每个样本在每次重抽中恰有一行；
* 单次重抽的抽中次数总和恒等于该区片区数，且每轮每区至少有一个片区被抽中，
  所以每次重抽的权重总和恒为正，产出文件可直接通过分析侧的分母校验。

确定性（编号一致性）：

* 随机流按 `(seed, 区, 重抽序号)` 独立派生（blake2b 取种 + Python MT19937，
  跨 Python 3 版本稳定），片区按字典序参与抽取，**输入行重排不改变任何权重**；
* 同一种子增加重抽次数时，派生键与重抽序号不变，**已有编号的权重逐位一致**，
  新编号只追加在后面。重抽编号形如 `rep000001`（六位零填充）。

产物：**两个 Parquet 都写完并通过行数/覆盖/非负校验后**才组装一个 ZIP
（`ZIP_STORED`，Parquet 本身已压缩），结构为

```
round1/samples.csv          # 上传原文件的字节副本
round1/replicates.parquet   # sample_id, rep_id, weight
round2/samples.csv
round2/replicates.parquet
manifest.json               # 种子、次数、各区片区清单与行数
```

解压后两轮文件可分别从 `POST /api/rounds/{1|2}` 原入口导入。任务文件落在数据卷
`jobs/<job_id>/` 下，仅保留最新 10 个成功任务；失败任务立即清理。下载用
`GET /api/weight-jobs/{job_id}/download`，任务号为服务端生成的十六进制令牌，
路径不接受其他字符。

## 2. 运行

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

Compose 约束：`mem_limit: 384m`、`cpus: 2`；数据放在命名卷 `duckdb_data`
（容器内 `/data`），DuckDB 数据文件、WAL 与临时落盘目录都在该卷上。
DuckDB 自身 `memory_limit=180MB`、`threads=2`，超限走磁盘临时区，不靠堆内存硬扛。

本地无 Docker 时（开发）：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
DATA_DIR=./data uvicorn app.main:app --port 8000
```

## 3. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页界面（权重生成、上传、状态、选区查询） |
| POST | `/api/weight-jobs` | multipart：两轮样本 CSV、对照 CSV、`n_replicates`、`seed`；成功返回任务号、行数与下载地址，失败 422 |
| GET | `/api/weight-jobs/{job_id}/download` | 下载配对权重 ZIP（任务号非法 422，不存在/已清理 404） |
| POST | `/api/rounds/{1\|2}` | multipart：`sample_csv`、`replicate_parquet`；成功返回代际号与行数，失败 422 |
| GET | `/api/status` | 两轮当前代际、样本/重抽/权重行数、区列表 |
| GET | `/api/districts` | 区列表 |
| GET | `/api/result?district=...` | 比例差与区间（留空时附原因） |

## 4. 实现要点

* **代际（generation）模型**：每次成功上传在同一 DuckDB 文件内新建
  `s_gen_<g>` / `r_gen_<g>` 两张表并在事务中改名发布（`meta_current` 指向当前代际）。
  校验全部在暂存表 + 单事务内完成，任何一步失败即回滚——**换表失败时旧代际原样可查**。
* **权重生成与分析库隔离**：`app/weights.py` 在数据卷 `jobs/<job_id>/` 下使用
  独立的临时 DuckDB（scratch 库 + 临时落盘目录），全程不打开分析库；生成失败只
  删除自己的任务目录，现行调查数据照常可查。重抽在 Python 侧按区流式抽号并批量
  写入 `hits` 表，Parquet 由 DuckDB 分区 join 后直接 `COPY` 落盘，权重行不驻留
  Python 内存。
* **生成任务串行化**：进程内 `_generation_lock` 保证同一时刻只有一个 scratch 库
  在跑，避免两个大任务叠加冲破 384 MiB；任务结束后只保留最新 10 个成功目录。
* **查询快照**：查询开始时通过 `meta_refcount` 钉住两轮当前代际；换表只改指针，
  被钉住的旧代际延迟到查询结束、引用计数归零后才删除。因此进行中的查询始终使用
  起始时刻的两套输入，不受并发换表影响。
* **单进程单 uvicorn worker**：上传由写锁串行化；查询走独立短连接。DuckDB 同一
  进程不允许只读/读写连接混用，读连接使用同配置、由应用层保证不写。
* **流式上传**：multipart 以 1 MiB 分块直接落临时目录，避免整文件驻留内存。
* 所有用户输入只作为**数据值**参数化或写入临时文件路径，表名/列名均为服务端
  固定生成；下载任务号限定为十六进制精确匹配，不存在 SQL 标识符或路径注入面。

## 5. 测试与实测

```bash
pytest -q                      # 47 个单元/接口测试
python scripts/make_demo.py    # 生成手工可算的小样本（含错配套件与权重生成套件）
python scripts/scale_test.py --rows 5000000
python scripts/weight_scale_test.py --rows 5000000
```

单元测试覆盖：手算基准的比例与分位数、缺轮、编号错配、主权重零分母、重抽零分母、
外键引用、缺漏/重复权重、非法 bus/权重、重复样本号、列结构不符、换表失败保旧、
换表成功发新代际、快照钉住与延迟删除，以及 Python 分位数与 `quantile_cont` 的
多规模交叉一致性。

权重生成测试（`tests/test_weights.py`，26 个）额外覆盖：ZIP 结构与原 CSV 字节
保留、权重=主权重×抽中次数的逐条核对、零权重显式落盘、两轮抽中次数配对、每次
重抽抽中数之和等于片区数、同种子同参数字节级一致、输入行重排结果不变、同种子
增加次数时旧编号不变、ZIP 解压后可走原 `/api/rounds` 入口导入、对照 CSV 的各类
违规（轮次值、重复归属、外键、缺归属、片区集合不一致、区集合不一致、片区跨区、
片区主权重合计非正）、失败不影响已发布数据、下载任务号校验与历史任务清理，
并钉死了随机流的具体取值序列。

### 500 万行实测（每轮 10,000 样本 × 500 次重抽 = 5,000,000 权重行，两轮共 1,000 万行）

分析侧多轮实测区间（`scripts/scale_test.py --rows 5000000`）：

| 指标 | 实测 |
|---|---|
| 单轮上传（含全部校验、建表发布） | 约 2.6–4.9 秒 |
| 分区查询（含配对、分位数） | 0.1–0.4 秒 |
| 服务进程峰值 RSS | 161–237 MiB（上限 384 MiB） |
| DuckDB 数据文件（两轮合计） | 约 77 MiB |
| 分位数与独立 DuckDB 重算 | 逐位一致（1e-9 容差内） |
| 换表失败后旧数据 | 代际、行数、结果均不变 |
| 换表成功 | 新代际发布、旧代际释放 |

生成侧实测（`scripts/weight_scale_test.py --rows 5000000`，200 个片区、8 个区）：

| 指标 | 实测 |
|---|---|
| 一个完整任务（校验 + 抽样 + 两个 Parquet + ZIP） | 约 8.3 秒（HTTP 实测 8.3 s） |
| 每个 Parquet | 500 万行，约 5.5 MiB |
| ZIP 体积 | 约 11.4 MiB（含两份原 CSV 与 manifest） |
| 服务进程峰值 RSS | 264–267 MiB（上限 384 MiB，DuckDB 限 180 MB） |
| 全部权重与独立重放的抽中次数 | 逐条精确相等（两轮共 1,000 万行） |
| 输入行重排 | Parquet 字节级一致（sha256 相同） |
| 同种子 500 → 510 次 | 前 500 个编号逐位不变，仅追加新编号 |
| 非法任务 | 422 + 中文原因，已发布调查数据不变、服务健康 |

> 实现注记：DuckDB 2.0.1 的 `executemany` 逐行 INSERT 在该负载下异常缓慢
> （10 万行约 178 秒）。抽号结果改为流式写入 scratch 目录下的 `hits.csv`，
> 再用一次 `read_csv_auto` 批量装载（同口径 0.3 秒以内），输出 Parquet 的
> sha256 与旧实现完全一致，且只有当前区的抽号状态驻留 Python 内存。

> 内存说明：本沙箱 cgroup 文件系统只读，无法用 `memory.max` 实测容器级 OOM 行为，
> 脚本以进程峰值 RSS 佐证；`scale_test.py` 保留 cgroup v2 分支，在有权限的环境
> （或直接 `docker compose up`）会按 384 MiB 限制执行并读取 `memory.peak`。
> 另经验证，同进程内多个连接并发重查询时 DuckDB 的 `memory_limit` 由共享缓冲管理器
> 统一约束（两连接并发的 RSS 增量约 31 MiB，而非两个 120 MB 上限相加），并发查询
> 不会使内存预算翻倍。

## 6. 遗留事项 / 已知边界

* 环境内无 Docker 守护进程，Compose 文件仅做了 YAML 与配置校验，未实际构建镜像；
  384 MiB 约束以同参数的进程实测佐证（生成侧峰值 264–267 MiB、分析侧 161–237 MiB），
  建议在有 Docker 的环境再跑一次 `docker compose up` 后接 `scale_test.py` 与
  `weight_scale_test.py`。
* 单 worker 设计：上传与权重生成各自串行（生成另有进程内锁，同一时刻只跑一个
  scratch 库），查询并发受 DuckDB 单实例能力限制；面向多用户高并发需另做队列或
  多实例分片。
* 权重生成任务只保留最新 10 个成功目录（`jobs/`，可在 `weights.JOBS_TO_KEEP`
  调整），更早的任务下载地址会返回 404；ZIP 已下载到本地的不受影响。
* 零/负权重按"非负"校验，未做权重设计层面的口径审计（如权数和与样本量关系）。
* 上传文件大小没有应用层显式上限（依赖反代/容器磁盘），超大文件会先完整落盘再校验。
* 分位数基于重抽差值的经验分布，未做正态近似或有限总体校正。
* 随机流依赖 Python `random.Random`（MT19937）的播种与抽取语义；跨 Python 大版本
  迁移时 `tests/test_weights.py::test_rng_helpers_are_stable` 会第一时间暴露变化。
