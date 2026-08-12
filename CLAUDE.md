# CLAUDE.md

本文件为 Claude Code 在此仓库工作时提供指引。

## 项目概述

`handwritten-v2` 是**手写的 Intel AMX int8 GEMM 算子实现模板 + 性能测试框架**。
目标不是产出一个通用 BLAS 库，而是作为**研究/基准测试台**：分析 AMX GEMM 的数据流瓶颈，
逼近 AMX 计算部件的理论峰值利用率，并对比不同 blocking / 预取 / 数据重排策略的效果。

- **计算**：`C(int32) += A(int8) × B(int8)`，核心指令 `_tile_dpbssd`。
- **目标硬件**：Intel Sapphire Rapids（实验机 Xeon w7-3565X，支持 `amx_tile / amx_int8 / amx_bf16`）。
- **理论峰值**：AMX int8 = 1024 MACs/cycle/core，利用率 = 实测 TOPS / 理论 TOPS。
- **约定**：所有矩阵 **row-major**（C/BLAS 惯例）。

## 目录结构

```
handwritten-v2/
├── CMakeLists.txt          # 顶层构建：公共 flags/标准/依赖(amx_common INTERFACE 库)
├── bench.toml              # 实验参数模板(freq/cores/dim/pack/MC…)，binary 与 bench.sh 共享
├── README.md               # 顶层文档：测试方法、优化策略（数据分块/软件预取/数据重排）
├── pics/                   # 文档引用的图片（*.png 走 git-lfs）
├── third_party/
│   └── CLI11.hpp           # 第三方单头文件 CLI 解析库（vendored，勿改）
├── src/
│   ├── common/             # 两个版本共用的公共头(include 路径由 CMake 暴露)
│   │   ├── numeric.hpp     # OFFSET2D 宏 + amx::min/max/ceil_div/round_up/round_down
│   │   ├── hw_prefetch.hpp # namespace HWPFCtrl: MSR 读写 + 硬件预取器开关
│   │   ├── cpu_affinity.hpp    # bind_thread_to_cpu / init_numa
│   │   ├── debug_print.hpp     # print_matrix + routine_graphs(ASCII 示意图)
│   │   ├── utils.hpp       # 伞头: 汇总以上四个(现有 #include "utils.hpp" 站点不变)
│   │   ├── thread_params.hpp   # amx::ThreadParams(框架与 kernel 的公共契约)
│   │   └── bench_harness.hpp   # 性能测试框架 PerformanceTester + test_correctness(自包含)
│   ├── offline-packing/    # 版本一：计时循环前，先把整个 A/B/C 预打包
│   │   ├── amx-gemm.hpp/.cpp   # kernel 实现
│   │   ├── gemm-test.cpp       # main() 入口
│   │   ├── CMakeLists.txt      # 目标 gemm-offline
│   │   └── sweep_k.sh          # 扫 K 维实验脚本
│   └── online-packing/     # 版本二：打包融合进计算循环（on-the-fly）
│       ├── amx-gemm.hpp/.cpp   # kernel 实现（重构版）
│       ├── gemm-test.cpp       # main() 入口
│       ├── CMakeLists.txt      # 目标 gemm-online
│       └── README.md           # GEMM 分类学详解（GEBP/GEPB/GEPM… + Goto 论文框架）
├── scripts/                # 编排层（跑测试用，不含 build）
│   ├── bench.sh            # 统一入口：--config 读 TOML，锁频+绑核+perf，trap 保证解锁
│   ├── cpu-isolation.py    # cgroup v2/IRQ/SMT 严格隔离，运行后恢复
│   ├── freq.sh             # 频率 lock/unlock
│   ├── corelist.py         # CPU 列表解析/展开
│   └── toml_get.py         # 从 TOML 取键(供 bench.sh 读 freq/cores/events/output)
└── tools/                  # 分析绘图脚本
    ├── plot_amx_util.py        # AMX 利用率 vs MNK
    ├── plot_stage_breakdown.py # pack/compute/unpack 各阶段开销占比
    └── plot_sweep_k.py         # 扫 K 维结果绘图
```

## 两个版本的关系（重要）

`offline-packing` 和 `online-packing` 是**同一算子的两代实现**，各自独立编译，互不依赖。
共享 `src/common/`（`utils.hpp`、`bench_harness.hpp`）和 `third_party/CLI11.hpp`
（include 路径由 CMake 的 `amx_common` 暴露，源码直接 `#include "utils.hpp"`，无 `../`）。

| | offline-packing | online-packing |
|---|---|---|
| 打包时机 | 计时前一次性 pack 整个 A/B/C | pack 融合进计算循环，逐 block 打包 |
| Buffer 设计 | 内嵌 `BufferA/B/C` 裸指针 + 手写 offset | 模板类 `Buffer<T>` 继承体系，支持 view/owning、Dense/Strided、Row/ColMajor |
| Kernel 选择 | 按 pack 开关选 `amx_gemm_core_*` 变体 | 按 M/N/K 大小自动选 7 种 routine（GEMM/GEPP/GEMP/GEPM/GEPB/GEBP/GEPDOT） |
| 参数 | `GEMMParams`（pack + swpf 开关） | `GEMMParams` + `BlockingConfig`（可调 MC/NC/KC）+ alpha/beta |
| 分类学 | 未实现完整分派 | 实现 Goto 论文的 8 情形分派（见 online README） |

`online-packing` 是较新、较完整的重构（见 git log：`add online-packing GEMM impl`）。
新工作优先基于 online 版本。改动时注意两版**不共享 kernel 代码**，别假设改一处两处都生效。

## 核心架构（两版通用的硬件抽象）

### Tile 寄存器分配：2A2B4C
8 个 tile 寄存器固定分配，micro-kernel 形状固定 **32×32×64 (MR×NR×KR)**：
- `C00/C01/C10/C11` (tile 0-3)：4 个 int32 累加器，各 16×16
- `A0/A1` (tile 4-5)：2 个 int8 A 分块，各 16×64
- `B0/B1` (tile 6-7)：2 个 int8 B 分块，KPACK 布局 16×64（VNNI：K 方向 4 个 int8 打包）
- 内循环 = 2×`TILELOADD` + 2×`TILELOADDT1` + 4×`_tile_dpbssd`（`run_4_tdp()`）

### L1 Bypass 数据流（项目的核心 insight）
AMX 支持从 **L2 直接加载 tile**（绕过 L1-D）。设计目标：**A & C 走 L1-D，B 驻留 L2**。
- L1 load 用 `_tile_loadd`（`load_*_l1`），L2 stream load 用 `_tile_stream_loadd`（`load_*_l2`，即 `TILELOADDT1`）。
- Cache blocking：online 默认 `MC=NC=512, KC=1280`（`BlockingConfig`，运行时可调）；
  offline 硬编码 `TM/TN/TK`（单核 1024/1024/1280，多核 512/512/1280）。

### 三大优化手段（详见顶层 README.md）
1. **数据分块**：让 working set 驻留 L2（~1.33MB），掩盖 L3/DRAM 延迟。
2. **软件预取**：`SWPFetcher`（offline）/ `SWPFHelper`（online）在循环中插 `_mm_prefetch`，降 L2 miss。
3. **数据重排 (pack)**：strided tileload → dense tileload，消除 L2 组相联 conflict miss
   （N 为 1024 倍数时尤其明显）。

### GEMM 分类学（online-packing）
按 M/N/K 相对 cache block 的大小分 8 种情形，规约到 3 条 routine（见 `find_best_routine()` 与 online README）：
- **Routine1** (M≥N): GEMM→GEPP→GEPB，最终 `GEPB_kernel`
- **Routine2** (M<N): GEMM→GEPP→GEBP，最终 `GEBP_kernel`
- **Routine3** (small M,N / large K): 退化到 `GEPDOT_` micro-kernel
- Pack 策略原则：**不重复 pack 同一块数据**；M 小不 packB，N 小不 packA，K 小不 unpackC。
- 注意 `find_best_routine()` 目前 `magic_number` 被**硬编码为 `0b100`（GEPB）**，
  自动分派那行被注释掉了 —— 这是当前的调试/实验状态，改动前先确认意图。

### 多线程模型（`GEMMKernelInt8MT`）
- 把 M×N 网格按 `MC×NC` 切成 block，**Round-Robin** 分给各线程，每 block 一个独立 `GEMMKernelInt8` 实例存入 `kernel_pool`。
- 每线程 `bind_thread_to_cpu(core_id)` 绑核，各自 `amx_init()`。NUMA-aware 时用 `init_numa()` 展开核列表。
- 单核路径与多核路径在 `make_factory` 里按 `core_list.size()` 分流。

## 构建与测试

**构建走 CMake**（顶层统一），**跑测试走 `scripts/`**（编排层）。两者分离：CMake 只管编译，
锁频/绑核/perf/numactl 都在脚本里。（旧的子目录 Makefile 仍在，属遗留，逐步淘汰。）

```bash
# --- 构建 ---
cmake -S . -B build          # 配置一次（生成 compile_commands.json 供 clangd）
cmake --build build -j       # 编译两个变体
#   产物: build/gemm-offline, build/gemm-online

# --- 跑测试（scripts/bench.sh 是统一入口）---
scripts/bench.sh -v online --config bench.toml
scripts/bench.sh -v offline --config bench.toml -m perf
scripts/bench.sh -v online --config bench.toml --dry-run
```

**关注点分离**：实验参数(freq/cores/dim/pack/MC…)集中在一份 **TOML**([bench.toml](bench.toml))，
binary 经 `--config` 直读，bench.sh 也从同一 TOML 读 freq(锁频)/cores(绑核)。
bench.sh 的 CLI 只留编排项。freq 单一数据源在 TOML；cores 支持 `"0-7"`(bench.sh 展开后转发
`--core-list` 给 binary，binary 靠 `allow_config_extras` 忽略 `cores` 键)。

`scripts/` 五件套：
- `bench.sh`：编排入口。`-v offline|online` 选变体，`--config <toml>`(必需)，`-m run|perf`，
  `--no-lock`/`--dry-run`。perf 事件从 TOML `events` 读。**用 trap 保证频率与 CPU 隔离一定恢复**。
- `cpu-isolation.py`：为 `cores` 所在物理核建立 isolated partition，迁移 IRQ、offline SMT sibling，结束后恢复。
- `freq.sh`：`lock <khz>`/`unlock`，恢复范围用 `FREQ_RESTORE_MIN/MAX/GOV` 覆盖（实验机专属）。
- `corelist.py`：统一解析 CPU 列表，并把 `0-3,8` 展开成核列表。
- `toml_get.py`：从 TOML 取顶层键(数组转逗号分隔)，供 bench.sh 读编排参数(用 `tomllib`)。

CLI/config 参数（全部可写进 `--config` TOML/INI）。注册按"谁消费谁注册"划分：
- **harness 注册**（`parse_args`，它自己消费的）：`-l/--core-list`, `-n/--node`, `-f/--freq`(kHz),
  `-r/--round`, `--no-hwpf`(默认开预取器，加此 flag 关闭), `--dim-m/-n/-k`(标量或 `start:end:step`), `-o/--output`。
- **变体注册**（各 main 的 `add_extra_opts`，kernel 概念）：`--no-packA/B/C`(绑到 `cfg.pack*`，harness 仍记入 CSV/表头)；
  offline 另有 `--no-swpfA/B/C`；online 另有 `--MC/--NC/--KC`、`--profile-single`（分阶段计时）。

harness 消费的项集中在 `BenchConfig`（`PerformanceTester::cfg`），解析与运行职责分离。
`PerformanceTester` 本身不绑定任何 GEMM 专属参数(pack/swpf/blocking)，保持对新变体的通用性。

**正确性测试**：`test_correctness()`（`bench_harness.hpp`），对比 AMX 结果与 `cpu_gemm_ref()`。
两个变体的 `gemm-test.cpp`（各自的 main）里都有现成调用，默认注释掉了，需要时取消注释。

### 测试框架流程（`PerformanceTester`）
1. `configure()`：解析 CLI/config → 按需开关硬件预取器（默认开，`--no-hwpf` 关，MSR `0x1A4`）→ 打开 CSV 日志 → 打印参数表头。
2. `benchmark(planner)`：遍历 `--dim-*` 展开的 (M,N,K) 列表，对每个尺寸 `make_operands` 分配 A/B/C →
   调 `planner` 建 kernel（含预打包，不计时）→ warm-up → 计时循环。
3. 报告 `Time / Perf(TOPS) / Util(%)`，终端打印定宽表格，同时写 **CSV**（默认 `gemm-i8-<核数>core.csv`，
   含表头 + 完整 config 列，供脚本/pandas 直读）。
- `GEMMPlanner` 是关键抽象：`(M,N,K,A,B,C) -> GEMMPlan`，把 **kernel 搭建/预打包** 与
  **被计时的纯计算** 分开，保证计时只覆盖真正的 GEMM。
- 分阶段计时走 `benchmark_stages(GEMMStagedPlanner)`：变体提供 `GEMMStagedPlan`（packA/packB/compute/unpackC
  四个可调用），harness 逐个计时填入 `StageProfile`，输出到 `<log>-stages.csv`。
- A/B/C 数据初始化可注入：`tester.init_operands = <OperandInit>`（默认 `fill_ones`；内置 `fill_const(a,b,c)`、
  `fill_random(seed,lo,hi)`）。值不影响 AMX 计时，主要用于正确性验证/复现。

## 硬性约束与约定

- **尺寸约束**：M、N 必须是 32 (MR/NR) 的倍数，K 必须是 64 (KR) 的倍数，否则 `std::abort()`。
  online 版还要求 blocking size ≥ micro-kernel size。
- **数据类型固定 int8→int32**；A/B 为 `int8_t`，C 为 `int32_t`。
- **内存对齐**：所有矩阵 buffer 64B (cacheline) 对齐；packed buffer 用 `MIN_STRIDE=64` 做 dense stride。
- **锁频后必须解锁**：`bench.sh` 用 trap 保证解锁；若直接调 `freq.sh lock`，务必配对 `freq.sh unlock`，
  否则 CPU 停在固定频率。`PerformanceTester` 析构时会自动重开被禁用的硬件预取器。
- **需要权限**：跑测试用 `sudo`（CPU 隔离、perf、MSR 读写、cpupower 都需要 root）。
- **`.gitignore`**：`build/`、`*.txt`（日志/结果）、`compile_commands.json` 不入库；`pics/*.png` 走 git-lfs。

## 代码风格

- 单头文件为主：kernel 声明在 `.hpp`，实现在同名 `.cpp`；用 `namespace amx`。
- 大量 `ALWAYS_INLINE` + `RESTRICT` + 宏（`OFFSET2D`、`load_tileX_lN`）压榨性能，改热路径务必保持内联属性。
- tile intrinsics 来自 `<immintrin.h>`；tile 配置用 `TileConfig`（64B 对齐结构 + `_tile_loadconfig`）。
- 注释中英混排，性能相关决策常有中文说明 —— 延续这个习惯，解释「为什么这样写」而非「写了什么」。
- 改 kernel 后：先 `test_correctness` 验证正确性，再 `make run` 看利用率是否回退。

## 术语速查

- **GEPB/GEBP/GEPDOT** 等：Goto《Anatomy of high-performance matrix multiplication》的分块 kernel 命名，详见 `online-packing/README.md`。
- **Panel / Block / Strip**：按 cache/register block size 对大矩阵的三级切分。
- **KPACK / VNNI**：AMX int8 要求 B 在 K 方向按 4 个 int8 交织（`KPACK_b8=4`）。
- **swpf**：software prefetch；**hwpf**：hardware prefetcher。
