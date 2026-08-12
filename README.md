# AMX GEMM 手写算子实现

> 这是一个 C++ 实现的手写 AMX GEMM 算子。

## 性能测试方法

1. 用 **CMake** 构建（顶层统一配置，两个变体一起编译）：

   `````bash
   cmake -S . -B build      # 配置一次，顺带生成 compile_commands.json（供 clangd）
   cmake --build build -j   # 编译
   `````

   产物：

   - `build/gemm-offline` --- 离线预打包版本（计时前一次性 pack 整个 A/B/C）
   - `build/gemm-online` --- 在线打包版本（打包融合进计算循环）

2. 可以从命令行传入一些参数，用来控制性能测试条件，或者控制GEMM算子的行为

   `````
   AMX GEMM Performance Test
   Usage: ./build/gemm-<variant> [OPTIONS]

   Options:
     -h,--help                   Print this help message and exit
     --config TEXT               从 TOML/INI 文件读取选项(CLI 可覆盖文件)
     -n,--node INT               Number of NUMA nodes
     -l,--core-list INT ...      Core list (e.g., 0,1,2,3)
     -f,--freq FLOAT             CPU Frequency in kHz
     -r,--round INT              Loop count
     --no-hwpf                   关闭硬件预取器(默认开启)
     --no-packA/B/C{false}       Disable packing for matrix A/B/C
     --dim-m/--dim-n/--dim-k TEXT  尺寸 sweep: 标量(1024)或区间(start:end:step)
     -o,--output TEXT            输出 CSV 日志路径
   `````

   sweep 三维 zip: 标量重复对齐最长维,多个区间须等长。例:
   `--dim-m 512:16384:256 --dim-n 512:16384:256 --dim-k 512:16384:256`(方阵),
   或 `--dim-m 32 --dim-n 32 --dim-k 64:4096:64`(固定 MN、扫 K)。

   两个变体各有专属参数：`gemm-offline` 额外有 `--no-swpfA/B/C`（关软件预取）；
   `gemm-online` 额外有 `--MC/--NC/--KC`（调 cache blocking）、
   `--buffer-allocation auto|regular|huge`（控制 packed Buffer 分配策略）和
   `--profile-single`（分阶段计时，额外输出 `*-stages.csv`）。

   输出日志为 **CSV**。

3. 跑测试统一用 `scripts/bench.sh`（它负责锁频 + 绑核 + perf，并用 trap 保证测试结束/中断/失败后恢复系统状态）。

   **关注点分离**：所有实验参数（freq/cores/dim/pack/MC…）写在一份 **TOML** 里，binary 与 bench.sh 共享；
   bench.sh 自己的 CLI 只留编排项（选变体/模式/perf 事件）。参考模板 [bench.toml](bench.toml)。

   `````bash
   # 编辑 bench.toml 设定 freq / cores / round / dim-* 等，然后:
   scripts/bench.sh -v online  --config bench.toml            # 按 TOML 跑
   scripts/bench.sh -v offline --config bench.toml -m perf    # 默认严格隔离物理核
   scripts/bench.sh -v online  --config bench.toml --dry-run  # 只预览命令
   scripts/bench.sh -v online  --config bench.toml -- -r 5    # -- 后临时覆盖 binary 参数
   `````

   bench.sh 的 CLI 选项（编排层）：

   ```
   用法:
   scripts/bench.sh -v <offline|online> --config <toml> [选项] [-- <透传给 gemm 的额外参数>]

   选项:
   -v, --variant   offline | online             (必需: 选哪个可执行文件)
       --config    实验参数 TOML 文件             (必需)
   -m, --mode      run | perf                   (默认 run; perf 挂 perf stat)
       --build-dir CMake 构建目录                (默认 build)
       --no-lock   不锁定系统频率
       --dry-run   只打印将执行的命令, 不实际运行
   -h, --help
   ```

   频率的单独锁定/解锁也可直接用 `scripts/freq.sh lock <khz>` 和 `scripts/freq.sh unlock`（底层是 **cpupower**）。

   `cores` 模式默认使用 cgroup v2 isolated partition，暂停 `irqbalance`、迁移 IRQ，并临时 offline
   未选中的 SMT sibling；trap 会在测试结束后恢复。运行前可查看目标物理核和 housekeeping CPU：

   ```bash
   scripts/cpu-isolation.py plan 15
   ```

   由内核或驱动管理且无法在运行时迁移的 IRQ 会打印 warning 后跳过，因此仍可能产生少量中断干扰。

   若 benchmark 被 `SIGKILL` 导致 trap 无法执行，可手动恢复：

   ```bash
   sudo scripts/cpu-isolation.py release
   ```

   `release` 会恢复 cpuset partition、IRQ affinity、irqbalance 和 SMT online 状态。

## Perf events 分析

> 本节简单说明 Perf events 的使用，以及针对 AMX GEMM 数据流瓶颈分析的常用 event。

测试服务器 Xeon w7-3565X 是 **Intel Sapphire Rapids** 架构，权威事件定义见 Intel 官方 JSON：<https://github.com/intel/perfmon/blob/main/SPR/events/sapphirerapids_core.json>
（官网 <https://perfmon-events.intel.com/> 的渲染页不全，以 JSON 为准）。

### 芯片拓扑：core / offcore / uncore

先建立物理图景 —— perf 事件按「计数器在芯片上的物理位置」分三类：**core**（核内）、**offcore**（本核发往核外的请求，仍由 core PMU 计数）、**uncore**（mesh 上的共享部件，各带独立 PMU）。

```text
┌─────────────── Socket — Sapphire Rapids die (core vs uncore) ───────────────┐
│                                                                             │
│   ┌── Core 0 ───────────┐   ┌── Core 1 ───────────┐                         │
│   │ FE / EXE / retire   │   │ FE / EXE / retire   │   Per-core private:     │
│   │ L1D 48KB │ L2 2MB   │   │ L1D 48KB │ L2 2MB   │    · L1I 32KB / L1D 48KB│
│   │ ┌─────────────────┐ │   │ ┌─────────────────┐ │    · L2 2MB (private)   │
│   │ │    Core PMU     │ │   │ │    Core PMU     │ │    · Core PMU           │
│   │ └─────────────────┘ │   │ └─────────────────┘ │                         │
│   └──────────┬──────────┘   └──────────┬──────────┘                         │
│              │  offcore requests ↓     │                                    │
│  ════════════╪══ Mesh interconnect ════╪══════════════════════════  ← uncore│
│        ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────┐         │
│        │  CHA   │  │  CHA   │  │   IMC    │  │  UPI   │  │  IIO   │         │
│        │L3 slice│  │L3 slice│  │ mem ctrl │  │ cross- │  │  PCIe  │         │
│        │  +PMU  │  │  +PMU  │  │  → DRAM  │  │ socket │  │  +PMU  │         │
│        │        │  │        │  │   +PMU   │  │  +PMU  │  │        │         │
│        └────────┘  └────────┘  └──────────┘  └────────┘  └────────┘         │
│                  ↑ shared units = uncore, each has its own PMU              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **core 事件** —— 核内私有部件（前端/执行/退休、L1D、SPR 上**私有的 L2**、Core PMU）。归属到单个逻辑核，可 `taskset` 按进程测量。
- **offcore 事件** —— 仍由 core PMU 计数、仍可按进程测量，但描述「从本核发出、**离核**去 L3/内存」的请求。
- **uncore 事件** —— mesh 上的共享部件（**CHA**=L3 分片、**IMC**=内存控制器、**UPI**=跨 socket、**IIO**=PCIe），各有独立 PMU。**per-socket、无法归属到线程**，只能系统级 `perf stat -a` 测量 —— 真实 DRAM 带宽（`uncore_imc/cas_count_*/`）就出自这里。

### Intel PMU 的两类计数器

每个逻辑核有两组硬件性能计数器：

| 类型                      | 数量 | 特点                                                         |
| ------------------------- | ---- | ------------------------------------------------------------ |
| **Fixed counters**        | 4    | 硬件固定绑定特定事件，永远可用，**不占**可编程名额；`perf stat` 默认输出前 3 个 |
| **Programmable counters** | 8    | 可配置任意 core event，同一时刻最多 8 个；超出触发内核 multiplexing（分时复用 + 线性外推，读数变估算） |

因为可编程计数器只有 8 个，一次挂太多会失去精度，建议每轮设置 2~4 个 perf event。

4 个 fixed counter 中，我们需要关注两个：

| 事件                                  | 含义                                         |
| ------------------------------------- | -------------------------------------------- |
| `INST_RETIRED.ANY`（`instructions`）  | 退休指令数，最精确的指令计数                 |
| `CPU_CLK_UNHALTED.THREAD`（`cycles`） | 当前进程的所有线程非 halt 运行 cycle（之和） |

### AMX GEMM 分析推荐的事件组合

**1. AMX 利用率**

```toml
events = ["cycles", "instructions", "exe.amx_busy"]
```

- `exe.amx_busy / cycles` = AMX(TMUL) 部件占用率。这是 SPR 上**唯一**的 AMX 专属 core 事件，没有 per-op 细分。
- 满速时该比值应趋近 1；与 harness 报的 `Util(%)`（实测 TOPS / 理论 TOPS）互为印证。

**2. 内存层次命中率（诊断 blocking 是否让 working set 驻留 L2）**

```toml
events = ["cycles",
          "mem_load_retired.l1_hit", "mem_load_retired.l2_hit",
          "mem_load_retired.l3_hit", "mem_load_retired.l3_miss"]
```

- 这些是 **retired load** 的精确（PEBS）命中分布，层次为 `L1_HIT → L2_HIT → L3_HIT → L3_MISS`。
- 例如，`l3_miss` 明显上升说明分块过大、数据没法驻留 L2，掉到 DRAM。

**3. L2 带宽与预取有效性（调 swpf / hwpf）**

```toml
events = ["cycles",
          "l2_lines_in.all", "l2_lines_out.non_silent",
          "l2_lines_out.useless_hwpf",
          "l2_rqsts.swpf_hit", "l2_rqsts.swpf_miss"]
```

- L2↔L3 双向带宽 ≈ `(l2_lines_in.all + l2_lines_out.non_silent) × 64B / time`，可以对比理论 L2 带宽看是否打满。
- `useless_hwpf / l2_rqsts.all_hwpf` = 硬件预取无效率（预取进来还没用就被驱逐）—— 偏高说明 hwpf 帮倒忙，考虑 `--no-hwpf`。
- `swpf_hit / (swpf_hit + swpf_miss)` = 软件预取命中率。

**4. 执行停顿分析（AMX 没打满时，定位是不是 memory-bound）**

```toml
events = ["cycles", "exe.amx_busy",
          "cycle_activity.stalls_l2_miss", "cycle_activity.stalls_l3_miss",
          "exe_activity.bound_on_loads", "resource_stalls.sb"]
```

- `stalls_l3_miss` 高 → blocking 不够，B 掉出 L2。
- `bound_on_loads` 高 → load 是瓶颈，AMX 在等数据（供数不足）。
- `resource_stalls.sb` 高 → store buffer 满，C 写回压力大。

**5. NUMA 诊断（多节点 `-n` / `numactl` 时）**

```toml
events = ["cycles",
          "mem_load_l3_miss_retired.local_dram",
          "mem_load_l3_miss_retired.remote_dram",
          "mem_load_l3_hit_retired.xsnp_fwd",
          "offcore_requests_outstanding.l3_miss_demand_data_rd"]
```

- 正确绑定（`--cpunodebind=X --membind=X`）时 `remote_dram` 应接近 0；不为 0 = 内存分配越 NUMA。
- `xsnp_fwd` = 跨核 HitM forward。多核只读共享 A/B 时理论应为 0，出现则暗示 false sharing 或有核在改数据。



---

## 算子优化方案

在 2A2B4C 分配 tmm 寄存器的基础上，我们还加入了**数据分块、软件预取、数据重排**等优化手段，来提高算子对 AMX 部件的利用率。

### 数据分块

**对于AMX数据流来说，L2 是最关键的一级Cache**。L1-D 只有48KB，如果所有的Tile数据都经过L1-D，不仅是放不下，还会导致数据污染，让标量流水线（处理循环控制、地址计算的部分）疯狂发生L1 Miss。而L3又带宽太低、延迟太高。只有L2是容量和吞吐带宽的“甜点区”。

Intel针对AMX做出的最重要的架构优化就是**L1 Bypass**，支持直接从L2加载数据到Tile寄存器。

所以为了充分利用L2的空间，我们需要对大矩阵进行数据分块，让分块后的数据在计算时能够驻留在L2，减少L2 Miss的次数。同时，为了掩盖L3/内存延迟，数据分块在L2中还需要做Double Buffering或Ping-Pong Buffer，不管是由软件层面显式的预取，或者由硬件预取器去实现。L2中要包含当前计算的分块+下一轮预取的分块。

基于上面的考虑，我们设计的amx-gemm算子的数据流是这样的：A & C 进入L1-D，B 驻留在L2。

![alt text](pics/image.png)

这样核心循环中两条TILELOADD+两条TILELOADDT1，提供的带宽能让AMX接近全速运行。

L2-blocking的大小是这样设计的：**TN** **= 512, TK = 1280**.

- L2中矩阵数据的Working Set大小是 ~1.33MB，留出足够的空间给指令代码、栈和其他开销；
- L1-D中矩阵数据的Working Set大小是 44KB。

### 软件预取

**我们分块算法的关键在于降低L2的Miss Rate。** 所以我们需要尽可能地保障下一轮要用到的数据分块在被使用之前已经被预取到L2中。除了依赖L2的硬件预取器，我们还可以进行软件预取。

在gemm循环中插入预取指令：

![alt text](pics/image-1.png)

### 数据重排

我们发现，当N的大小取1024的倍数时，L2命中率会有一个明显的下降。这是因为L2命中率还会受到其相联度的影响！L2是16路组相联结构，而矩阵B、C的每一行相同位置的元素由于间隔是(2^n)B，更容易落在同一个Set中，这样就极易导致Conflict Miss，预取的数据又被替换出去了。

我们可以将矩阵数据在计算之前进行重排，**将strided tileload变成dense tileload**，即原来分布在不同行中要加载的数据按照访问顺序紧密排列。这样就能完全消除L2相联度的限制，同时也更有利于L2硬件预取器发挥最大效果。

![alt text](pics/image-2.png)
