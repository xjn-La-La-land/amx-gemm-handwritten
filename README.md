# AMX GEMM 手写算子实现

> 这是一个C++实现的手写AMX GEMM算子，支持多线程并行。
>
> 可以统计AMX运算部件的利用率，还可以查看硬件性能计数器。希望分析AMX GEMM的数据流瓶颈。

## 性能测试方法

1. `make`生成BIN文件`build/gemm-test`

2. 可以从命令行传入一些参数，用来控制性能测试条件，或者控制GEMM算子的行为

   `````
   AMX GEMM Performance Test
   Usage: ./build/gemm-test [OPTIONS]
   
   Options:
     -h,--help                   Print this help message and exit
     -n,--node INT               Number of NUMA nodes
     -l,--core-list INT ...      Core list (e.g., 0,1,2,3)
     -f,--freq FLOAT             CPU Frequency in kHz
     -r,--round INT              Loop count
     --no-hwpf                   Disable HW Prefetcher
     --no-packA{false}           Disable packing for matrix A
     --no-packB{false}           Disable packing for matrix B
     --no-packC{false}           Disable packing for matrix C
     --no-swpfA{false}           Disable software prefetch for matrix A
     --no-swpfB{false}           Disable software prefetch for matrix B
     --no-swpfC{false}           Disable software prefetch for matrix C
     -o,--output TEXT            Output log file path
   `````

3. 也可以用我们在Makefile中写好的脚本：

   1. `make lockfreq`/`make unlockfreq`使用**cpupower**工具来控制所有核心的频率。

   2. `make run`会先锁定所有核心的频率，然后通过**taskset**将测试绑定在选定的CPU核上，测试结束会释放所有核心的频率。可以传入这些参数：

      - CORE=0; CORE=0,1,2; CORE=0-15 --- 选择CPU核心
      - FREQ=3000000 --- 设定测试运行频率（单位KHz）
      - LOOP=1000 --- 设定测试运行次数
      - e.g. `make run CORE=0 FREQ=3000000 LOOP=1000`

   3. `make perf`在`make run`基础上，通过**perf**工具统计相关性能计数器的值。统计的性能计数事件在**$(PERFFLAGS)**中，可以在其中添加想要统计的事件。不过由于硬件性能计数器数量有限，建议同时统计2~4个事件。

      下面有一些比较有用的perf事件：

      `````
      -e cycles -e instruction
      
      # 测 AMX 指令使用情况
      # 			 -e exe.amx_busy
      # 			 -e amx_ops_retired.int8
      
      # L1D 缓存相关事件
      #            -e l1d.hwpf_miss
      # 			 -e l1d.replacement
      #            -e l1d_pend_miss.pending
      
      # L2 缓存相关事件
      # 			 -e l2_request.all -e l2_request.miss\
      #            -e l2_rqsts.references -e l2_rqsts.miss\
      # 1️⃣ Demand Data Reads（普通 load/store）
      #            -e l2_rqsts.all_demand_references -e l2_rqsts.all_demand_miss\
      #            -e l2_rqsts.all_demand_data_rd -e l2_rqsts.demand_data_rd_hit -e l2_rqsts.demand_data_rd_miss\
      # 2️⃣ RFO（store miss → Read For Ownership）
      #            -e l2_rqsts.all_rfo -e l2_rqsts.rfo_hit -e l2_rqsts.rfo_miss\
      # 3️⃣ Instruction fetch（code read）
      #            -e l2_rqsts.all_code_rd -e l2_rqsts.code_rd_hit -e l2_rqsts.code_rd_miss\
      # 4️⃣ 预取（HW / SW prefetch）
      #            -e l2_rqsts.all_hwpf -e l2_rqsts.hwpf_miss\
      #            -e l2_rqsts.swpf_hit -e l2_rqsts.swpf_miss
      # 5️⃣ 测 prefetch 的有效性
      #            -e l2_lines_out.useless_hwpf
      # 6️⃣ 估计 L2 bandwidth
      #            -e l2_lines_in.all
      # 7️⃣ 判断 L2 写回压力
      #            -e l2_lines_out.non_silent -e l2_lines_out.silent
      
      # L3 缓存相关事件
      # 1️⃣ L3 request counter（demand-only, 普通 load/store）
      #            -e longest_lat_cache.reference -e longest_lat_cache.miss
      # 2️⃣ retired load 中 hit L3 的不同情况
      #            -e mem_load_l3_hit_retired.xsnp_none (Load 在 L3 命中，并且不需要 snoop)
      #            -e mem_load_l3_hit_retired.xsnp_no_fwd (L3 hit，但需要 snoop 其他 core，未 forward)
      #            -e mem_load_l3_hit_retired.xsnp_fwd (L3 hit，但实际数据来自跨核 HitM forward)
      #            -e mem_load_l3_hit_retired.xsnp_miss (L3 hit，但 snoop miss（仍然是 L3 命中）)
      # 3️⃣ retired load 中 miss L3 的不同情况(NUMA相关！)
      #            -e mem_load_l3_miss_retired.local_dram (L3 miss → 本地 DRAM)
      #            -e mem_load_l3_miss_retired.remote_dram (L3 miss → 远端 DRAM)
      #            -e mem_load_l3_miss_retired.remote_hitm (L3 miss → 数据从远端 core 的 cache（Modified）forward)
      #            -e mem_load_l3_miss_retired.remote_fwd (L3 miss → 数据从远端 core 的 cache（Shared/Exclusive）forward)
      #            -e mem_load_l3_miss_retired.remote_pmm (L3 miss → Intel Optane PMM)
      `````

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

**我们分块算法的关键在于降低L2的Miss Rate。**所以我们需要尽可能地保障下一轮要用到的数据分块在被使用之前已经被预取到L2中。除了依赖L2的硬件预取器，我们还可以进行软件预取。

在gemm循环中插入预取指令：

![alt text](pics/image-1.png)

### 数据重排

我们发现，当N的大小取1024的倍数时，L2命中率会有一个明显的下降。这是因为L2命中率还会受到其相联度的影响！L2是16路组相联结构，而矩阵B、C的每一行相同位置的元素由于间隔是(2^n)B，更容易落在同一个Set中，这样就极易导致Conflict Miss，预取的数据又被替换出去了。

我们可以将矩阵数据在计算之前进行重排，**将strided tileload变成dense tileload**，即原来分布在不同行中要加载的数据按照访问顺序紧密排列。这样就能完全消除L2相联度的限制，同时也更有利于L2硬件预取器发挥最大效果。

![alt text](pics/image-2.png)

