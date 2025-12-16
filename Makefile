SRC = $(wildcard *.c) $(wildcard *.cpp)
OBJ = $(patsubst %.c,$(BUILD_DIR)/%.o,$(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC)))
DEP = $(OBJ:.o=.d)
BUILD_DIR = build
BIN = $(BUILD_DIR)/gemm-test

CC = gcc
CXX = g++

# 1. 基础优化
OPT_FLAGS = -O3 -march=native -fopenmp -flto=auto
# 2. 数学与循环优化
# -ffast-math: 激进的浮点优化
# -funroll-loops: 激进的循环展开
MATH_FLAGS = -ffast-math -funroll-loops
# 3. 调试与宏定义
# -g: 生成调试符号(不影响性能，方便 perf/vtune)
# -DNDEBUG: 禁用 assert
# -Wall: 开启警告，防止低级错误
DEBUG_FLAGS = -g -DNDEBUG -Wall -Wextra

CFLAGS = $(OPT_FLAGS) $(MATH_FLAGS) $(DEBUG_FLAGS) -fno-strict-aliasing
CFLAGS += -MMD -MP
LDFLAGS = $(OPT_FLAGS) -lnuma

core ?= 1

# core = 0,1,2 => NUM_CORE = 3
# core = 0-29  => NUM_CORE = 30
NUM_CORE := $(shell echo $(core) | awk '\
BEGIN{n=0} \
{ gsub(/,/," ",$$0); \
	for(i=1;i<=NF;i++){ \
		if ($$i ~ /-/) { \
		split($$i,a,"-"); n += a[2]-a[1]+1 \
		} else {n++} \
	} \
	print n \
}')

# core = 0,1,2 => CORE_LIST = 0 1 2
# core = 0-29  => CORE_LIST = 0 1 2 ... 29
CORE_LIST := $(shell echo $(core) | awk '\
BEGIN {OFS=" "} \
{ gsub(/,/," ",$$0); \
	for(i=1;i<=NF;i++){ \
		if ($$i ~ /-/) { \
			split($$i,a,"-"); \
			for(j=a[1];j<=a[2];j++) printf j" " \
		} else { printf $$i" " } \
	} \
}')

CORE_LIST_ARG := $(shell echo $(CORE_LIST) | sed 's/ /,/g')


FREQ ?= 3000000 # 3.0 GHz
LOOP ?= 10      # Number of rounds to run


# open-hyper-threading!
export KMP_AFFINITY=compact,1,0,granularity=fine


# Performance counter flags!

# 测每核/进程的 L1/L2/L3 命中与 miss（进程级）
PERFFLAGS += -e cycles -e instructions\
			 -e l2_rqsts.all_demand_references -e l2_rqsts.all_demand_miss\
			 -e l2_rqsts.swpf_hit -e l2_rqsts.swpf_miss

# 测 AMX 指令使用情况
# 			 -e exe.amx_busy
# 			 -e amx_ops_retired.int8

# L1D 缓存相关事件
#            -e l1d.hwpf_miss\
# 			 -e l1d.replacement\
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



#            -e mem_inst_retired.any
#            -e mem_inst_retired.all_loads -e mem_inst_retired.all_stores
#            -e mem_load_completed.l1_miss_any
# 			 -e mem_load_retired.l1_hit -e mem_load_retired.l1_miss
# 			 -e mem_load_retired.l2_hit -e mem_load_retired.l2_miss
# 			 -e mem_load_retired.l3_hit -e mem_load_retired.l3_miss

# 测系统/每 socket 的 DRAM 传输

lockfreq:
	@echo "🔒 Locking CPU frequency to $(FREQ)..."
	# 设置 governor 为 performance
	sudo cpupower frequency-set -g performance > /dev/null
	# 锁定最大最小频率为 $(FREQ)
	sudo cpupower frequency-set -d $(FREQ) -u $(FREQ) > /dev/null
	@echo "✅ CPU frequency locked to $(FREQ)"

unlockfreq:
	@echo "🔓 Restoring CPU frequency policy..."
	# 恢复频率范围（自动调节）
	sudo cpupower frequency-set -d 1.0GHz -u 5.0GHz > /dev/null
	# 恢复 powersave governor
	sudo cpupower frequency-set -g powersave > /dev/null
	@echo "✅ CPU frequency restored"

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ $(LDFLAGS) -o $@

-include $(wildcard $(DEP))

run: $(BIN)
	make lockfreq
	taskset -c $(core) sudo ./$(BIN) \
		--freq $(FREQ) --round $(LOOP) --core-list $(CORE_LIST_ARG)
	make unlockfreq

perf: $(BIN)
	make lockfreq
	taskset -c $(core) sudo perf stat $(PERFFLAGS) ./$(BIN) \
		--freq $(FREQ) --round $(LOOP) --core-list $(CORE_LIST_ARG)
	make unlockfreq


run-1-node: $(BIN)
	# node 0
	make lockfreq core="0-29"
	numactl --cpunodebind=0 --membind=0 ./$(BIN) \
		--freq $(FREQ) --round $(LOOP) --node 1
	make unlockfreq core="0-29"

run-2-node: $(BIN)
	# node 0,1
	make lockfreq core="0-59"
	numactl --cpunodebind=0,1 --membind=0,1 ./$(BIN) \
		--freq $(FREQ) --round $(LOOP) --node 2
	make unlockfreq core="0-59"
	
run-4-node: $(BIN)
	# node = 0,1,2,3
	make lockfreq core="0-119"
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(BIN) \
		--freq $(FREQ) --round $(LOOP) --node 4
	make unlockfreq core="0-119"
	
clean:
	rm -rf $(BUILD_DIR)

.DEFAULT_GOAL := $(BIN)
