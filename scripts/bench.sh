#!/usr/bin/env bash
# bench.sh —— AMX GEMM 基准测试编排(取代原 Makefile 的 run/perf)。
#
# 用 trap 保证无论测试成功、失败还是被 Ctrl-C 中断, 频率和 CPU 隔离都会恢复
#
# 用法:
#   scripts/bench.sh -v <offline|online> --config <toml> [选项] [-- <透传给 gemm 的额外参数>]
#
# 选项:
#   -v, --variant   offline | online             (必需: 选哪个可执行文件)
#       --config    实验参数 TOML 文件            (必需)
#   -m, --mode      run | perf                   (默认 run; perf 挂 perf stat)
#       --build-dir CMake 构建目录                (默认 build)
#       --no-lock   不锁定系统频率
#       --dry-run   只打印将执行的命令, 不实际运行
#   -h, --help
#
# TOML 里 bench.sh 关心的键:
#   freq  = 3000000       # kHz, 用于锁频(与 binary 算利用率的分母同源)
#   cores  = "0-7"        # 核规格, 支持 0 / 0,1,2 / 0-15; taskset 模式用
#   events = ["cycles", "instructions", ...]   # (可选) perf 事件数组; 缺省用内置默认
#
# 示例:
#   scripts/bench.sh -v online  --config bench.toml            # 按 TOML 跑
#   scripts/bench.sh -v offline --config bench.toml -m perf    # 加 perf stat
#   scripts/bench.sh -v online  --config bench.toml -- -r 5    # -- 后临时覆盖 binary 参数

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=freq.sh
source "$SCRIPT_DIR/freq.sh"

# 从 TOML 取键(缺键返回空)。数组默认逗号拼接; 传 --lines 则逐行(见 toml_get.py)。
toml_get() { python3 "$SCRIPT_DIR/toml_get.py" "$CONFIG" "$@"; }

# 默认 perf 事件(数组形式, 每项一个事件名)。当 CLI 与 TOML 都没给时回落到此。
DEFAULT_EVENTS=(cycles instructions MEM_LOAD_RETIRED.L1_HIT MEM_LOAD_RETIRED.L1_MISS)

# 测 AMX 指令使用情况
# 			 -e exe.amx_busy
# 			 -e amx_ops_retired.int8

# L1D 缓存相关事件
#            -e MEM_LOAD_RETIRED.L1_HIT -e MEM_LOAD_RETIRED.L1_MISS
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



# 默认值
VARIANT=""
CONFIG=""
MODE="run"
BUILD_DIR="build"
NO_LOCK=0
DRY_RUN=0

# 打印文件顶部紧邻 shebang 的注释块作为帮助信息:
# 跳过第 1 行 shebang, 逐行去掉 "# " 前缀, 遇到第一个非注释行即停 —— 不依赖硬编码行号。
usage() {
    awk 'NR==1 { next }
         /^#/  { sub(/^# ?/, ""); print; next }
         { exit }' "${BASH_SOURCE[0]}"
}

die() { echo "bench.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--variant)  VARIANT="${2:?}"; shift 2 ;;
        --config)      CONFIG="${2:?}"; shift 2 ;;
        -m|--mode)     MODE="${2:?}"; shift 2 ;;
        --build-dir)   BUILD_DIR="${2:?}"; shift 2 ;;
        --no-lock)     NO_LOCK=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA_ARGS=("$@"); break ;;
        *)             die "unknown option: $1 (see --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# 校验与派生
# ---------------------------------------------------------------------------
case "$VARIANT" in
    offline) BIN="$BUILD_DIR/gemm-offline" ;;
    online)  BIN="$BUILD_DIR/gemm-online" ;;
    "")      die "missing --variant (offline|online)" ;;
    *)       die "invalid --variant: $VARIANT (offline|online)" ;;
esac
[[ "$MODE" == run || "$MODE" == perf ]] || die "invalid --mode: $MODE (run|perf)"
[[ -x "$BIN" ]] || die "executable not found: $BIN — build first: cmake --build $BUILD_DIR"
[[ -n "$CONFIG" ]] || die "missing --config <toml>"
[[ -f "$CONFIG" ]] || die "config file not found: $CONFIG"

# 从 TOML 读 bench.sh 关心的编排参数
FREQ="$(toml_get freq)"
CORES="$(toml_get cores)"

# 锁频需要 freq(除非 --no-lock)
if (( ! NO_LOCK )) && [[ -z "$FREQ" ]]; then
    die "config '$CONFIG' 缺 freq(锁频需要); 或用 --no-lock 跳过锁频"
fi
# 严格隔离按显式 CPU 列表工作，不猜测 NUMA node 中该选哪一个 SMT thread。
[[ -n "$CORES" ]] || die "config '$CONFIG' 需提供 cores(如 cores = \"0-7\")"

# 解析 perf 事件为 -e 参数数组: TOML events 数组, 缺则回落 DEFAULT_EVENTS。
declare -a PERF_EVENTS=()
readarray -t _evs < <(toml_get events --lines)   # 每行一个事件名(裸事件含逗号也安全)
if (( ${#_evs[@]} )); then
    for e in "${_evs[@]}"; do PERF_EVENTS+=(-e "$e"); done
else
    for e in "${DEFAULT_EVENTS[@]}"; do PERF_EVENTS+=(-e "$e"); done
fi

# 命令执行封装: --dry-run 时只打印
run_cmd() {
    printf '  + %s\n' "$*" >&2
    (( DRY_RUN )) && return 0
    "$@"
}

# dry-run / no-lock 感知的锁频封装
do_lock()   { (( NO_LOCK )) && return 0; run_cmd freq_lock "$FREQ"; }
do_unlock() { (( NO_LOCK )) && return 0; run_cmd freq_unlock; }

# ---------------------------------------------------------------------------
# 组装 binary 调用命令(数组形式, 避免 word-splitting/注入)
# ---------------------------------------------------------------------------
# binary 从 --config 读全套实验参数(round/dim/pack/MC…); bench.sh 只额外转发
# 绑核相关(核列表需展开成 int; binary 的 CLI 覆盖 TOML)。
GEMM_ARGS=(--config "$CONFIG")
declare -a LAUNCH=()   # binary 前缀(taskset)
declare -a WRAP=()     # perf 包装

# cores(支持 0-7)展开成 0,1,..,7; taskset 用原规格, binary 用展开列表。
core_list="$(python3 "$SCRIPT_DIR/corelist.py" "$CORES")"
LAUNCH=(taskset -c "$CORES")
GEMM_ARGS+=(--core-list "$core_list")

if [[ "$MODE" == perf ]]; then
    WRAP=(perf stat "${PERF_EVENTS[@]}")
fi

EXTRA_ARGS=("${EXTRA_ARGS[@]:-}")
[[ -z "${EXTRA_ARGS[0]:-}" ]] && EXTRA_ARGS=()

# binary 由 sudo 启动时，首次创建的日志会归 root。先由当前用户创建最终
# output 文件，让 root 进程只负责追加，并在旧日志不可写时尽早报错。
OUTPUT_PATH="$(toml_get output)"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; ++i)); do
    case "${EXTRA_ARGS[i]}" in
        -o|--output)
            ((i + 1 < ${#EXTRA_ARGS[@]})) || die "${EXTRA_ARGS[i]} requires a path"
            OUTPUT_PATH="${EXTRA_ARGS[++i]}"
            ;;
        --output=*) OUTPUT_PATH="${EXTRA_ARGS[i]#--output=}" ;;
        -o?*)       OUTPUT_PATH="${EXTRA_ARGS[i]#-o}" ;;
    esac
done
if [[ -n "$OUTPUT_PATH" ]]; then
    OUTPUT_DIR="${OUTPUT_PATH%/*}"
    [[ "$OUTPUT_DIR" == "$OUTPUT_PATH" ]] && OUTPUT_DIR="."
    [[ -d "$OUTPUT_DIR" ]] || die "output directory not found: $OUTPUT_DIR"
    run_cmd touch -- "$OUTPUT_PATH"
fi

# 把完整 taskset/perf/binary 命令放进 acquire 已建立的 isolated slice。
CMD=(sudo systemd-run --scope --quiet --slice=benchmark.slice)
CMD+=("${LAUNCH[@]}" "${WRAP[@]}" "$BIN" "${GEMM_ARGS[@]}" "${EXTRA_ARGS[@]}")

# ---------------------------------------------------------------------------
# 执行: 隔离 → 锁频 → 跑测试 → trap 逆序恢复
# ---------------------------------------------------------------------------
echo "== AMX GEMM bench: variant=$VARIANT mode=$MODE ==" >&2

ISOLATION_ACTIVE=0
LOCK_ACTIVE=0
cleanup() {
    local ec=$? cleanup_ec=0
    trap - EXIT INT TERM
    set +e
    if (( LOCK_ACTIVE )); then
        do_unlock || cleanup_ec=$?
    fi
    if (( ISOLATION_ACTIVE )); then
        run_cmd sudo "$SCRIPT_DIR/cpu-isolation.py" release || cleanup_ec=$?
    fi
    (( ec == 0 && cleanup_ec != 0 )) && ec=$cleanup_ec
    exit "$ec"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_cmd sudo "$SCRIPT_DIR/cpu-isolation.py" acquire "$CORES"
ISOLATION_ACTIVE=1
if (( ! NO_LOCK )); then
    LOCK_ACTIVE=1
    do_lock
fi

run_cmd "${CMD[@]}"

# 正常路径的解锁与隔离释放由 EXIT trap 统一处理
