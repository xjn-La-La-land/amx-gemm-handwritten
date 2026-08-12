#pragma once
// 自包含: 只依赖 common 层。ThreadParams 来自 thread_params.hpp,
// HWPFCtrl(禁用/恢复硬件预取器)来自 hw_prefetch.hpp。不依赖任何变体的 amx-gemm.hpp。

#include "hw_prefetch.hpp"    // HWPFCtrl::enable/disable_prefetchers
#include "page_alloc.hpp"
#include "thread_params.hpp"  // amx::ThreadParams
#include "CLI11.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define LINE "-------------------------------------------------------------------\n"

// ============================================================================
// Plan/Planner 抽象: 把"每规模一次的 setup(建 kernel + 预打包, 可含按形状的启发式选择)"
// 与 "被计时的纯 compute"分开。
//   GEMMPlanner(M,N,K,A,B,C) 做完所有 setup, 返回一个零参数的 GEMMPlan;
//   GEMMPlan() 每调一次执行一次 compute(捕获已建好的 kernel)。
// 计时循环只调 GEMMPlan, 故 setup 开销不计入。
// ============================================================================
// GEMMPlan: 被计时的纯 compute 回调 + 该 shape 选中的 kernel 名(仅写入 CSV, 不进终端表格)。
// kernel_name 留空则 CSV 该列为空(如 offline 版本 / 多核 MT 暂未实现命名)。
struct GEMMPlan {
    std::function<void()> run;  // compute kernel 调用
    std::shared_ptr<void> info; // kernel 相关信息

    GEMMPlan() = default;
    // 从任意无参可调用对象构造(lambda 可直接转换), 可选带自定义 info 载体。
    template <typename F,
              typename = std::enable_if_t<std::is_invocable_v<F>>>
    GEMMPlan(F&& f, std::shared_ptr<void> info = nullptr)
        : run(std::forward<F>(f)), info(std::move(info)) {}
};
using GEMMPlanner = std::function<GEMMPlan(
    int M, int N, int K,
    const int8_t* A, const int8_t* B, int32_t* C)>;

// ============================================================================
// 尺寸 sweep: 每个维度是标量(如 "1024")或区间("start:end:step", 含端点)。
// expand 后三维 zip: 标量重复以对齐最长维, 多个区间必须等长。
//   square sweep : --dim-m 512:16384:256 --dim-n 512:16384:256 --dim-k 512:16384:256
//   fixed-MN varK: --dim-m 32 --dim-n 32 --dim-k 64:4096:64
// ============================================================================
struct Dims { int M, N, K; };

inline std::vector<int> expand_dim_spec(const std::string& spec) {
    auto colon1 = spec.find(':');
    if (colon1 == std::string::npos) {           // 标量
        return { std::stoi(spec) };
    }
    auto colon2 = spec.find(':', colon1 + 1);
    if (colon2 == std::string::npos)
        throw std::invalid_argument("dim spec must be 'start:end:step': " + spec);
    int start = std::stoi(spec.substr(0, colon1));
    int end   = std::stoi(spec.substr(colon1 + 1, colon2 - colon1 - 1));
    int step  = std::stoi(spec.substr(colon2 + 1));
    if (step <= 0) throw std::invalid_argument("dim spec step must be > 0: " + spec);
    std::vector<int> out;
    for (int v = start; v <= end; v += step) out.push_back(v);
    return out;
}

// 三个维度规格 zip 成尺寸列表
inline std::vector<Dims> build_sweep(const std::string& m_spec,
                                     const std::string& n_spec,
                                     const std::string& k_spec) {
    std::vector<int> ms = expand_dim_spec(m_spec);
    std::vector<int> ns = expand_dim_spec(n_spec);
    std::vector<int> ks = expand_dim_spec(k_spec);
    size_t len = std::max({ms.size(), ns.size(), ks.size()});

    auto check = [&](const std::vector<int>& v, const char* name) {
        if (v.size() != 1 && v.size() != len)
            throw std::invalid_argument(std::string("dim ") + name +
                " must be scalar or length " + std::to_string(len));
    };
    check(ms, "m"); check(ns, "n"); check(ks, "k");

    auto at = [](const std::vector<int>& v, size_t i) {
        return v.size() == 1 ? v[0] : v[i];
    };
    std::vector<Dims> dims;
    dims.reserve(len);
    for (size_t i = 0; i < len; ++i)
        dims.push_back({ at(ms, i), at(ns, i), at(ks, i) });
    return dims;
}

// ============================================================================
// BenchConfig: 所有可配置项集中于此(POD)。由 CLI/config 文件解析填充,
// PerformanceTester 只消费它。字段与 CLI 选项一一对应。
// ============================================================================
struct BenchConfig {
    amx::ThreadParams thread_params;
    double frequency_khz = 3.0e6;     // 仅用于计算理论 TOPS 分母
    int    loop_count    = 10;
    bool   hwpf_enabled  = true;      // 默认开启硬件预取器; --no-hwpf 关闭(隔离软件预取效果)
    bool   packA = true, packB = true, packC = true;
    std::string log_filename;         // 空则默认 gemm-i8-<cores>core.csv
    // sweep 维度规格(标量或 start:end:step)
    std::string dim_m = "1024", dim_n = "1024", dim_k = "1024";
};

// 分阶段计时结果(online 的 pack/compute/unpack 拆解)
struct StageProfile {
    double packA_seconds   = 0.0;
    double packB_seconds   = 0.0;
    double compute_seconds = 0.0;
    double unpackC_seconds = 0.0;
    double total_seconds() const {
        return packA_seconds + packB_seconds + compute_seconds + unpackC_seconds;
    }
};

// GEMMPlan 的分阶段版: 把单趟 compute 拆成可分别计时的四个阶段(每阶段跑一次)。
// 用于 profile 模式。由 GEMMStagedPlanner 在 setup 阶段构建。
struct GEMMStagedPlan {
    std::function<void()> packA;
    std::function<void()> packB;
    std::function<void()> compute;
    std::function<void()> unpackC;
};
using GEMMStagedPlanner = std::function<GEMMStagedPlan(
    int M, int N, int K,
    const int8_t* A, const int8_t* B, int32_t* C)>;

// ============================================================================
// Operand 初始化: 填充 A/B/C 测试数据的回调。A/B 是 int8, C 是 int32。
// 值大小不影响 AMX 计时(同样的 MAC 数), 但对正确性验证/复现很有用。
// 默认 fill_ones; 通过 PerformanceTester::init_operands 覆盖。
// ============================================================================
using OperandInit = std::function<void(int M, int N, int K,
                                       int8_t* A, int8_t* B, int32_t* C)>;

// 全部填常量(默认 1)
inline OperandInit fill_const(int8_t a = 1, int8_t b = 1, int32_t c = 1) {
    return [a, b, c](int M, int N, int K, int8_t* A, int8_t* B, int32_t* C) {
        std::fill_n(A, (size_t)M * K, a);
        std::fill_n(B, (size_t)K * N, b);
        std::fill_n(C, (size_t)M * N, c);
    };
}

// 默认: A=B=C=1
inline void fill_ones(int M, int N, int K, int8_t* A, int8_t* B, int32_t* C) {
    fill_const(1, 1, 1)(M, N, K, A, B, C);
}

// A/B 随机 int8[lo,hi], C 清零(适合验证/压 corner case)。seed 固定则可复现。
inline OperandInit fill_random(unsigned seed = 42, int lo = -128, int hi = 127) {
    return [seed, lo, hi](int M, int N, int K, int8_t* A, int8_t* B, int32_t* C) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> d(lo, hi);
        for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = static_cast<int8_t>(d(rng));
        for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = static_cast<int8_t>(d(rng));
        std::fill_n(C, (size_t)M * N, int32_t{0});
    };
}

// ============================================================================
// PerformanceTester: 拥有一个 BenchConfig, 负责解析 → 环境准备 → sweep 计时 → 报告。
// ============================================================================
class PerformanceTester {
public:
    BenchConfig cfg;
    OperandInit init_operands = fill_ones;   // A/B/C 初始化回调; 可在 configure 后覆盖

    std::string info_columns; // 额外 CSV 列的表头
    std::function<std::string(const std::shared_ptr<void>&)> info_serialize; // 把某个 plan 的 info 序列化成对应的一行值

    PerformanceTester() = default;

    ~PerformanceTester() {
        // 若测试期间关闭了硬件预取器, 退出时恢复
        if (!cfg.hwpf_enabled)
            HWPFCtrl::enable_prefetchers(cfg.thread_params.core_list);
    }

    // 解析 CLI/config → 配置硬件预取器 → 打开 CSV 日志 → 打印参数表头。
    // add_extra_opts 注册变体专属选项; print_extra_params 在表头补充打印。
    void configure(int argc, char** argv,
                  std::function<void(CLI::App&)> add_extra_opts = nullptr,
                  std::function<void()> print_extra_params = nullptr) {
        parse_args(argc, argv, std::move(add_extra_opts));

        if (cfg.hwpf_enabled)
            HWPFCtrl::enable_prefetchers(cfg.thread_params.core_list);
        else
            HWPFCtrl::disable_prefetchers(cfg.thread_params.core_list);

        open_log();
        report_params(std::move(print_extra_params));
    }

    // 吞吐测试: 遍历 config 的 sweep 尺寸, 对每个 (M,N,K) 建 kernel(不计时)→ 计时循环 → 报告。
    void benchmark(GEMMPlanner planner) {
        for (const Dims& d : build_sweep(cfg.dim_m, cfg.dim_n, cfg.dim_k)) {
            Operands op = make_operands(d.M, d.N, d.K);
            GEMMPlan plan = planner(d.M, d.N, d.K,
                                    static_cast<int8_t*>(op.A.data()),
                                    static_cast<int8_t*>(op.B.data()),
                                    static_cast<int32_t*>(op.C.data()));
            std::string info_row = (info_serialize && plan.info) ? info_serialize(plan.info)
                                                                 : std::string();
            report_performance(d.M, d.N, d.K, measure(plan.run), info_row);
        }
    }

    // 分阶段测试: 遍历 sweep, 把每趟拆成 packA/packB/compute/unpackC 分别计时。
    void benchmark_stages(GEMMStagedPlanner planner) {
        stage_log_header();
        for (const Dims& d : build_sweep(cfg.dim_m, cfg.dim_n, cfg.dim_k)) {
            Operands op = make_operands(d.M, d.N, d.K);
            GEMMStagedPlan plan = planner(d.M, d.N, d.K,
                                          static_cast<int8_t*>(op.A.data()),
                                          static_cast<int8_t*>(op.B.data()),
                                          static_cast<int32_t*>(op.C.data()));
            StageProfile p;
            if (plan.packA)   p.packA_seconds   = measure(plan.packA);
            if (plan.packB)   p.packB_seconds   = measure(plan.packB);
            if (plan.compute) p.compute_seconds = measure(plan.compute);
            if (plan.unpackC) p.unpackC_seconds = measure(plan.unpackC);
            report_stages(d.M, d.N, d.K, p);
        }
    }

private:
    std::ofstream log_file;
    std::ofstream stage_file;

    template <typename T>
    static amx::page_alloc::Allocation alloc(size_t n) {
        return amx::page_alloc::allocate_huge_page(n * sizeof(T));
    }

    // 一组 GEMM 输入/输出缓冲(请求 2MB 大页)
    struct Operands {
        amx::page_alloc::Allocation A;
        amx::page_alloc::Allocation B;
        amx::page_alloc::Allocation C;
    };
    Operands make_operands(int M, int N, int K) {
        Operands op{ alloc<int8_t>((size_t)M * K),
                     alloc<int8_t>((size_t)K * N),
                     alloc<int32_t>((size_t)M * N) };
        init_operands(M, N, K,
                      static_cast<int8_t*>(op.A.data()),
                      static_cast<int8_t*>(op.B.data()),
                      static_cast<int32_t*>(op.C.data()));
        return op;
    }

    // 跑 loop_count 次并返回总秒数(不含一次 warm-up)
    double measure(const std::function<void()>& fn) {
        fn(); // warm up
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.loop_count; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    void parse_args(int argc, char** argv,
                    std::function<void(CLI::App&)> add_extra_opts) {
        CLI::App app{"AMX GEMM Performance Test"};
        app.set_config("--config", "", "Read options from a TOML/INI file (CLI overrides file)");
        app.allow_config_extras(true);
        
        app.add_option("-n,--node", cfg.thread_params.num_numa_node, "Number of NUMA nodes")
            ->each([&](const std::string&) { cfg.thread_params.numa_aware = true; });
        app.add_option("-l,--core-list", cfg.thread_params.core_list, "Core list (e.g., 0,1,2,3)")
            ->delimiter(',');
        app.add_option("-f,--freq", cfg.frequency_khz, "CPU Frequency in kHz");
        app.add_option("-r,--round", cfg.loop_count, "Loop count");
        app.add_flag("--no-hwpf{false}", cfg.hwpf_enabled, "Disable HW prefetchers (default: ON)");
        app.add_option("--dim-m", cfg.dim_m, "M sweep: scalar or start:end:step");
        app.add_option("--dim-n", cfg.dim_n, "N sweep: scalar or start:end:step");
        app.add_option("--dim-k", cfg.dim_k, "K sweep: scalar or start:end:step");
        app.add_option("-o,--output", cfg.log_filename, "Output CSV log file path");

        if (add_extra_opts) add_extra_opts(app);

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            std::exit(app.exit(e));
        }
    }

    void open_log() {
        if (cfg.log_filename.empty()) {
            int n = static_cast<int>(cfg.thread_params.core_list.size());
            cfg.log_filename = "gemm-i8-" + std::to_string(n) + "core.csv";
        }
        bool fresh = !std::ifstream(cfg.log_filename).good()
                     || std::ifstream(cfg.log_filename).peek() == std::ifstream::traits_type::eof();
        log_file.open(cfg.log_filename, std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << cfg.log_filename << "\n";
            std::abort();
        }
        if (fresh) {
            // pack 是 GEMM 专属概念, 不由 harness 硬编码; 变体经 info_columns 自行注册。
            log_file << "M,N,K,time_s,tops,util_pct,cores,freq_khz,hwpf";
            if (!info_columns.empty()) log_file << "," << info_columns;
            log_file << "\n";
        }
    }

    void report_params(std::function<void()> print_extra_params) {
        std::cout << LINE;
        std::cout << "Running AMX int8 GEMM Performance Test on "
                  << cfg.thread_params.core_list.size() << " CPU Cores, "
                  << "at " << std::fixed << std::setprecision(2)
                  << cfg.frequency_khz / 1e6 << " GHz!\n";
        std::cout << "Prefetch Options:\n";
        std::cout << "  Hardware Prefetchers: " << (cfg.hwpf_enabled ? "On" : "Off") << "\n";
        if (print_extra_params) print_extra_params();
        std::cout << LINE;
        print_header();
    }

    // ---- 终端 Table 格式 ----
    static constexpr int W_DIM = 8, W_TIME = 16, W_PERF = 16, W_UTIL = 16;
    static constexpr int TOTAL_WIDTH = 3 * W_DIM + W_TIME + W_PERF + W_UTIL;

    void print_header() {
        std::cout << std::right
                  << std::setw(W_DIM) << "M" << std::setw(W_DIM) << "N" << std::setw(W_DIM) << "K"
                  << std::setw(W_TIME) << "Time (s)" << std::setw(W_PERF) << "Perf (TOPS)"
                  << std::setw(W_UTIL) << "Util (%)" << "\n"
                  << std::string(TOTAL_WIDTH, '-') << "\n";
    }

    void print_data_row(int M, int N, int K, double s, double tops, double util) {
        std::cout << std::right << std::fixed
                  << std::setw(W_DIM) << M << std::setw(W_DIM) << N << std::setw(W_DIM) << K
                  << std::setw(W_TIME) << std::setprecision(6) << s
                  << std::setw(W_PERF) << std::setprecision(2) << tops
                  << std::setw(W_UTIL) << std::setprecision(2) << (util * 100.0) << "\n";
    }

    // ---- CSV 数据行(自描述: 含完整 config)----
    // info_row: 变体注册的 info_serialize 产出的一行(逗号分隔), 追加在核心列之后;
    void write_log(int M, int N, int K, double s, double tops, double util,
                   const std::string& info_row) {
        log_file << M << "," << N << "," << K << ","
                 << std::fixed << std::setprecision(6) << s << ","
                 << std::setprecision(4) << tops << ","
                 << std::setprecision(2) << (util * 100.0) << ","
                 << cfg.thread_params.core_list.size() << ","
                 << std::setprecision(0) << cfg.frequency_khz << ","
                 << cfg.hwpf_enabled;
        if (!info_row.empty())
            log_file << "," << info_row;

        log_file << "\n";
        log_file.flush();
    }

    void report_performance(int M, int N, int K, double elapsed_seconds,
                            const std::string& info_row) {
        uint64_t mac_count = (uint64_t)M * N * K * cfg.loop_count;
        double tops = (double)mac_count * 2.0 / 1e12 / elapsed_seconds;
        // AMX int8 理论峰值: 1024 MACs/cycle/core
        double ideal_tops = 1024.0 * cfg.thread_params.core_list.size()
                          * cfg.frequency_khz * 1000.0 * 2 / 1e12;
        double utilization = tops / ideal_tops;
        print_data_row(M, N, K, elapsed_seconds, tops, utilization);   // 终端表格不含自定义列
        write_log(M, N, K, elapsed_seconds, tops, utilization, info_row);
    }

    // ---- stage profile 输出: 独立 CSV(<log>-stages.csv) + 终端 ----
    void stage_log_header() {
        std::string name = cfg.log_filename.substr(0, cfg.log_filename.find_last_of('.')) + "-stages.csv";
        bool fresh = !std::ifstream(name).good()
                     || std::ifstream(name).peek() == std::ifstream::traits_type::eof();
        stage_file.open(name, std::ios::app);
        if (fresh) stage_file << "M,N,K,stage,seconds,share_pct\n";
    }

    void report_stages(int M, int N, int K, const StageProfile& p) {
        double total = p.total_seconds();
        auto emit = [&](const char* stage, double s) {
            double share = total > 0.0 ? s / total * 100.0 : 0.0;
            std::cout << "  " << std::left << std::setw(8) << stage
                      << std::right << std::setw(12) << std::fixed
                      << std::setprecision(6) << s << " s  ("
                      << std::setw(6) << std::setprecision(2) << share << "%)\n";
            if (stage_file.is_open())
                stage_file << M << "," << N << "," << K << "," << stage << ","
                           << std::fixed << std::setprecision(6) << s << ","
                           << std::setprecision(2) << share << "\n";
        };
        std::cout << "Stage Profile (M=" << M << " N=" << N << " K=" << K << "):\n";
        emit("packA", p.packA_seconds);
        emit("packB", p.packB_seconds);
        emit("compute", p.compute_seconds);
        emit("unpackC", p.unpackC_seconds);
        emit("total", total);
        std::cout << LINE;
        if (stage_file.is_open()) stage_file.flush();
    }
};

// 在同一输入上跑 amx_planner 与 ref_planner(各自独立输出缓冲), 逐元素比对。
inline void test_correctness(int M, int N, int K,
                             GEMMPlanner amx_planner,
                             GEMMPlanner ref_planner) {
    auto deleter = [](void* p) { free(p); };
    std::unique_ptr<int8_t, decltype(deleter)> A(static_cast<int8_t*>(aligned_alloc(64, M * K * sizeof(int8_t))), deleter);
    std::unique_ptr<int8_t, decltype(deleter)> B(static_cast<int8_t*>(aligned_alloc(64, K * N * sizeof(int8_t))), deleter);
    std::unique_ptr<int32_t, decltype(deleter)> C1(static_cast<int32_t*>(aligned_alloc(64, M * N * sizeof(int32_t))), deleter);
    std::unique_ptr<int32_t, decltype(deleter)> C2(static_cast<int32_t*>(aligned_alloc(64, M * N * sizeof(int32_t))), deleter);

    // 标准做法: A/B 随机 int8, C 清零。复用内置 fill_random(固定种子, 可复现)。
    fill_random(42)(M, N, K, A.get(), B.get(), C1.get());  // 填 A/B 随机, C1=0
    std::fill_n(C2.get(), M * N, int32_t{0});              // C2 同样清零

    amx_planner(M, N, K, A.get(), B.get(), C1.get()).run();
    ref_planner(M, N, K, A.get(), B.get(), C2.get()).run();

    bool passed = true;
    for (int i = 0; i < M * N; i++) {
        if (C1.get()[i] != C2.get()[i]) {
            std::cerr << "[Error] AMX GEMM result does not match reference at index "
                      << i << "!\n";
            passed = false;
        }
    }
    if (passed)
        std::cout << "Correctness test passed! AMX GEMM results match reference.\n";
}
