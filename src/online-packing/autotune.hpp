#pragma once
// 经验式 autotune: 对每个 shape, 在 (kernel, loop_order) × (MC, NC, KC) 的合法网格里逐配置
// 计时, 选最快, 按 shape 缓存。所有候选对同一 shape 算出同一个 C(只差 packing / loop 顺序
// 与分块大小), 故直接对比计时合法。C++ 没有 Triton @autotune 的装饰器魔法, 本质就是这个
// "枚举候选 → 各跑几轮计时 → 取最快 → 按 shape 缓存"的循环。
//
// 依赖 amx-gemm.hpp: GEMMKernelInt8 / GEMMParams / LoopOrder / kernel_specs() / loop_order_str()。

#include "amx-gemm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace amx {

// 单核 GEMM 的 autotuner。持有配置与按 shape 的赢家缓存, 供计时循环(如 make_planner)复用。
// 非线程安全: 面向单核 planner 的顺序调用。
class Autotuner {
public:
    // 是否启用 autotune 不在此: 存在一个 Autotuner 实例即代表启用, 关闭时干脆不构建它。
    struct Options {
        bool tune_kernel = true;        // 关: 只调 blocking, kernel/loop_order 固定为 heuristic 选择
        std::vector<int> mcs, ncs, kcs; // blocking 候选; 空 = 该维用 base 单值(不扫)
        int reps = 5;                   // 每配置计时轮数
    };

    explicit Autotuner(Options opt) : opt_(std::move(opt)) {}

    const Options& options() const { return opt_; }

    // 返回该 shape 选定的 GEMMParams(在 base 上覆写 kernel/loop_order/MC/NC/KC)。
    // 无有效候选(空网格) -> 原样返回 base(kernel==nullptr -> ctor 的 select_kernel 走 heuristic)。
    // 在真实 A/B/C 上计时(会累加改写 C, 对 AMX 计时无影响); 要求 amx_init() 已在本线程调过。
    GEMMParams tune(int M, int N, int K,
                    const int8_t *A, const int8_t *B, int32_t *C) {
        const std::array<int, 3> key{M, N, K};
        if (auto it = cache_.find(key); it != cache_.end())
            return it->second; // 命中缓存: 跳过整轮调优

        const auto cfgs = build_configs(M, N, K, A, B, C);
        GEMMParams best = GEMMParams();
        double best_s = std::numeric_limits<double>::infinity();
        bool found = false;
        for (const GEMMParams &cfg : cfgs) {
            GEMMKernelInt8 k(M, N, K, K, N, N, A, B, C, cfg); // ctor 分配该 config 的 buffer(不计时)
            const double s = time_candidate(k);
            if (s < best_s) { best_s = s; best = cfg; found = true; }
        } // k 出作用域 -> buffer RAII 释放

        // 只缓存有效结果; 空网格时 best 仍是默认 GEMMParams -> 上层走 heuristic。
        if (found) cache_.emplace(key, best);
        return best;
    }

private:
    Options opt_;
    std::map<std::array<int, 3>, GEMMParams> cache_; // (M,N,K) -> 赢家配置

    // 搜索空间 = 合法 (kernel, loop_order) × 各自 loop order 涉及的 blocking 网格
    std::vector<GEMMParams> build_configs(
            int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C) const {
        auto axis = [](const std::vector<int> &v, int fb) {
            return v.empty() ? std::vector<int>{fb} : v; // 空 = 用 base 单值(不扫该维)
        };
        const GEMMParams base;
        const std::vector<int> MCS = axis(opt_.mcs, base.MC);
        const std::vector<int> NCS = axis(opt_.ncs, base.NC);
        const std::vector<int> KCS = axis(opt_.kcs, base.KC);

        std::vector<GEMMParams> cfgs;
        auto emit = [&](FuncPtr kernel, LoopOrder lo) {
            auto add = [&](int MC, int NC, int KC) {
                GEMMParams p = base; // 继承 alpha/beta
                p.kernel = kernel; p.loop_order = lo;
                p.MC = MC; p.NC = NC; p.KC = KC;
                cfgs.push_back(p);
            };
            switch (lo) {
            case LoopOrder::KN_GEPB:   for (int nc : NCS) for (int kc : KCS) add(base.MC, nc, kc); break;
            case LoopOrder::KM_GEBP:   for (int mc : MCS) for (int kc : KCS) add(mc, base.NC, kc); break;
            case LoopOrder::MN_GEPDOT: add(base.MC, base.NC, base.KC); break;
            case LoopOrder::NONE: break;
            }
        };

        if (opt_.tune_kernel) {
            for (const auto &s : GEMMKernelInt8::kernel_specs())
                emit(s.kernel, s.loop_order);
        } else {
            // 固定为 heuristic 在默认 blocking 下的选择: 探测一次读回它选的 kernel+loop order
            GEMMKernelInt8 probe(M, N, K, K, N, N, A, B, C);
            emit(probe.selected_kernel_ptr(), probe.loop_order());
        }
        return cfgs;
    }

    // 单配置计时: 1 warmup + reps 次稳态 GEMM(), 取 MIN(排名要最干净的下界)
    double time_candidate(GEMMKernelInt8 &k) const {
        k.GEMM(); // warmup: 触页 / 填 fused packed buffer / 热 tile 配置
        double best = std::numeric_limits<double>::infinity();
        for (int r = 0; r < opt_.reps; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            k.GEMM();
            const auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        return best;
    }
};

} // namespace amx
