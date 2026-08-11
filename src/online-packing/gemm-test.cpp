#include "amx-gemm.hpp"
#include "autotune.hpp"
#include "bench_harness.hpp"

#include <string>
#include <vector>

using namespace amx;

// online 自定义的日志信息载体: 随 GEMMPlan.info 传出, 由下方注册的 info_serialize 写入 CSV。
// 以后要记录更多"选择信息"(pack 计划、loop 顺序等), 往这里加字段, 同步 info_columns / info_serialize 即可。
struct KernelLogInfo {
    std::string kernel; // 选中的 kernel 名(GEPB/GEBP/...)
    std::string loop_order;
    bool packA, packB, packC;
    int MC, NC, KC;
};

// 分阶段计时用: 建好 kernel(含 alloc_buffers)后, 把四个阶段暴露成 GEMMStagedPlan。
// 计时/输出交给 PerformanceTester 统一处理(不再在这里手写打印)。
static GEMMStagedPlanner make_staged_planner(const GEMMParams& params) {
    return [params](int M, int N, int K,
                              const int8_t* A, const int8_t* B, int32_t* C) -> GEMMStagedPlan {
        auto kernel = std::make_shared<GEMMKernelInt8>(
            M, N, K, K, N, N, A, B, C, params);
        GEMMKernelInt8::amx_init();
        kernel->alloc_buffers();

        GEMMStagedPlan plan;
        plan.packA   = [kernel]() { kernel->pack_A(); };
        plan.packB   = [kernel]() { kernel->pack_B(); };
        plan.compute = [kernel]() { kernel->GEMM_compute(); };
        plan.unpackC = [kernel]() { kernel->unpack_C(); };
        return plan;
    };
}

static GEMMPlanner make_planner(const GEMMParams& params,
                                const ThreadParams& thread_params,
                                std::shared_ptr<Autotuner> tuner) {
    // tuner 由调用方构建(nullptr = 未开启 autotune); 它持有跨整个 sweep 复用的赢家缓存。
    return [params, &thread_params, tuner](int M, int N, int K,
                                           const int8_t* A, const int8_t* B,
                                           int32_t* C) -> GEMMPlan {
        // 多核路径: 暂不 autotune(设计中标注为后续扩展), 保持原行为。
        if (thread_params.core_list.size() != 1) {
            auto kernel = std::make_shared<GEMMKernelInt8MT>(
                M, N, K, K, N, N, A, B, C, thread_params, params);
            kernel->init_kernels();
            return GEMMPlan{[kernel]() { kernel->GEMM(); }};
        }

        GEMMKernelInt8::amx_init();
        const GEMMParams chosen_params = tuner ? tuner->tune(M, N, K, A, B, C) : params;

        // 用最终配置建真 kernel, 返回正常 plan + info(写入 CSV 的选择信息)。
        auto kernel = std::make_shared<GEMMKernelInt8>(
            M, N, K, K, N, N, A, B, C, chosen_params);
        auto info = std::make_shared<KernelLogInfo>(KernelLogInfo{
            .kernel     = kernel->kernel_name(),
            .loop_order = kernel->loop_order_name(),
            .packA      = kernel->is_A_packed(),
            .packB      = kernel->is_B_packed(),
            .packC      = kernel->is_C_packed(),
            .MC         = chosen_params.MC,
            .NC         = chosen_params.NC,
            .KC         = chosen_params.KC,
        });
        return GEMMPlan{[kernel]() { kernel->GEMM(); }, info};
    };
}

int main(int argc, char** argv) {
    GEMMParams params;
    bool profile_single = false;
    bool autotune = false;        // --autotune: 是否构建 Autotuner
    Autotuner::Options at;

    PerformanceTester tester;
    // 注册自定义日志列: 把选中的 kernel 配置写进 CSV(需在 configure() 前设置, 表头在那里写)
    tester.info_columns = "kernel,loop_order,packA,packB,packC,MC,NC,KC";
    tester.info_serialize = [](const std::shared_ptr<void>& p) -> std::string {
        const auto* k = static_cast<const KernelLogInfo*>(p.get());
        return k->kernel + "," + k->loop_order + "," +
               std::to_string(k->packA) + "," + std::to_string(k->packB) + "," +
               std::to_string(k->packC) + "," +
               std::to_string(k->MC) + "," + std::to_string(k->NC) + "," +
               std::to_string(k->KC);
    };
    // online 默认: 方阵 sweep 512..16384 step 256(可用 --dim-* / --config 覆盖)
    tester.cfg.dim_m = "512";
    tester.cfg.dim_n = "512";
    tester.cfg.dim_k = "1280";
    tester.configure(argc, argv,
        [&](CLI::App& app) {
            // 注: online 的 pack 选择完全由 M/N/K 尺寸驱动(find_best_routine),
            // kernel 不消费 params.packA/B/C, 故不提供 --no-pack* 开关。
            app.add_option("--MC", params.MC, "Blocking size in M dimension")
                ->check(CLI::PositiveNumber);
            app.add_option("--NC", params.NC, "Blocking size in N dimension")
                ->check(CLI::PositiveNumber);
            app.add_option("--KC", params.KC, "Blocking size in K dimension")
                ->check(CLI::PositiveNumber);
            app.add_flag("--profile-single", profile_single,
                         "Profile packA/packB/compute/unpackC for single-core runs");
            // ---- autotune(单核): 每 shape 经验式挑最快 (kernel,loop_order)+(MC,NC,KC) ----
            app.add_flag("--autotune", autotune,
                         "Empirically pick the fastest config per shape (single-core)");
            app.add_flag("--no-autotune-kernel{false}", at.tune_kernel,
                         "Don't tune kernel/loop_order; fix it to heuristic");
            app.add_option("--autotune-mc", at.mcs, "MC candidate list, e.g. 256,512,1024")
                ->delimiter(',')->check(CLI::PositiveNumber);
            app.add_option("--autotune-nc", at.ncs, "NC candidate list, e.g. 256,512,1024")
                ->delimiter(',')->check(CLI::PositiveNumber);
            app.add_option("--autotune-kc", at.kcs, "KC candidate list, e.g. 640,1280,2560")
                ->delimiter(',')->check(CLI::PositiveNumber);
            app.add_option("--autotune-reps", at.reps, "Timed GEMM() reps per candidate")
                ->check(CLI::PositiveNumber);
        },
        [&]() {
            std::cout << "Cache Block Size: MC=" << params.MC
                      << ", NC=" << params.NC << ", KC=" << params.KC << "\n";
            std::cout << "  Software Prefetch: On\n";
            if (profile_single)
                std::cout << "  Single-core Stage Profile: On\n";
            // 此回调在 parse 之后、表头之前运行, 故 at.* 与 --MC/--NC/--KC 均已定稿。
            // blocking 候选空 = 该维用 base 单值(Autotuner 内部按此回退), 显示时同样回退。
            if (autotune) {
                auto show = [](const std::vector<int>& v, int fb) {
                    if (v.empty()) return std::to_string(fb);
                    std::string s;
                    for (size_t i = 0; i < v.size(); ++i)
                        s += (i ? "," : "") + std::to_string(v[i]);
                    return s;
                };
                std::cout << "  Autotune: On (tune_kernel=" << (at.tune_kernel ? "yes" : "no")
                          << ", reps=" << at.reps << ")\n"
                          << "    MC candidates: " << show(at.mcs, params.MC) << "\n"
                          << "    NC candidates: " << show(at.ncs, params.NC) << "\n"
                          << "    KC candidates: " << show(at.kcs, params.KC) << "\n";
            }
        }
    );

    // ============= Performance Test ================
    // autotune 关闭时不构建 Autotuner, planner 收到 nullptr 直接走 heuristic(chosen = params)。
    auto tuner = autotune ? std::make_shared<Autotuner>(std::move(at)) : nullptr;
    if (profile_single)
        tester.benchmark_stages(make_staged_planner(params));
    else
        tester.benchmark(make_planner(params, tester.cfg.thread_params, tuner));

    return 0;



    // ============= Correctness Test ================
    // test_correctness(1024, 1024, 5120,
    //     make_planner(params, blocking, tester.cfg.thread_params),
    //     [](int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) -> GEMMPlan {
    //         GEMMParams ref_params = {.packA = true, .packB = true, .packC = true};
    //         auto k = std::make_shared<GEMMKernelInt8>(M, N, K, K, N, N, A, B, C, ref_params);
    //         GEMMKernelInt8::amx_init();
    //         return [k]() { k->cpu_gemm_ref(); };
    //     }
    // );
}
