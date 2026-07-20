#include "amx-gemm.hpp"
#include "bench_harness.hpp"

using namespace amx;

// online 自定义的日志信息载体: 随 GEMMPlan.info 传出, 由下方注册的 info_serialize 写入 CSV。
// 以后要记录更多"选择信息"(pack 计划、loop 顺序等), 往这里加字段, 同步 info_columns / info_serialize 即可。
struct KernelLogInfo {
    std::string kernel; // 选中的 kernel 名(GEPB/GEBP/...)
    std::string loop_order;
    bool packA, packB, packC;
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
                                const ThreadParams& thread_params) {
    return [params, &thread_params](int M, int N, int K,
                                               const int8_t* A, const int8_t* B,
                                               int32_t* C) -> GEMMPlan {
        if (thread_params.core_list.size() == 1) {
            auto kernel = std::make_shared<GEMMKernelInt8>(
                M, N, K, K, N, N, A, B, C, params);
            GEMMKernelInt8::amx_init();
            // select_kernel() 已在构造时选定 kernel, 把选择信息塞进自定义 info 载体随 plan 传出
            auto info = std::make_shared<KernelLogInfo>(KernelLogInfo{
                .kernel     = kernel->kernel_name(),
                .loop_order = kernel->loop_order_name(),
                .packA      = kernel->is_A_packed(),
                .packB      = kernel->is_B_packed(),
                .packC      = kernel->is_C_packed(),
            });
            return GEMMPlan{[kernel]() { kernel->GEMM(); }, info};
        } else {
            auto kernel = std::make_shared<GEMMKernelInt8MT>(
                M, N, K, K, N, N, A, B, C, thread_params, params);
            kernel->init_kernels();
            return [kernel]() { kernel->GEMM(); };
        }
    };
}

int main(int argc, char** argv) {
    GEMMParams params;
    params.beta = 1.0f;
    bool profile_single = false;

    PerformanceTester tester;
    // 注册自定义日志列: 把选中的 kernel 配置写进 CSV(需在 configure() 前设置, 表头在那里写)
    tester.info_columns = "kernel,loop_order,packA,packB,packC";
    tester.info_serialize = [](const std::shared_ptr<void>& p) -> std::string {
        const auto* k = static_cast<const KernelLogInfo*>(p.get());
        return k->kernel + "," + k->loop_order + "," +
               std::to_string(k->packA) + "," + std::to_string(k->packB) + "," +
               std::to_string(k->packC);
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
        },
        [&]() {
            std::cout << "Cache Block Size: MC=" << params.MC
                      << ", NC=" << params.NC << ", KC=" << params.KC << "\n";
            std::cout << "  Software Prefetch: On\n";
            if (profile_single)
                std::cout << "  Single-core Stage Profile: On\n";
        }
    );

    // ============= Performance Test ================
    if (profile_single)
        tester.benchmark_stages(make_staged_planner(params));
    else
        tester.benchmark(make_planner(params, tester.cfg.thread_params));

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
