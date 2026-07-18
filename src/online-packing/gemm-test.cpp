#include "amx-gemm.hpp"
#include "bench_harness.hpp"

using namespace amx;

// 分阶段计时用: 建好 kernel(含 alloc_buffers)后, 把四个阶段暴露成 GEMMStagedPlan。
// 计时/输出交给 PerformanceTester 统一处理(不再在这里手写打印)。
static GEMMStagedPlanner make_staged_planner(const GEMMParams& params,
                                             const BlockingConfig& blocking) {
    return [params, blocking](int M, int N, int K,
                              const int8_t* A, const int8_t* B, int32_t* C) -> GEMMStagedPlan {
        auto kernel = std::make_shared<GEMMKernelInt8>(
            M, N, K, K, N, N, A, B, C, params, blocking);
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
                                const BlockingConfig& blocking,
                                const ThreadParams& thread_params) {
    return [params, blocking, &thread_params](int M, int N, int K,
                                               const int8_t* A, const int8_t* B,
                                               int32_t* C) -> GEMMPlan {
        if (thread_params.core_list.size() == 1) {
            auto kernel = std::make_shared<GEMMKernelInt8>(
                M, N, K, K, N, N, A, B, C, params, blocking);
            GEMMKernelInt8::amx_init();
            return [kernel]() { kernel->GEMM(); };
        } else {
            auto kernel = std::make_shared<GEMMKernelInt8MT>(
                M, N, K, K, N, N, A, B, C, thread_params, params, blocking);
            kernel->init_kernels();
            return [kernel]() { kernel->GEMM(); };
        }
    };
}

int main(int argc, char** argv) {
    GEMMParams params;
    params.beta = 1.0f;
    BlockingConfig blocking;
    bool profile_single = false;

    PerformanceTester tester;
    // online 默认: 方阵 sweep 512..16384 step 256(可用 --dim-* / --config 覆盖)
    tester.cfg.dim_m = "512:16384:256";
    tester.cfg.dim_n = "512:16384:256";
    tester.cfg.dim_k = "512:16384:256";
    tester.configure(argc, argv,
        [&](CLI::App& app) {
            app.add_flag("--no-packA{false}", tester.cfg.packA, "Disable packing for matrix A");
            app.add_flag("--no-packB{false}", tester.cfg.packB, "Disable packing for matrix B");
            app.add_flag("--no-packC{false}", tester.cfg.packC, "Disable packing for matrix C");
            app.add_option("--MC", blocking.MC, "Blocking size in M dimension")
                ->check(CLI::PositiveNumber);
            app.add_option("--NC", blocking.NC, "Blocking size in N dimension")
                ->check(CLI::PositiveNumber);
            app.add_option("--KC", blocking.KC, "Blocking size in K dimension")
                ->check(CLI::PositiveNumber);
            app.add_flag("--profile-single", profile_single,
                         "Profile packA/packB/compute/unpackC for single-core runs");
        },
        [&]() {
            std::cout << "Cache Block Size: MC=" << blocking.MC
                      << ", NC=" << blocking.NC << ", KC=" << blocking.KC << "\n";
            std::cout << "  Software Prefetch: On\n";
            if (profile_single)
                std::cout << "  Single-core Stage Profile: On\n";
        }
    );

    params.packA = tester.cfg.packA;
    params.packB = tester.cfg.packB;
    params.packC = tester.cfg.packC;

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

    // ============= Performance Test ================
    if (profile_single)
        tester.benchmark_stages(make_staged_planner(params, blocking));
    else
        tester.benchmark(make_planner(params, blocking, tester.cfg.thread_params));

    return 0;
}
