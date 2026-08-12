#include "amx-gemm.hpp"
#include "bench_harness.hpp"

using namespace amx;

// offline 自定义日志列: 记录 pack/swpf 开关。随 GEMMPlan.info 传出。
struct PackLogInfo {
    bool packA, packB, packC;
    bool swpfA, swpfB, swpfC;
};

static GEMMPlanner make_planner(const GEMMParams& params,
                                const ThreadParams& thread_params) {
    return [params, &thread_params](int M, int N, int K,
                                    const int8_t* A, const int8_t* B,
                                    int32_t* C) -> GEMMPlan {
        auto info = std::make_shared<PackLogInfo>(
            PackLogInfo{params.packA, params.packB, params.packC,
                        params.swpfA, params.swpfB, params.swpfC});
        if (thread_params.core_list.size() == 1) {
            auto kernel = std::make_shared<GEMMKernelInt8>(
                M, N, K, K, N, N, A, B, C, params);
            GEMMKernelInt8::amx_init();
            kernel->prepare_packed_data();
            return GEMMPlan{[kernel]() { kernel->amx_gemm_compute(); }, info};
        } else {
            auto kernel = std::make_shared<GEMMKernelInt8MT>(
                M, N, K, K, N, N, A, B, C, thread_params);
            kernel->init_kernels();
            kernel->prepare_packed_data();
            return GEMMPlan{[kernel]() { kernel->amx_gemm_compute(); }, info};
        }
    };
}

int main(int argc, char** argv) {
    GEMMParams params;

    PerformanceTester tester;
    // 注册自定义日志列(需在 configure() 前设置, 表头在那里写)
    tester.info_columns = "packA,packB,packC,swpfA,swpfB,swpfC";
    tester.info_serialize = [](const std::shared_ptr<void>& p) -> std::string {
        const auto* i = static_cast<const PackLogInfo*>(p.get());
        return std::to_string(i->packA) + "," + std::to_string(i->packB) + "," +
               std::to_string(i->packC) + "," + std::to_string(i->swpfA) + "," +
               std::to_string(i->swpfB) + "," + std::to_string(i->swpfC);
    };
    // offline 默认: 固定 32x32, K 从 64 扫到 4096(可用 --dim-* / --config 覆盖)
    tester.cfg.dim_m = "32";
    tester.cfg.dim_n = "32";
    tester.cfg.dim_k = "64:4096:64";

    tester.configure(argc, argv,
        [&](CLI::App& app) {
            app.add_flag("--no-packA{false}", tester.cfg.packA, "Disable packing for matrix A");
            app.add_flag("--no-packB{false}", tester.cfg.packB, "Disable packing for matrix B");
            app.add_flag("--no-packC{false}", tester.cfg.packC, "Disable packing for matrix C");
            app.add_flag("--no-swpfA{false}", params.swpfA,
                         "Disable software prefetch for matrix A");
            app.add_flag("--no-swpfB{false}", params.swpfB,
                         "Disable software prefetch for matrix B");
            app.add_flag("--no-swpfC{false}", params.swpfC,
                         "Disable software prefetch for matrix C");
        },
        [&]() {
            std::cout << "Matrix Layout: A - " << (tester.cfg.packA ? "packed" : "normal") << ", "
                  << "B - " << (tester.cfg.packB ? "packed" : "normal") << ", "
                  << "C - " << (tester.cfg.packC ? "packed" : "normal") << "\n";
            if (tester.cfg.packA) {
                std::cout << "  Software Prefetch A: " << (params.swpfA ? "On" : "Off") << "\n";
                std::cout << "  Software Prefetch B: " << (params.swpfB ? "On" : "Off") << "\n";
                std::cout << "  Software Prefetch C: " << (params.swpfC ? "On" : "Off") << "\n";
            } else {
                std::cout << "  Software Prefetch: Off\n";
            }
        }
    );

    params.packA = tester.cfg.packA;
    params.packB = tester.cfg.packB;
    params.packC = tester.cfg.packC;

    // ============= Correctness Test ================
    // test_correctness(1024, 1024, 2048,
    //     make_planner(params, tester.cfg.thread_params),
    //     [](int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) -> GEMMPlan {
    //         auto k = std::make_shared<GEMMKernelInt8>(M, N, K, K, N, N, A, B, C);
    //         GEMMKernelInt8::amx_init();
    //         return [k]() { k->cpu_gemm_ref(); };
    //     }
    // );

    // ============= Performance Test ================
    tester.benchmark(make_planner(params, tester.cfg.thread_params));

    return 0;
}
