#include "amx-gemm.hpp"
#include <iomanip>
#include "../CLI11.hpp" // CLIUtils

#define LINE "-------------------------------------------------------------------\n"

using namespace amx;

// Performance test
class PerformanceTester {
    ThreadParams thread_params = ThreadParams();
    GEMMParams gemm_params = GEMMParams();
    double frequency_hz = 2.3e9;
    int loop_count = 10;
    bool disable_hwpf = false;
    std::string log_filename;

public:
    PerformanceTester() = default;

    ~PerformanceTester() {
        if (disable_hwpf) { // 恢复硬件预取器
            HWPFCtrl::enable_prefetchers(thread_params.core_list);
        }
    }

    // 初始化测试环境
    void init_env(int argc, char** argv) {
        parse_args(argc, argv);

        // gemm_params.packA = false;
        // gemm_params.packB = false;
        // gemm_params.packC = false;

        // gemm_params.swpfB = false;
        // gemm_params.swpfC = false;
        // gemm_params.swpfA = false;
        // disable_hwpf = true;

        if (disable_hwpf) 
            HWPFCtrl::disable_prefetchers(thread_params.core_list);
        else
            HWPFCtrl::enable_prefetchers(thread_params.core_list);
        report_params();
    }
    
    void run_test(int M, int N, int K) {
        // malloc matrices
        auto deleter = [](void* p) { free(p); };
        std::unique_ptr<int8_t, decltype(deleter)> A(
            static_cast<int8_t*>(aligned_alloc(64, M * K * sizeof(int8_t))), deleter);
        std::unique_ptr<int8_t, decltype(deleter)> B(
            static_cast<int8_t*>(aligned_alloc(64, K * N * sizeof(int8_t))), deleter);
        std::unique_ptr<int32_t, decltype(deleter)> C(
            static_cast<int32_t*>(aligned_alloc(64, M * N * sizeof(int32_t))), deleter);

        // initialize matrices
        std::fill_n(A.get(), M * K, 1);
        std::fill_n(B.get(), K * N, 1);
        std::fill_n(C.get(), M * N, 1);

        if (thread_params.core_list.size() == 1) {
            GEMMKernelInt8 kernel(M, N, K, K, N, N, A.get(), B.get(), C.get(), gemm_params);
            kernel.amx_init();
            kernel.prepare_packed_data();
            kernel.amx_gemm_compute();
            kernel.restore_packed_data();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < loop_count; i++) {
                kernel.prepare_packed_data();
                kernel.amx_gemm_compute();
                kernel.restore_packed_data();
            }
            auto end_time = std::chrono::high_resolution_clock::now();

            double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
            report_performance(M, N, K, elapsed_seconds);
        } else {
            GEMMKernelInt8MT kernel(M, N, K, K, N, N, A.get(), B.get(), C.get(), thread_params);
            kernel.init_kernels();
            kernel.prepare_packed_data();
            kernel.amx_gemm_compute(); // warm up

            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < loop_count; i++) 
                kernel.amx_gemm_compute();
            auto end_time = std::chrono::high_resolution_clock::now();

            // kernel.restore_packed_data();

            double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
            report_performance(M, N, K, elapsed_seconds);
        }
    }

private:

    // 通过 CLI11库解析命令行参数
    void parse_args(int argc, char** argv) {
        CLI::App app{"AMX GEMM Performance Test"};
        app.add_option("-n,--node", thread_params.num_numa_node, "Number of NUMA nodes")
            ->each([&](const std::string&) { 
                thread_params.numa_aware = true; 
            });
        app.add_option("-l,--core-list", thread_params.core_list, "Core list (e.g., 0,1,2,3)")
            ->delimiter(',');
        app.add_option("-f,--freq", frequency_hz, "CPU Frequency in kHz");
        app.add_option("-r,--round", loop_count, "Loop count");
        app.add_flag("--no-hwpf", disable_hwpf, "Disable HW Prefetcher");
        // Pack 控制 (默认为 true，Flag 触发时设为 false)
        app.add_flag("--no-packA{false}", gemm_params.packA, "Disable packing for matrix A");
        app.add_flag("--no-packB{false}", gemm_params.packB, "Disable packing for matrix B");
        app.add_flag("--no-packC{false}", gemm_params.packC, "Disable packing for matrix C");
        // 软件预取控制 (默认为 true，Flag 触发时设为 false)
        app.add_flag("--no-swpfA{false}", gemm_params.swpfA, "Disable software prefetch for matrix A");
        app.add_flag("--no-swpfB{false}", gemm_params.swpfB, "Disable software prefetch for matrix B");
        app.add_flag("--no-swpfC{false}", gemm_params.swpfC, "Disable software prefetch for matrix C");

        app.add_option("-o,--output", log_filename, "Output log file path");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            app.exit(e); // 自动打印 Help 信息并退出
        }

        frequency_hz *= 1000.0; // 转换为 Hz

        if (log_filename.empty()) {
            int num_cores = thread_params.core_list.size();
            log_filename = "gemm-i8-" + std::to_string(num_cores) + "core.txt";
        }
        std::ofstream log_file(log_filename, std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << log_filename << std::endl;
            std::abort();
        }
    }


    // print test parameters
    void report_params() {
        std::cout << LINE;
        std::cout << "Running AMX int8 GEMM Performance Test on " << thread_params.core_list.size() << " CPU Cores, "
                  << "at " << std::fixed << std::setprecision(2) << frequency_hz / 1e9 << " GHz!\n";
        std::cout << "Matrix Layout: A - " << (gemm_params.packA ? "packed" : "normal") << ", "
                  << "B - " << (gemm_params.packB ? "packed" : "normal") << ", "
                  << "C - " << (gemm_params.packC ? "packed" : "normal") << "\n";
        std::cout << "Cache Block Size: TM=" << GEMMKernelInt8::TM
                  << ", TN=" << GEMMKernelInt8::TN
                  << ", TK=" << GEMMKernelInt8::TK << "\n";
        std::cout << "Prefetch Options:\n";
        std::cout << "  Hardware Prefetchers: " << (disable_hwpf ? "Off" : "On") << "\n";
        if (gemm_params.packA) {
            std::cout << "  Software Prefetch A: " << (gemm_params.swpfA ? "On" : "Off") << "\n";
            std::cout << "  Software Prefetch B: " << (gemm_params.swpfB ? "On" : "Off") << "\n";
            std::cout << "  Software Prefetch C: " << (gemm_params.swpfC ? "On" : "Off") << "\n";
        } else {
            std::cout << "  Software Prefetch: Off\n";
        }
        std::cout << LINE;
        print_header();
    }

    const int W_DIM = 8;        // M, N, K 的宽度
    const int W_TIME = 16;      // 时间宽度
    const int W_PERF = 16;      // TOPS 宽度
    const int W_UTIL = 16;      // 利用率宽度
    const int TOTAL_WIDTH = 3 * W_DIM + W_TIME + W_PERF + W_UTIL;

    void print_header() {
        std::cout << std::right // 整体右对齐
                << std::setw(W_DIM) << "M"
                << std::setw(W_DIM) << "N"
                << std::setw(W_DIM) << "K"
                << std::setw(W_TIME) << "Time (s)"
                << std::setw(W_PERF) << "Perf (TOPS)"
                << std::setw(W_UTIL) << "Util (%)" 
                << "\n";
        std::cout << std::string(TOTAL_WIDTH, '-') << "\n";
    }

    void print_data_row(int M, int N, int K, double elapsed_seconds, double tops, double utilization) {
        std::cout << std::right << std::fixed // 固定小数点格式
                << std::setw(W_DIM) << M
                << std::setw(W_DIM) << N
                << std::setw(W_DIM) << K
                << std::setw(W_TIME) << std::setprecision(6) << elapsed_seconds 
                << std::setw(W_PERF) << std::setprecision(2) << tops 
                << std::setw(W_UTIL) << std::setprecision(2) << (utilization * 100.0) 
                << "\n";
    }

    void write_log(int M, int N, int K, double elapsed_seconds, double tops, double utilization) {
        static std::ofstream log_file(log_filename, std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << log_filename << std::endl;
            return;
        }
        log_file << std::right << std::fixed
                << "M N K = " << std::setw(W_DIM) << M
                << " " << std::setw(W_DIM) << N
                << " " << std::setw(W_DIM) << K << ", "
                << "Elapsed time = " << std::setw(W_TIME) << std::setprecision(6) << elapsed_seconds << ", "
                << "Performance = " << std::setw(W_PERF) << std::setprecision(2) << tops << ", "
                << "Utilization = " << std::setw(W_UTIL) << std::setprecision(2) << (utilization * 100.0) << "%\n";
    }

    void report_performance(int M, int N, int K, double elapsed_seconds) {
        uint64_t mac_count = (uint64_t)M * N * K * loop_count;
        double tops = (double)mac_count * 2.0 / 1e12 / elapsed_seconds;

        // AMX int8 理论峰值计算能力：1024 MACs/cycle/core
        double ideal_mac_per_cycle = 1024 * thread_params.core_list.size();
        // double total_cycles = elapsed_seconds * frequency_hz;
        double ideal_tops = ideal_mac_per_cycle * frequency_hz * 2 / 1e12;
        double utilization = tops / ideal_tops;

        print_data_row(M, N, K, elapsed_seconds, tops, utilization);
        write_log(M, N, K, elapsed_seconds, tops, utilization); // 写入日志
    }
};



void test_correctness() 
{
    int M = 1024, N = 1024, K = 2048;
    // malloc matrices
    auto deleter = [](void* p) { free(p); };
    std::unique_ptr<int8_t, decltype(deleter)> A(
        static_cast<int8_t*>(aligned_alloc(64, M * K * sizeof(int8_t))), deleter);
    std::unique_ptr<int8_t, decltype(deleter)> B(
        static_cast<int8_t*>(aligned_alloc(64, K * N * sizeof(int8_t))), deleter);
    std::unique_ptr<int32_t, decltype(deleter)> C1(
        static_cast<int32_t*>(aligned_alloc(64, M * N * sizeof(int32_t))), deleter);
    std::unique_ptr<int32_t, decltype(deleter)> C2(
        static_cast<int32_t*>(aligned_alloc(64, M * N * sizeof(int32_t))), deleter);

    // initialize matrices
    std::fill_n(A.get(), M * K, 2);
    std::fill_n(B.get(), K * N, 1);
    std::fill_n(C1.get(), M * N, 1);
    std::fill_n(C2.get(), M * N, 1);

    GEMMParams params = {
        .packA = true,
        .packB = true,
        .packC = true,
        .swpfA = true,
        .swpfB = true,
        .swpfC = true
    };

    GEMMKernelInt8 kernel_amx(M, N, K, K, N, N, A.get(), B.get(), C1.get(), params);
    GEMMKernelInt8 kernel_ref(M, N, K, K, N, N, A.get(), B.get(), C2.get(), params);

    kernel_amx.amx_gemm();
    kernel_ref.cpu_gemm_ref();

    for (int i = 0; i < M * N; i++) {
        if (C1.get()[i] != C2.get()[i]) {
            std::cerr << "[Error] AMX GEMM result does not match reference at index " << i << "!\n";
        }
    }

    std::cout << "Correctness test passed! AMX GEMM results match reference implementation.\n";
    std::cout << "=============== AMX GEMM Results: ==================\n";
    kernel_amx.print_results();
    std::cout << "=============== Reference GEMM Results: ============\n";
    kernel_ref.print_results();
}




int main(int argc, char** argv) {
    // ============= Correctness Test ================
    // test_correctness();
    
    // ============= Performance Test ================
    PerformanceTester tester;
    tester.init_env(argc, argv);

    const int TM = GEMMKernelInt8::TM;
    const int TN = GEMMKernelInt8::TN;
    const int TK = GEMMKernelInt8::TK;

    for (int i = 512; i <= 16384; i += 256) {
        int m = i;
        int n = i;
        int k = i;
        tester.run_test(m, n, k);
    }

    // for (int i = 1; i <= 8; i++) 
    //     tester.run_test(TM, TN, i * TK);

    // tester.run_test(TM, TN, TK);

    return 0;
}