#include "../CLI11.hpp" // CLIUtils
#include "amx-gemm.hpp"
#include <iomanip>

#define LINE                                                                   \
  "-------------------------------------------------------------------\n"

using namespace amx;

// Performance test
class PerformanceTester {
  ThreadParams thread_params = ThreadParams();
  GEMMParams gemm_params = GEMMParams();
  BlockingConfig blocking = BlockingConfig();
  double frequency_hz = 2.3e9;
  int loop_count = 10;
  bool disable_hwpf = false;
  bool profile_single = false;
  std::string log_filename;
  std::ofstream log_file;

  struct StageProfile {
    double packA_seconds = 0.0;
    double packB_seconds = 0.0;
    double compute_seconds = 0.0;
    double unpackC_seconds = 0.0;
    double online_pack_seconds = 0.0;

    double total_seconds() const {
      return packA_seconds + packB_seconds + compute_seconds + unpackC_seconds;
    }
  };

public:
  PerformanceTester() = default;

  ~PerformanceTester() {
    if (disable_hwpf) { // 恢复硬件预取器
      HWPFCtrl::enable_prefetchers(thread_params.core_list);
    }
  }

  // 初始化测试环境
  void init_env(int argc, char **argv) {
    parse_args(argc, argv);

    if (disable_hwpf)
      HWPFCtrl::disable_prefetchers(thread_params.core_list);
    else
      HWPFCtrl::enable_prefetchers(thread_params.core_list);

    gemm_params.beta = 1.0f;
    report_params();
  }

  void offline_packing_wrapper(GEMMKernelInt8 &k) {
    k.prepare_packed_data();
    k.GEMM_compute();
    k.restore_packed_data();
  }

  void run_test(int M, int N, int K) {
    // malloc matrices
    auto deleter = [](void *p) { free(p); };
    std::unique_ptr<int8_t, decltype(deleter)> A(
        static_cast<int8_t *>(aligned_alloc(64, M * K * sizeof(int8_t))),
        deleter);
    std::unique_ptr<int8_t, decltype(deleter)> B(
        static_cast<int8_t *>(aligned_alloc(64, K * N * sizeof(int8_t))),
        deleter);
    std::unique_ptr<int32_t, decltype(deleter)> C(
        static_cast<int32_t *>(aligned_alloc(64, M * N * sizeof(int32_t))),
        deleter);

    // initialize matrices
    std::fill_n(A.get(), M * K, 1);
    std::fill_n(B.get(), K * N, 1);
    std::fill_n(C.get(), M * N, 1);

    if (thread_params.core_list.size() == 1) {
      GEMMKernelInt8 kernel(M, N, K, K, N, N, A.get(), B.get(), C.get(),
                            gemm_params, blocking);
      kernel.amx_init();

      if (profile_single) {
        StageProfile stage_profile = profile_single_kernel(kernel);
        report_performance(M, N, K, stage_profile.online_pack_seconds);
        report_stage_profile(stage_profile, log_file);
        report_stage_profile(stage_profile, std::cout);
      } else {
        // kernel.prepare_packed_data();
        // auto elapsed_seconds = measure_stage([&]() { kernel.GEMM_compute(); });
        auto elapsed_seconds = measure_stage([&]() { kernel.GEMM(); });
        report_performance(M, N, K, elapsed_seconds);
      }
    } else {
      GEMMKernelInt8MT kernel(M, N, K, K, N, N, A.get(), B.get(), C.get(),
                              thread_params, gemm_params, blocking);
      kernel.init_kernels();
      // kernel.prepare_packed_data();
      // kernel.GEMM_compute(); // warm up
      kernel.GEMM(); // warm up

      auto start_time = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < loop_count; i++)
        // kernel.GEMM_compute();
        kernel.GEMM();
      auto end_time = std::chrono::high_resolution_clock::now();

      // kernel.restore_packed_data();

      double elapsed_seconds =
          std::chrono::duration<double>(end_time - start_time).count();
      report_performance(M, N, K, elapsed_seconds);
    }
  }

private:
  template <typename Fn> double measure_stage(Fn &&fn) {
    fn(); // warm up
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loop_count; ++i) {
      fn();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  }

  StageProfile profile_single_kernel(GEMMKernelInt8 &kernel) {
    StageProfile profile;

    profile.online_pack_seconds = measure_stage(
        [&]() { kernel.GEMM(); }); // 包含 online packing/unpacking 的整体时间

    kernel.alloc_buffers(); // 包括 packA/B/C 的 buffer 分配
    profile.packA_seconds = measure_stage([&]() { kernel.pack_A(); });
    profile.packB_seconds = measure_stage([&]() { kernel.pack_B(); });
    profile.compute_seconds = measure_stage([&]() { kernel.GEMM_compute(); });
    profile.unpackC_seconds = measure_stage([&]() { kernel.unpack_C(); });

    return profile;
  }

  // 通过 CLI11库解析命令行参数
  void parse_args(int argc, char **argv) {
    CLI::App app{"AMX GEMM Performance Test"};
    app.add_option("-n,--node", thread_params.num_numa_node,
                   "Number of NUMA nodes")
        ->each([&](const std::string &) { thread_params.numa_aware = true; });
    app.add_option("-l,--core-list", thread_params.core_list,
                   "Core list (e.g., 0,1,2,3)")
        ->delimiter(',');
    app.add_option("-f,--freq", frequency_hz, "CPU Frequency in kHz");
    app.add_option("-r,--round", loop_count, "Loop count");
    app.add_flag("--no-hwpf", disable_hwpf, "Disable HW Prefetcher");
    app.add_flag("--profile-single", profile_single,
                 "Profile packA/packB/compute/unpackC for single-core runs");
    // Pack 控制 (默认为 true，Flag 触发时设为 false)
    app.add_flag("--no-packA{false}", gemm_params.packA,
                 "Disable packing for matrix A");
    app.add_flag("--no-packB{false}", gemm_params.packB,
                 "Disable packing for matrix B");
    app.add_flag("--no-packC{false}", gemm_params.packC,
                 "Disable packing for matrix C");
    // Blocking size control
    app.add_option("--MC", blocking.MC, "Blocking size in M dimension")
        ->check(CLI::PositiveNumber);
    app.add_option("--NC", blocking.NC, "Blocking size in N dimension")
        ->check(CLI::PositiveNumber);
    app.add_option("--KC", blocking.KC, "Blocking size in K dimension")
        ->check(CLI::PositiveNumber);

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
    log_file.open(log_filename, std::ios::app);
    if (!log_file.is_open()) {
      std::cerr << "Failed to open log file: " << log_filename << std::endl;
      std::abort();
    }
  }

  // print test parameters
  void report_params() {
    std::cout << LINE;
    std::cout << "Running AMX int8 GEMM Performance Test on "
              << thread_params.core_list.size() << " CPU Cores, "
              << "at " << std::fixed << std::setprecision(2)
              << frequency_hz / 1e9 << " GHz!\n";
    std::cout << "Matrix Layout: A - "
              << (gemm_params.packA ? "packed" : "normal") << ", "
              << "B - " << (gemm_params.packB ? "packed" : "normal") << ", "
              << "C - " << (gemm_params.packC ? "packed" : "normal") << "\n";
    std::cout << "GEMM Params: alpha=" << gemm_params.alpha
              << ", beta=" << gemm_params.beta << "\n";
    std::cout << "Cache Block Size: MC=" << blocking.MC
              << ", NC=" << blocking.NC << ", KC=" << blocking.KC << "\n";
    std::cout << "Prefetch Options:\n";
    std::cout << "  Hardware Prefetchers: " << (disable_hwpf ? "Off" : "On")
              << "\n";
    std::cout << "  Software Prefetch: On\n";
    if (profile_single) {
      std::cout << "  Single-core Stage Profile: On\n";
    }
    std::cout << LINE;
    print_header();
  }

  const int W_DIM = 8;   // M, N, K 的宽度
  const int W_TIME = 16; // 时间宽度
  const int W_PERF = 16; // TOPS 宽度
  const int W_UTIL = 16; // 利用率宽度
  const int TOTAL_WIDTH = 3 * W_DIM + W_TIME + W_PERF + W_UTIL;

  void print_header() {
    std::cout << std::right // 整体右对齐
              << std::setw(W_DIM) << "M" << std::setw(W_DIM) << "N"
              << std::setw(W_DIM) << "K" << std::setw(W_TIME) << "Time (s)"
              << std::setw(W_PERF) << "Perf (TOPS)" << std::setw(W_UTIL)
              << "Util (%)"
              << "\n";
    std::cout << std::string(TOTAL_WIDTH, '-') << "\n";
  }

  void print_data_row(int M, int N, int K, double elapsed_seconds, double tops,
                      double utilization) {
    std::cout << std::right << std::fixed // 固定小数点格式
              << std::setw(W_DIM) << M << std::setw(W_DIM) << N
              << std::setw(W_DIM) << K << std::setw(W_TIME)
              << std::setprecision(6) << elapsed_seconds << std::setw(W_PERF)
              << std::setprecision(2) << tops << std::setw(W_UTIL)
              << std::setprecision(2) << (utilization * 100.0) << "\n";
  }

  void write_log(int M, int N, int K, double elapsed_seconds, double tops,
                 double utilization) {
    log_file << std::right << std::fixed << "M N K = " << std::setw(W_DIM) << M
             << " " << std::setw(W_DIM) << N << " " << std::setw(W_DIM) << K
             << ", "
             << "Elapsed time = " << std::setw(W_TIME) << std::setprecision(6)
             << elapsed_seconds << ", "
             << "Performance = " << std::setw(W_PERF) << std::setprecision(2)
             << tops << ", "
             << "Utilization = " << std::setw(W_UTIL) << std::setprecision(2)
             << (utilization * 100.0) << "%\n";
  }

  void report_stage_profile(const StageProfile &profile,
                            std::ostream &os = std::cout) {
    const double total = profile.total_seconds();
    auto print_stage = [&](const char *name, double seconds) {
      double share = total > 0.0 ? seconds / total * 100.0 : 0.0;
      os << "  " << std::left << std::setw(8) << name << std::right
         << std::setw(12) << std::fixed << std::setprecision(6) << seconds
         << " s"
         << "  (" << std::setw(6) << std::setprecision(2) << share << "%)\n";
    };

    os << "Stage Profile:\n";
    print_stage("packA", profile.packA_seconds);
    print_stage("packB", profile.packB_seconds);
    print_stage("compute", profile.compute_seconds);
    print_stage("unpackC", profile.unpackC_seconds);
    print_stage("total", total);
    print_stage("online_packing", profile.online_pack_seconds);
    os << LINE;
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

void test_correctness() {
  int M = 1024, N = 1024, K = 5120;
  // malloc matrices
  auto deleter = [](void *p) { free(p); };
  std::unique_ptr<int8_t, decltype(deleter)> A(
      static_cast<int8_t *>(aligned_alloc(64, M * K * sizeof(int8_t))),
      deleter);
  std::unique_ptr<int8_t, decltype(deleter)> B(
      static_cast<int8_t *>(aligned_alloc(64, K * N * sizeof(int8_t))),
      deleter);
  std::unique_ptr<int32_t, decltype(deleter)> C1(
      static_cast<int32_t *>(aligned_alloc(64, M * N * sizeof(int32_t))),
      deleter);
  std::unique_ptr<int32_t, decltype(deleter)> C2(
      static_cast<int32_t *>(aligned_alloc(64, M * N * sizeof(int32_t))),
      deleter);

  // initialize matrices
  std::fill_n(A.get(), M * K, 2);
  std::fill_n(B.get(), K * N, 1);
  std::fill_n(C1.get(), M * N, 1);
  std::fill_n(C2.get(), M * N, 1);

  GEMMParams params = {
      .packA = true,
      .packB = true,
      .packC = true,
  };

  GEMMKernelInt8 kernel_amx(M, N, K, K, N, N, A.get(), B.get(), C1.get(),
                            params);
  GEMMKernelInt8 kernel_ref(M, N, K, K, N, N, A.get(), B.get(), C2.get(),
                            params);

  kernel_amx.amx_init();
  kernel_amx.GEMM();
  kernel_ref.cpu_gemm_ref();

  for (int i = 0; i < M * N; i++) {
    if (C1.get()[i] != C2.get()[i]) {
      std::cerr << "[Error] AMX GEMM result does not match reference at index "
                << i << "!\n";
    }
  }

  std::cout << "Correctness test passed! AMX GEMM results match reference "
               "implementation.\n";
  std::cout << "=============== AMX GEMM Results: ==================\n";
  kernel_amx.print_results();
  std::cout << "=============== Reference GEMM Results: ============\n";
  kernel_ref.print_results();
}

int main(int argc, char **argv) {
  // ============= Correctness Test ================
  // test_correctness();

  // ============= Performance Test ================
  PerformanceTester tester;
  tester.init_env(argc, argv);

  const int MC = BlockingConfig::DEFAULT_MC;
  const int NC = BlockingConfig::DEFAULT_NC;
  const int KC = BlockingConfig::DEFAULT_KC;

  for (int i = 512; i <= 16384; i += 256) {
    int m = i;
    int n = i;
    int k = i;
    tester.run_test(m, n, k);
  }

  // tester.run_test(12800, 12800, 12800);

  // for (int i = 1; i <= 8; i++)
  //     tester.run_test(MC, NC, i * KC);

  // tester.run_test(MC, NC, KC);

  return 0;
}
