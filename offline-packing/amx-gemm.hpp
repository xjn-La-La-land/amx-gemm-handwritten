#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <memory>
#include <thread>
#include <optional>

#include "../utils.hpp"

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif


namespace amx {

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Tile Register Constants
#define MAX_ROWS     16
#define MAX_COLS_i8  64
#define MAX_COLS_i32 16
#define TILE_SIZE_i8  (MAX_ROWS * MAX_COLS_i8)
#define TILE_SIZE_i32 (MAX_ROWS * MAX_COLS_i32)
#define MIN_STRIDE 64 // minimum stride in bytes
#define CACHELINE_SIZE 64

#define KPACK_b8  4
#define KPACK_b16 2
#define KPACK_b32 1

// 2A2B4C Tile Blocking
#define M_STEP (MAX_ROWS * 2)
#define N_STEP (MAX_ROWS * 2)
#define K_STEP MAX_COLS_i8

// Define tile config data structure
struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    std::array<uint8_t, 14> reserved_0 = {};
    std::array<uint16_t, 8> colsb;
    std::array<uint8_t, 16> reserved_1 = {};
    std::array<uint8_t,  8> rows;
    std::array<uint8_t,  8> reserved_2 = {};

    TileConfig() {
        palette_id = 1;
        start_row = 0;
        for (int i = 0; i < 8; i++) {
        set_row_col(i, 0, 0);
        }
    }

    void set_row_col(int i, uint8_t row, uint16_t col) {
        colsb[i] = col;
        rows[i] = row;
    }
    void set_config() { _tile_loadconfig(this); }
};
static_assert(sizeof(TileConfig) == 64);

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("Fail to do XFEATURE_XTILEDATA\n");
        return false;
    } else {
        // printf("TILE DATA USE SET - OK \n");
        return true;
    }
}

// parameter structure for GEMM kernel
struct GEMMParams {
    // Data layout
    bool packA = true;
    bool packB = true;
    bool packC = true;

    // SWPF settings
    bool swpfA = true;
    bool swpfB = true;
    bool swpfC = true;
};

// AMX GEMM Kernel for int8
class GEMMKernelInt8 {
public:
    // tile register alloc
    #define C00 0
    #define C01 1
    #define C10 2
    #define C11 3
    #define A0  4
    #define A1  5
    #define B0  6
    #define B1  7
    // cache blocking sizes
    static constexpr int TM = 512;
    static constexpr int TN = 512;
    static constexpr int TK = 1280;

    GEMMKernelInt8(int M, int N, int K,
                   int lda, int ldb, int ldc,
                   const void* RESTRICT A,
                   const void* RESTRICT B,
                   void* RESTRICT C,
                   const GEMMParams& params = GEMMParams())
        : M(M), N(N), K(K),
          lda(lda), ldb(ldb), ldc(ldc),
          A(static_cast<const int8_t*>(A)),
          B(static_cast<const int8_t*>(B)),
          C(static_cast<int32_t*>(C)),
          params(params),
          bufferA(this), bufferB(this), bufferC(this)
    {
        if (M % M_STEP != 0 || N % N_STEP != 0 || K % K_STEP != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of blocking sizes!\n";
            std::abort();
        }
    }

    // Initialize AMX and tile configuration
    static void amx_init() {
        if (!set_tiledata_use()) {
            std::cerr << "AMX initialization failed! Check if your CPU supports AMX." << std::endl;
            std::abort();
        }
        TileConfig tile_data; // Initialize tile configuration
        // tile 0,1,2,3 for C
        tile_data.set_row_col(C00, MAX_ROWS, MAX_COLS_i32 * sizeof(int32_t));
        tile_data.set_row_col(C01, MAX_ROWS, MAX_COLS_i32 * sizeof(int32_t));
        tile_data.set_row_col(C10, MAX_ROWS, MAX_COLS_i32 * sizeof(int32_t));
        tile_data.set_row_col(C11, MAX_ROWS, MAX_COLS_i32 * sizeof(int32_t));
        // tile 4,5 for A
        tile_data.set_row_col(A0, MAX_ROWS, MAX_COLS_i8 * sizeof(int8_t));
        tile_data.set_row_col(A1, MAX_ROWS, MAX_COLS_i8 * sizeof(int8_t));
        // tile 6,7 for B
        tile_data.set_row_col(B0, MAX_COLS_i8 / KPACK_b8, MAX_ROWS * KPACK_b8 * sizeof(int8_t));
        tile_data.set_row_col(B1, MAX_COLS_i8 / KPACK_b8, MAX_ROWS * KPACK_b8 * sizeof(int8_t));

        tile_data.set_config(); // Load tile configuration into hardware
    }

    // data relayout & packing buffers
    void prepare_packed_data() {
        if (params.packA) bufferA.pack();
        if (params.packB) bufferB.pack();
        if (params.packC) bufferC.pack();
    }

    void amx_gemm_compute(); // AMX GEMM compute function

    // write back packed C matrix to original layout
    void restore_packed_data() {
        if (params.packC) bufferC.unpack();
    }

    void cpu_gemm_ref(); // 3-nested loops with no amx
    // Top-level AMX GEMM function
    void amx_gemm() {
        amx_init();
        prepare_packed_data();
        amx_gemm_compute();
        restore_packed_data();
    }

    // print matrices for debugging
    void print_results() {
        print_matrix("A", A, M, K, lda);
        print_matrix("B", B, K, N, ldb);
        print_matrix("C", C, M, N, ldc);
    }


private:
    // Matrix parameters
    const int M, N, K;       // Matrix size
    const int lda, ldb, ldc; // Leading dimensions
    // data
    const int8_t* RESTRICT A;
    const int8_t* RESTRICT B;
    int32_t* RESTRICT C;

    GEMMParams params; // GEMM parameters


    ////////////////////////////////////////////////////////
    // Helper functions for tile operations
    ////////////////////////////////////////////////////////

    // tilezero 4 tileC
    static ALWAYS_INLINE void clear_4_tileC() {
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
    }

    // run 4 tdps
    static ALWAYS_INLINE void run_4_tdp() {
        _tile_dpbssd(C00, A0, B0);
        _tile_dpbssd(C01, A0, B1);
        _tile_dpbssd(C10, A1, B0);
        _tile_dpbssd(C11, A1, B1);
    }

    // L1 Cache Operations (Tile Load / Store)

    #define load_tileA_l1(dst, a_base, r, c, lda) \
        _tile_loadd(dst, &a_base[OFFSET2D(r, c, lda)], lda * sizeof(int8_t))
    #define load_tileB_l1(dst, b_base, r, c, ldb) \
        _tile_loadd(dst, &b_base[OFFSET2D(r/KPACK_b8, c*KPACK_b8, ldb*KPACK_b8)], ldb * KPACK_b8 * sizeof(int8_t))
    #define load_tileC_l1(dst, c_base, r, c, ldc) \
        _tile_loadd(dst, &c_base[OFFSET2D(r, c, ldc)], ldc * sizeof(int32_t))
    #define store_tileC_l1(src, c_base, r, c, ldc) \
        _tile_stored(src, &c_base[OFFSET2D(r, c, ldc)], ldc * sizeof(int32_t))

    // tileloadd A0 and A1
    static ALWAYS_INLINE void load_2_tileA_l1(const int8_t* a, int lda) {
        _tile_loadd(A0, a, lda * sizeof(int8_t));                  // Load A0
        _tile_loadd(A1, a + MAX_ROWS * lda, lda * sizeof(int8_t)); // Load A1
    }
    // dense tileloadd A0 and A1(for packed A)
    static ALWAYS_INLINE void load_2_tileA_l1(const int8_t* a) {
        _tile_loadd(A0, a, MIN_STRIDE);                // Load A0
        _tile_loadd(A1, a + TILE_SIZE_i8, MIN_STRIDE); // Load A1
    }
    // tileloadd B0 and B1
    static ALWAYS_INLINE void load_2_tileB_l1(const int8_t* b, int ldb) {
        _tile_loadd(B0, b, ldb * sizeof(int8_t));               // Load B0
        _tile_loadd(B1, b + MAX_COLS_i8, ldb * sizeof(int8_t)); // Load B1
    }
    // dense tileloadd B0 and B1(for packed B)
    static ALWAYS_INLINE void load_2_tileB_l1(const int8_t* b) {
        _tile_loadd(B0, b, MIN_STRIDE);                // Load B0
        _tile_loadd(B1, b + TILE_SIZE_i8, MIN_STRIDE); // Load B1
    }
    // tileloadd C0,C1,C2,C3
    static ALWAYS_INLINE void load_4_tileC_l1(const int32_t* c, int ldc) {
        _tile_loadd(C00, c, ldc * sizeof(int32_t));                         // Load C0
        _tile_loadd(C01, c + MAX_COLS_i32, ldc * sizeof(int32_t));          // Load C1
        _tile_loadd(C10, c + MAX_ROWS * ldc, ldc * sizeof(int32_t));        // Load C2
        _tile_loadd(C11, c + MAX_ROWS * ldc + MAX_COLS_i32, ldc * sizeof(int32_t)); // Load C3
    }
    // dense tileloadd C0,C1,C2,C3(for packed C)
    static ALWAYS_INLINE void load_4_tileC_l1(const int32_t* c) {
        _tile_loadd(C00, c, MIN_STRIDE);                         // Load C0
        _tile_loadd(C01, c + TILE_SIZE_i32, MIN_STRIDE);         // Load C1
        _tile_loadd(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE);     // Load C2
        _tile_loadd(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE);     // Load C3
    }
    // tilestored C0,C1,C2,C3
    static ALWAYS_INLINE void store_4_tileC_l1(int32_t* c, int ldc) {
        _tile_stored(C00, c, ldc * sizeof(int32_t));                         // Store C0
        _tile_stored(C01, c + MAX_COLS_i32, ldc * sizeof(int32_t));          // Store C1
        _tile_stored(C10, c + MAX_ROWS * ldc, ldc * sizeof(int32_t));        // Store C2
        _tile_stored(C11, c + MAX_ROWS * ldc + MAX_COLS_i32, ldc * sizeof(int32_t)); // Store C3
    }
    // dense tilestored C0,C1,C2,C3(for packed C)
    static ALWAYS_INLINE void store_4_tileC_l1(int32_t* c) {
        _tile_stored(C00, c, MIN_STRIDE);                     // Store C0
        _tile_stored(C01, c + TILE_SIZE_i32, MIN_STRIDE);     // Store C1
        _tile_stored(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE); // Store C2
        _tile_stored(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE); // Store C3
    }


    // L2 Cache Operations (Tile Load / Store)

    #define load_tileA_l2(dst, a_base, r, c, lda) \
        _tile_stream_loadd(dst, &a_base[OFFSET2D(r, c, lda)], lda * sizeof(int8_t))
    #define load_tileB_l2(dst, b_base, r, c, ldb) \
        _tile_stream_loadd(dst, &b_base[OFFSET2D(r/KPACK_b8, c*KPACK_b8, ldb*KPACK_b8)], ldb * KPACK_b8 * sizeof(int8_t))
    #define load_tileC_l2(dst, c_base, r, c, ldc) \
        _tile_stream_loadd(dst, &c_base[OFFSET2D(r, c, ldc)], ldc * sizeof(int32_t))

    // tileloaddt1 A0 and A1
    static ALWAYS_INLINE void load_2_tileA_l2(const int8_t* a, int lda) {
        _tile_stream_loadd(A0, a, lda * sizeof(int8_t));                  // Load A0
        _tile_stream_loadd(A1, a + MAX_ROWS * lda, lda * sizeof(int8_t)); // Load A1
    }
    // dense tileloaddt1 A0 and A1(for packed A)
    static ALWAYS_INLINE void load_2_tileA_l2(const int8_t* a) {
        _tile_stream_loadd(A0, a, MIN_STRIDE);                // Load A0
        _tile_stream_loadd(A1, a + TILE_SIZE_i8, MIN_STRIDE); // Load A1
    }
    // tileloaddt1 B0 and B1
    static ALWAYS_INLINE void load_2_tileB_l2(const int8_t* b, int ldb) {
        _tile_stream_loadd(B0, b, ldb * sizeof(int8_t));               // Load B0
        _tile_stream_loadd(B1, b + MAX_COLS_i8, ldb * sizeof(int8_t)); // Load B1
    }
    // dense tileloaddt1 B0 and B1(for packed B)
    static ALWAYS_INLINE void load_2_tileB_l2(const int8_t* b) {
        _tile_stream_loadd(B0, b, MIN_STRIDE);                // Load B0
        _tile_stream_loadd(B1, b + TILE_SIZE_i8, MIN_STRIDE); // Load B1
    }
    // tileloaddt1 C0,C1,C2,C3
    static ALWAYS_INLINE void load_4_tileC_l2(const int32_t* c, int ldc) {
        _tile_stream_loadd(C00, c, ldc * sizeof(int32_t));                     // Load C0
        _tile_stream_loadd(C01, c + MAX_COLS_i32, ldc * sizeof(int32_t));      // Load C1
        _tile_stream_loadd(C10, c + MAX_ROWS * ldc, ldc * sizeof(int32_t));    // Load C2
        _tile_stream_loadd(C11, c + MAX_ROWS * ldc + MAX_COLS_i32, ldc * sizeof(int32_t)); // Load C3
    }
    // dense tileloaddt1 C0,C1,C2,C3(for packed C)
    static ALWAYS_INLINE void load_4_tileC_l2(const int32_t* c) {
        _tile_stream_loadd(C00, c, MIN_STRIDE);                     // Load C0
        _tile_stream_loadd(C01, c + TILE_SIZE_i32, MIN_STRIDE);     // Load C1
        _tile_stream_loadd(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE); // Load C2
        _tile_stream_loadd(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE); // Load C3
    }

    // data relayout

    struct FreeDeleter {
        void operator()(void* p) const { std::free(p); }
    };

    // Buffer for packed matrix A
    struct BufferA {
        const GEMMKernelInt8* context = nullptr; // 指向外部class对象
        BufferA(const GEMMKernelInt8* ctx): context(ctx) {}
        void pack(); // pack A matrix

        ALWAYS_INLINE const int8_t* get() { return data.get(); }
        ALWAYS_INLINE const int8_t* get_block(int tk, int tm, int lda) {
            if (get() == nullptr) { // no pack
                return &context->A[OFFSET2D(tm, tk, lda)];
            } else {
                return get() + tk * context->M + tm * MIN(context->K - tk, TK);
            }
        }

    private:
        std::unique_ptr<int8_t[], FreeDeleter> data{nullptr};
    };

    // Buffer for packed matrix B
    struct BufferB {
        const GEMMKernelInt8* context = nullptr;
        BufferB(const GEMMKernelInt8* ctx): context(ctx) {}
        void pack(); // pack B matrix

        ALWAYS_INLINE const int8_t* get() { return data.get(); }
        ALWAYS_INLINE const int8_t *get_block(int tk, int tn, int ldb) {
            if (get() == nullptr) { // no pack
                return &context->B[OFFSET2D(tk, tn, ldb)];
            } else {
                return get() + tk * context->N + tn * MIN(context->K - tk, TK);
            }
        }

    private:
        std::unique_ptr<int8_t[], FreeDeleter> data{nullptr};
    };

    // Buffer for packed matrix C
    struct BufferC {
        const GEMMKernelInt8* context = nullptr;
        BufferC(const GEMMKernelInt8* ctx): context(ctx) {}
        void pack();   // pack C matrix
        void unpack(); // unpack C matrix

        ALWAYS_INLINE int32_t* get() { return data.get(); }
        ALWAYS_INLINE int32_t *get_block(int tm, int tn, int ldc) {
            if (get() == nullptr) { // no pack
                return &context->C[OFFSET2D(tm, tn, ldc)];
            } else {
                if (context->M >= context->N) {
                    return get() + tn * context->M + tm * MIN(context->N - tn, TN);
                } else {
                    return get() + tm * context->N + tn * MIN(context->M - tm, TM);
                }
            }
        }

    private:
        std::unique_ptr<int32_t[], FreeDeleter> data{nullptr};
    };

    BufferA bufferA; // Buffer for packed A
    BufferB bufferB; // Buffer for packed B
    BufferC bufferC; // Buffer for packed C
    // block size info for computation kernel
    struct taskSize {
        const int8_t *A, *B;
        int32_t *C;
        const int M, N, K;
    };

    taskSize get_full_task() {
        return taskSize{
            params.packA ? bufferA.get() : A,
            params.packB ? bufferB.get() : B,
            params.packC ? bufferC.get() : C,
            M, N, K
        };
    }

    // compute kernel
    void amx_gemm_naive(taskSize *task = nullptr);

    void amx_gemm_core(taskSize *task = nullptr);
    void amx_gemm_core_packB(taskSize *task = nullptr);
    void amx_gemm_core_packAB_v1(taskSize *task = nullptr);
    void amx_gemm_core_packAB_v2(taskSize *task = nullptr);
    void amx_gemm_core_packABC_v1(taskSize *task = nullptr);
    void amx_gemm_core_packABC_v2(taskSize *task = nullptr);
    void amx_gemm_core_experimental(taskSize *task = nullptr);

    void amx_gemm_blocking();

    template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
    void amx_gemm_core_packAB_v1_template(taskSize *task);
    template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
    void amx_gemm_core_packAB_v2_template(taskSize *task);
    template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
    void amx_gemm_core_packABC_v1_template(taskSize *task);
    template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
    void amx_gemm_core_packABC_v2_template(taskSize *task);

    // software prefetch manager
    struct SWPFetcher {
        bool on = true;
        size_t size;
        int step;
        const int8_t* ptr = nullptr;
        const _mm_hint HINT; // 预取级别

        size_t pfched_size = 0;

        SWPFetcher() = default;
        SWPFetcher(size_t size, int step, const int8_t* ptr, const _mm_hint HINT = _MM_HINT_T1)
            : size(size), step(step), ptr(ptr), HINT(HINT) {}

        ALWAYS_INLINE void prefetch() {
            if (!on || pfched_size >= size) [[unlikely]] return;

            #pragma GCC unroll 4
            for (int i = 0; i < step; i++) {
                _mm_prefetch(ptr, HINT);
                ptr += CACHELINE_SIZE;
            }
            pfched_size += step * CACHELINE_SIZE;
        }
    };


    // SWPWrapper: SWPF=true时，封装SWPFetcher；SWPF=false时，空实现
    template<bool Enable>
    struct SWPWrapper {
        // SWPF=false：完全空实现
        ALWAYS_INLINE void init(size_t, int, const int8_t*) {}
        ALWAYS_INLINE void init(size_t, int, const int8_t*, const _mm_hint) {}
        ALWAYS_INLINE void set_on(bool) {}
        ALWAYS_INLINE void prefetch() {}
        ALWAYS_INLINE const int8_t*& ptr() {
            static const int8_t* dummy = nullptr;
            return dummy;
        }
    };

};

// Thread parameters for multi-threaded GEMM
struct ThreadParams {
    std::vector<int> core_list = {0};
    bool numa_aware = false;
    int num_numa_node = 1;
};


// AMX GEMM Kernel for int8 on multi-threads
class GEMMKernelInt8MT {
public:
    GEMMKernelInt8MT(int M, int N, int K,
                      int lda, int ldb, int ldc,
                      const void* RESTRICT A,
                      const void* RESTRICT B,
                      void* RESTRICT C,
                      const ThreadParams& params = ThreadParams())
        : M(M), N(N), K(K),
          lda(lda), ldb(ldb), ldc(ldc),
          A(static_cast<const int8_t*>(A)),
          B(static_cast<const int8_t*>(B)),
          C(static_cast<int32_t*>(C)),
          params(params)
    {
        if (M % M_STEP != 0 || N % N_STEP != 0 || K % K_STEP != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of blocking sizes!\n";
            std::abort();
        }

        // set thread number
        if (this->params.numa_aware) {
            init_numa(this->params.num_numa_node, this->params.core_list);
        }
    }

    // cache blocking sizes
    static constexpr int TM = 512;
    static constexpr int TN = 512;
    static constexpr int TK = 1280;

    void init_kernels(); // initialize kernel instances per thread
    void prepare_packed_data();
    void amx_gemm_compute();
    void restore_packed_data();

    // Top-level AMX GEMM function
    void amx_gemm() {
        init_kernels();
        prepare_packed_data();
        amx_gemm_compute();
        restore_packed_data();
    }


private:
    // Matrix parameters
    const int M, N, K;       // Matrix size
    const int lda, ldb, ldc; // Leading dimensions
    // data
    const int8_t* RESTRICT A;
    const int8_t* RESTRICT B;
    int32_t* RESTRICT C;

    ThreadParams params; // Thread parameters


    // 数据重排和计算分离的接口,需要维护一个 Kernel 队列
    std::vector<std::unique_ptr<GEMMKernelInt8>> kernel_pool;
    // 内部 Worker 函数，每个线程执行这个函数
    void init_kernel_per_thread(int tid, int core_id);
    void prepare_packed_data_per_thread(int tid, int core_id);
    void amx_gemm_compute_per_thread(int tid, int core_id);
    void restore_packed_data_per_thread(int tid, int core_id);

};


} // namespace amx