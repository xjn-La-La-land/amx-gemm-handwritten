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

#include "utils.hpp"
#include "thread_params.hpp"

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

#include "buffer.hpp"   // Buffer / View / Role / Layout / make_view
#include "packing.hpp"  // pack_from / unpack_to + tile 原语 + tile 常量

namespace amx {

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18


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

// tile register alloc: 2A2B4C Tile Blocking
#define C00 0
#define C01 1
#define C10 2
#define C11 3
#define A0  4
#define A1  5
#define B0  6
#define B1  7
// register blocking 2A2B4C, micro-kernel shape is fixed = 32 x 32 x 64



// [Important!] We assume all matrices are in row-major order, as in C blas.
// LoopOrder: marco-kernel 的循环嵌套顺序。8 种矩阵形状(GEMM/GEPP/GEMP/GEPB/GEPM/GEBP/GEPDOT)
// 最终归纳到 3 种两级 loop 顺序, 每种以一个 Goto 叶子 micro-kernel 收尾。
enum class LoopOrder {
    KN_GEPB,    // for k { for n { GEPB   } }
    KM_GEBP,    // for k { for m { GEBP   } }
    MN_GEPDOT,  // for m { for n { GEPDOT } }
    NONE,
};
// loop 嵌套的可读描述, 供日志/调试打印
inline const char* loop_order_str(LoopOrder lo) {
    switch (lo) {
        case LoopOrder::KN_GEPB:   return "for k { for n { GEPB } }";
        case LoopOrder::KM_GEBP:   return "for k { for m { GEBP } }";
        case LoopOrder::MN_GEPDOT: return "for m { for n { GEPDOT } }";
    }
    return "unknown";
}


class GEMMKernelInt8; // 前置声明: GEMMParams::FuncPtr 需要它
using FuncPtr = void (GEMMKernelInt8::*)();

// parameter structure for GEMM kernel
struct GEMMParams {
    float alpha = 1.0f;
    float beta = 1.0f;
    // L2 / TLB / associativity 驱动的 cache blocking 大小
    int MC = 512;
    int NC = 512;
    int KC = 1280;
    // manual kernel selection
    FuncPtr kernel = nullptr;
    LoopOrder loop_order = LoopOrder::NONE;
};




// software prefetch helper for A, B, C
template <bool Strided>
struct SWPFHelper {
    bool on = true;
    const int8_t* ptr = nullptr;
private:
    size_t size = 0;
    int step = 0;
    _mm_hint HINT = _MM_HINT_T1; // 预取级别
    size_t _bytes = 0;
    size_t _row_bytes = 0;
    size_t bytes_per_row = 0;
    size_t stride_in_bytes = 0;

public:
    SWPFHelper() = default;
    // continuous (dense) data prefetch
    SWPFHelper(size_t size, int step, const _mm_hint HINT = _MM_HINT_T1)
        : size(size), step(step), HINT(HINT) {
        static_assert(!Strided, "dense constructor used for a strided SWPFHelper");
    }
    // strided data prefetch
    SWPFHelper(size_t size, int step, size_t bytes_per_row, size_t stride_in_bytes, const _mm_hint HINT = _MM_HINT_T1)
        : size(size), step(step), HINT(HINT), bytes_per_row(bytes_per_row), stride_in_bytes(stride_in_bytes) {
        static_assert(Strided, "strided constructor used for a dense SWPFHelper");
    }

    ALWAYS_INLINE void init(const int8_t* ptr) {
        this->ptr = ptr;
        _bytes = 0;
        if constexpr (Strided) _row_bytes = 0; // init 时 ptr 落在行首，行内计数归零
    }

    ALWAYS_INLINE void prefetch() {
        if (!ptr || !on || _bytes >= size) [[unlikely]] return;

        #pragma GCC unroll 4
        for (int i = 0; i < step; i++) {
            _mm_prefetch(ptr, HINT);
            _bytes += CACHELINE_SIZE;
            if constexpr (Strided) {
                _row_bytes += CACHELINE_SIZE;
                if (_row_bytes == bytes_per_row) {
                    ptr += stride_in_bytes - bytes_per_row; // move to the beginning of next row
                    _row_bytes = 0;
                } else {
                    ptr += CACHELINE_SIZE;
                }
            } else {
                ptr += CACHELINE_SIZE;
            }
        }
    }
};

// deduction guides: 由构造实参个数静态选出 Dense / Strided 特化
SWPFHelper(size_t, int) -> SWPFHelper<false>;
SWPFHelper(size_t, int, _mm_hint) -> SWPFHelper<false>;
SWPFHelper(size_t, int, int, int) -> SWPFHelper<true>;
SWPFHelper(size_t, int, int, int, _mm_hint) -> SWPFHelper<true>;




// AMX GEMM Kernel for int8
class GEMMKernelInt8 {
public:

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
          MC(params.MC),
          NC(params.NC),
          KC(params.KC)
    {
        if (M % MR != 0 || N % NR != 0 || K % KR != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }
        if (MC % MR != 0 || NC % NR != 0 || KC % KR != 0) {
            std::cerr << "[Error] Blocking sizes must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }

        select_kernel(); // heuristic tune
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


    void cpu_gemm_ref(); // 3-nested loops with no amx

    void prepare_packed_data();
    void restore_packed_data();
    void GEMM_compute(); // AMX GEMM compute function
    void GEMM();         // AMX GEMM with online packing/unpacking

    // print matrices for debugging
    void print_results() {
        print_matrix("A", A, M, K, lda);
        print_matrix("B", B, K, N, ldb);
        print_matrix("C", C, M, N, ldc);
    }

    // utils
    void alloc_buffers();
    void pack_A();
    void pack_B();
    void unpack_C();

    FuncPtr   selected_kernel_ptr() const { return selected_spec_->kernel; }
    LoopOrder loop_order()          const { return selected_spec_->loop_order; }
    const char* kernel_name()       const { return selected_spec_->kernel_name; }
    const char* loop_order_name()   const { return loop_order_str(selected_spec_->loop_order); }
    
    bool is_A_packed() const { return bufA.valid(); }
    bool is_B_packed() const { return bufB.valid(); }
    bool is_C_packed() const { return bufC.valid(); }

    struct KernelSpec {
        FuncPtr     kernel;      // 即 void (GEMMKernelInt8::*)()
        LoopOrder   loop_order;
        const char* kernel_name;
        bool        packA, packB, packC;
    };
    static const std::array<KernelSpec, 9>& kernel_specs();


private:
    // shape
    const int M, N, K;       // Matrix size
    const int lda, ldb, ldc; // Leading dimensions
    // data
    const int8_t* RESTRICT A;
    const int8_t* RESTRICT B;
    int32_t* RESTRICT C;

    const GEMMParams params; // GEMM parameters
    const int MC, NC, KC;    // Cache blocking sizes

    // 选中的 routine 行(kernel / loop_order / name / pack 全从这里派生), 指向 kernel_specs() 静态表。
    const KernelSpec* selected_spec_ = nullptr;

    // data relayout for whole matrix A, B, C
    Buffer<int8_t>  bufA;   // 纯存储;shape/stride/role 由 make_view 在使用点赋予
    Buffer<int8_t>  bufB;
    Buffer<int32_t> bufC;

    // shape check for GEMM
    void check_shape_health(const View<int8_t>& bufA,
                            const View<int8_t>& bufB,
                            const View<int32_t>& bufC)
    {
        if (bufA.cols() != bufB.rows() || bufA.rows() != bufC.rows() || bufB.cols() != bufC.cols()) {
            std::cerr << "[Error] Shape mismatch for GEMM! A: " << bufA.rows() << "x" << bufA.cols()
                      << ", B: " << bufB.rows() << "x" << bufB.cols()
                      << ", C: " << bufC.rows() << "x" << bufC.cols() << std::endl;
            std::abort();
        }
    }
    
    static constexpr int SCALE_FACTOR = 4;
    bool large_enough_m() {
        return M >= SCALE_FACTOR * MC;
    }
    bool large_enough_n() {
        return N >= SCALE_FACTOR * NC;
    }
    bool large_enough_k() {
        return K >= SCALE_FACTOR * KC;
    }

    void select_kernel();

    // compute kernels
    void GEMM_();
    void GEPP_();
    void GEMP_();
    void GEPM_();
    void GEPB_();
    void GEBP_();
    void GEPDOT_();

    void GEBP_kernel(View<int8_t> blockA, View<int8_t> panelB, View<int32_t> panelC, 
                     bool acc = true, const int8_t* B_kc_base = nullptr); 

    template <bool B_is_dense, bool C_is_dense>
    void GEBP_kernel_impl(View<int8_t> blockA, View<int8_t> panelB, View<int32_t> panelC, 
                          bool acc, const int8_t* B_kc_base);

    struct GEPBKernelConfig {
        bool fuse_packA = false;
        const int8_t* originA = nullptr;
        bool overwrite_C = false;

        uint8_t dense_ab_index() const {
            return (static_cast<uint8_t>(fuse_packA) << 1) |
                   static_cast<uint8_t>(overwrite_C);
        }

        uint8_t overwrite_c_index() const {
            return static_cast<uint8_t>(overwrite_C);
        }
    };

    void GEPB_kernel(View<int8_t> panelA, View<int8_t> blockB, View<int32_t> panelC, 
                     const GEPBKernelConfig& cfg);

    template <bool fuse_packA, bool overwrite_C>
    void GEPB_kernel_impl_denseABC(
        View<int8_t> panelA,
        View<int8_t> blockB, 
        View<int32_t> panelC,
        const int8_t* originA);

    template <bool fuse_packA, bool overwrite_C>
    void GEPB_kernel_impl_denseAB_stridedC(
        View<int8_t> panelA,
        View<int8_t> blockB, 
        View<int32_t> panelC,
        const int8_t* originA);

    template <bool overwrite_C>
    void GEPB_kernel_impl_denseBC_stridedA(
        View<int8_t> panelA, 
        View<int8_t> blockB, 
        View<int32_t> panelC);

    template <bool overwrite_C>
    void GEPB_kernel_impl_denseB_stridedAC(
        View<int8_t> panelA, 
        View<int8_t> blockB, 
        View<int32_t> panelC);
    
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

    // A0/A1 — dense(packed A,running 指针,stride=MIN_STRIDE)
    static ALWAYS_INLINE void load_2_tileA_l1(const int8_t* a) {
        _tile_loadd(A0, a, MIN_STRIDE);
        _tile_loadd(A1, a + TILE_SIZE_i8, MIN_STRIDE);
    }
    // A0/A1 — 通用(strided 或 dense):第二 tile 恒在 MAX_ROWS*row_bytes 处。供 View::tile 用。
    static ALWAYS_INLINE void load_2_tileA_l1(Tile<int8_t> t) {
        _tile_loadd(A0, t.data, t.row_bytes);
        _tile_loadd(A1, t.data + MAX_ROWS * t.row_bytes, t.row_bytes);
    }
    // B0/B1 — dense(packed B)
    static ALWAYS_INLINE void load_2_tileB_l1(const int8_t* b) {
        _tile_loadd(B0, b, MIN_STRIDE);
        _tile_loadd(B1, b + TILE_SIZE_i8, MIN_STRIDE);
    }
    static ALWAYS_INLINE void load_2_tileB_l1(Tile<int8_t> t) {
        _tile_loadd(B0, t.data, t.row_bytes);
        _tile_loadd(B1, t.data + MAX_ROWS * t.row_bytes, t.row_bytes);
    }
    // C00..C11 — dense(packed C,4 tile 连续码放)
    static ALWAYS_INLINE void load_4_tileC_l1(const int32_t* c) {
        _tile_loadd(C00, c, MIN_STRIDE);
        _tile_loadd(C01, c + TILE_SIZE_i32, MIN_STRIDE);
        _tile_loadd(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE);
        _tile_loadd(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE);
    }
    // C00..C11 — strided(原始 C 的 2x2 网格),byte-stride 来自 View::tile
    static ALWAYS_INLINE void load_4_tileC_l1(Tile<int32_t> t) {
        int32_t* c10 = reinterpret_cast<int32_t*>(
            reinterpret_cast<int8_t*>(t.data) + MAX_ROWS * t.row_bytes); // 下移 16 行
        _tile_loadd(C00, t.data,                t.row_bytes);
        _tile_loadd(C01, t.data + MAX_COLS_i32, t.row_bytes);            // 右移 16 列
        _tile_loadd(C10, c10,                   t.row_bytes);
        _tile_loadd(C11, c10 + MAX_COLS_i32,    t.row_bytes);
    }
    // store C00..C11 — dense(packed C)
    static ALWAYS_INLINE void store_4_tileC_l1(int32_t* c) {
        _tile_stored(C00, c, MIN_STRIDE);
        _tile_stored(C01, c + TILE_SIZE_i32, MIN_STRIDE);
        _tile_stored(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE);
        _tile_stored(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE);
    }
    // store C00..C11 — strided(2x2 网格)
    static ALWAYS_INLINE void store_4_tileC_l1(Tile<int32_t> t) {
        int32_t* c10 = reinterpret_cast<int32_t*>(
            reinterpret_cast<int8_t*>(t.data) + MAX_ROWS * t.row_bytes);
        _tile_stored(C00, t.data,                t.row_bytes);
        _tile_stored(C01, t.data + MAX_COLS_i32, t.row_bytes);
        _tile_stored(C10, c10,                   t.row_bytes);
        _tile_stored(C11, c10 + MAX_COLS_i32,    t.row_bytes);
    }


    // L2 Cache Operations (Tile Load / Store)

    #define load_tileA_l2(dst, a_base, r, c, lda) \
        _tile_stream_loadd(dst, &a_base[OFFSET2D(r, c, lda)], lda * sizeof(int8_t))
    #define load_tileB_l2(dst, b_base, r, c, ldb) \
        _tile_stream_loadd(dst, &b_base[OFFSET2D(r/KPACK_b8, c*KPACK_b8, ldb*KPACK_b8)], ldb * KPACK_b8 * sizeof(int8_t))
    #define load_tileC_l2(dst, c_base, r, c, ldc) \
        _tile_stream_loadd(dst, &c_base[OFFSET2D(r, c, ldc)], ldc * sizeof(int32_t))

    // A0/A1 — dense(stream,packed A)
    static ALWAYS_INLINE void load_2_tileA_l2(const int8_t* a) {
        _tile_stream_loadd(A0, a, MIN_STRIDE);
        _tile_stream_loadd(A1, a + TILE_SIZE_i8, MIN_STRIDE);
    }
    // A0/A1 — 通用(strided 或 dense,stream)
    static ALWAYS_INLINE void load_2_tileA_l2(Tile<int8_t> t) {
        _tile_stream_loadd(A0, t.data, t.row_bytes);
        _tile_stream_loadd(A1, t.data + MAX_ROWS * t.row_bytes, t.row_bytes);
    }
    // B0/B1 — dense(stream,packed B)
    static ALWAYS_INLINE void load_2_tileB_l2(const int8_t* b) {
        _tile_stream_loadd(B0, b, MIN_STRIDE);
        _tile_stream_loadd(B1, b + TILE_SIZE_i8, MIN_STRIDE);
    }
    static ALWAYS_INLINE void load_2_tileB_l2(Tile<int8_t> t) {
        _tile_stream_loadd(B0, t.data, t.row_bytes);
        _tile_stream_loadd(B1, t.data + MAX_ROWS * t.row_bytes, t.row_bytes);
    }
    // C00..C11 — dense(stream,packed C)
    static ALWAYS_INLINE void load_4_tileC_l2(const int32_t* c) {
        _tile_stream_loadd(C00, c, MIN_STRIDE);
        _tile_stream_loadd(C01, c + TILE_SIZE_i32, MIN_STRIDE);
        _tile_stream_loadd(C10, c + 2 * TILE_SIZE_i32, MIN_STRIDE);
        _tile_stream_loadd(C11, c + 3 * TILE_SIZE_i32, MIN_STRIDE);
    }
    // C00..C11 — strided(stream,2x2 网格)
    static ALWAYS_INLINE void load_4_tileC_l2(Tile<int32_t> t) {
        int32_t* c10 = reinterpret_cast<int32_t*>(
            reinterpret_cast<int8_t*>(t.data) + MAX_ROWS * t.row_bytes);
        _tile_stream_loadd(C00, t.data,                t.row_bytes);
        _tile_stream_loadd(C01, t.data + MAX_COLS_i32, t.row_bytes);
        _tile_stream_loadd(C10, c10,                   t.row_bytes);
        _tile_stream_loadd(C11, c10 + MAX_COLS_i32,    t.row_bytes);
    }

};




// Thread parameters for multi-threaded GEMM: 见 common/thread_params.hpp
// (与 bench_harness.hpp 共用同一定义)


// AMX GEMM Kernel for int8 on multi-threads
class GEMMKernelInt8MT {
public:

    GEMMKernelInt8MT(int M, int N, int K,
                      int lda, int ldb, int ldc,
                      const void* RESTRICT A,
                      const void* RESTRICT B,
                      void* RESTRICT C,
                      const ThreadParams& params = ThreadParams(),
                      const GEMMParams& gemm_params = GEMMParams())
        : M(M), N(N), K(K),
          lda(lda), ldb(ldb), ldc(ldc),
          A(static_cast<const int8_t*>(A)),
          B(static_cast<const int8_t*>(B)),
          C(static_cast<int32_t*>(C)),
          params(params),
          gemm_params(gemm_params),
          MC(gemm_params.MC),
          NC(gemm_params.NC),
          KC(gemm_params.KC)
    {
        if (M % MR != 0 || N % NR != 0 || K % KR != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }
        if (MC % MR != 0 || NC % NR != 0 || KC % KR != 0) {
            std::cerr << "[Error] Blocking sizes must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }

        // set thread number
        if (this->params.numa_aware) {
            init_numa(this->params.num_numa_node, this->params.core_list);
        }
    }

    void init_kernels(); // initialize kernel instances per thread
    void prepare_packed_data();
    void restore_packed_data();
    void GEMM_compute(); // AMX GEMM compute function

    void GEMM(); // Top-level AMX GEMM function with online packing/unpacking


private:
    // Matrix parameters
    const int M, N, K;       // Matrix size
    const int lda, ldb, ldc; // Leading dimensions
    // data
    const int8_t* RESTRICT A;
    const int8_t* RESTRICT B;
    int32_t* RESTRICT C;

    ThreadParams params; // Thread parameters
    GEMMParams gemm_params; // GEMM parameters forwarded to per-thread kernels
    const int MC, NC, KC; // Cache blocking sizes


    // 数据重排和计算分离的接口,需要维护一个 Kernel 队列
    std::vector<std::unique_ptr<GEMMKernelInt8>> kernel_pool;
    // 内部 Worker 函数，每个线程执行这个函数
    void init_kernel_per_thread(int tid, int core_id);

    void GEMM_per_thread(int tid, int core_id);
    void prepare_packed_data_per_thread(int tid, int core_id);
    void GEMM_compute_per_thread(int tid, int core_id);
    void restore_packed_data_per_thread(int tid, int core_id);

};


} // namespace amx
