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
static constexpr int MR = MAX_ROWS * 2;
static constexpr int NR = MAX_ROWS * 2;
static constexpr int KR = MAX_COLS_i8;


// L2 / TLB / associativity 驱动的 cache blocking 大小
struct BlockingConfig {
    static constexpr int DEFAULT_MC = 512;
    static constexpr int DEFAULT_NC = 512;
    static constexpr int DEFAULT_KC = 1280;

    int MC = DEFAULT_MC;
    int NC = DEFAULT_NC;
    int KC = DEFAULT_KC;

    BlockingConfig(int mc, int nc, int kc) : MC(mc), NC(nc), KC(kc) {}
    BlockingConfig() = default;
};

// parameter structure for GEMM kernel
struct GEMMParams {
    float alpha = 1.0f;
    float beta = 1.0f;
    // Data layout
    bool packA = true;
    bool packB = true;
    bool packC = true;
};

// [Important!] We assume all matrices are in row-major order, as in C blas.
enum class DataLayout { Strided, Dense };
enum class PackDirection { RowMajor, ColMajor };
// GEMM Kernel Routine
// Routine 1: GEMM => GEPP => GEPB (最终规约到 GEPB kernel)
// Routine 2: GEMM => GEPP => GEBP (最终规约到 GEBP kernel)
// Routine 3: GEMM => GEPM => GEPDOT (最终规约到 GEPDOT kernel)
enum class KernelRoutine {
    Routine1_GEPB,
    Routine2_GEBP,
    Routine3_GEPDOT,
};

// GEMM Buffers for data packing and relayout
template <typename T>
class Buffer {
public:
    Buffer() = default;
    // v1: just a view, does not own the data
    Buffer(T* data, int rows, int cols, int stride, 
           DataLayout layout = DataLayout::Strided)
        : data_(data), rows_(rows), cols_(cols), layout_(layout) {

        ownership_ = false;
        stride_ = layout == DataLayout::Strided ? stride : MIN_STRIDE / sizeof(T);
    }
    // v2: we allocate memory for packed data, and own the data
    Buffer(int rows, int cols, PackDirection dir = PackDirection::RowMajor) 
        : rows_(rows), cols_(cols), direction_(dir) {
        
        // alloc memory
        void *raw = std::aligned_alloc(CACHELINE_SIZE, rows * cols * sizeof(T));
        if (!raw) throw std::bad_alloc();
        data_ = static_cast<T*>(raw);

        layout_ = DataLayout::Dense;
        ownership_ = true;
    }

    virtual ~Buffer() { deallocate(); }

    void allocate_dense(int rows, int cols, PackDirection dir = PackDirection::RowMajor) {
        deallocate();
        void *raw = std::aligned_alloc(CACHELINE_SIZE, rows * cols * sizeof(T));
        if (!raw) throw std::bad_alloc();

        data_ = static_cast<T*>(raw);
        ownership_ = true;
        rows_ = rows;
        cols_ = cols;
        layout_ = DataLayout::Dense;
        direction_ = dir;
    }

    void release_owned_data() {
        deallocate();
        data_ = nullptr;
        ownership_ = false;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    bool valid() const { return data_ != nullptr; }

    inline int rows() const { return rows_; }
    inline int cols() const { return cols_; }
    inline int size() const { return rows_ * cols_; }
    inline size_t size_in_bytes() const { return rows_ * cols_ * sizeof(T); }
    inline int stride() const { return stride_; }
    inline DataLayout layout() const { return layout_; }
    inline PackDirection direction() const { return direction_; }
    inline bool owns_data() const { return ownership_; }

    inline bool is_dense() const { return layout_ == DataLayout::Dense; }
    inline bool is_strided() const { return layout_ == DataLayout::Strided; }

    // pointer arithmetic for strided layouts
    T* ptr(int r, int c) {
        assert(this->is_strided());
        return data_ + r * stride_ + c;
    }

    T& operator()(int r, int c) { return *ptr(r, c); }
    const T& operator()(int r, int c) const { return *ptr(r, c); }

    // pointer arithmetic for dense layouts
    virtual T* packed_ptr(int r, int c) = 0;

    inline void reset_rows(int rows) { rows_ = rows; }
    inline void reset_cols(int cols) { cols_ = cols; }

    virtual void pack_from(const T* src, int stride, int cache_blocking_sz = 0) = 0;
    virtual void unpack_to(T* dst, int stride, bool acc = true, int cache_blocking_sz = 0) const = 0;


protected:
    T* data_ = nullptr;
    bool ownership_ = false; // whether this buffer owns the data and is responsible for freeing it
    int rows_ = 0;
    int cols_ = 0;
    DataLayout layout_ = DataLayout::Dense;
    int stride_ = MIN_STRIDE / sizeof(T); 
    PackDirection direction_ = PackDirection::RowMajor;

    inline void deallocate() {
        if (ownership_ && data_) { std::free(data_); }
    }
};


// Buffer for A data
template <typename T>
class BufferA : public Buffer<T> {
public:
    using Buffer<T>::Buffer;
    void pack_from(const T* src, int stride, int cache_blocking_sz = 0) override;
    void unpack_to(T*, int, bool = true, int = 0) const override {}

    T* packed_ptr(int r, int c) override {
        assert(this->is_dense());
        assert(r % MR == 0 && c % KR == 0);
        if (this->direction() == PackDirection::RowMajor) {
            return this->data_ + r * this->cols_ + c * MR;
        } else {
            return this->data_ + c * this->rows_ + r * KR;
        }
    }
};

// Buffer for B data
template <typename T>
class BufferB : public Buffer<T> {
public:
    using Buffer<T>::Buffer;
    void pack_from(const T* src, int stride, int cache_blocking_sz = 0) override;
    void unpack_to(T*, int, bool = true, int = 0) const override {}

    T* packed_ptr(int r, int c) override {
        assert(this->is_dense());
        assert(r % KR == 0 && c % NR == 0);
        if (this->direction() == PackDirection::RowMajor) {
            return this->data_ + r * this->cols_ + c * KR;
        } else {
            return this->data_ + c * this->rows_ + r * NR;
        }
    }
};

// Buffer for C data
template <typename T>
class BufferC : public Buffer<T> {
public:
    using Buffer<T>::Buffer;
    void pack_from(const T*, int, int = 0) override {}
    // origin C(strided) += BufferC(packed) if acc=true, otherwise overwrite
    void unpack_to(T* dst, int stride, bool acc = true, int cache_blocking_sz = 0) const override;

    T* packed_ptr(int r, int c) override {
        assert(this->is_dense());
        assert(r % MR == 0 && c % NR == 0);
        if (this->direction() == PackDirection::RowMajor) {
            return this->data_ + r * this->cols_ + c * MR;
        } else {
            return this->data_ + c * this->rows_ + r * NR;
        }
    }

    T* next_tile_ptr(int r, int c) {
        assert(this->is_strided());
        int next_r = c + NR < this->cols_ ? r : r + MR;
        int next_c = c + NR < this->cols_ ? c + NR : 0;
        return this->ptr(next_r, next_c);
    }
};


// software prefetch helper for A, B, C
struct SWPFHelper {
    bool on = true;
    const int8_t* ptr = nullptr;
private:
    size_t size = 0;
    int step = 0;
    _mm_hint HINT = _MM_HINT_T1; // 预取级别
    size_t _bytes = 0;
    int bytes_per_row = 0, stride_in_bytes = 0;

public:
    SWPFHelper() = default;
    // continuous data prefetch
    SWPFHelper(size_t size, int step, const _mm_hint HINT = _MM_HINT_T1)
        : size(size), step(step), HINT(HINT) {}
    // strided data prefetch
    SWPFHelper(size_t size, int step, int bytes_per_row, int stride_in_bytes, const _mm_hint HINT = _MM_HINT_T1)
        : size(size), step(step), HINT(HINT), bytes_per_row(bytes_per_row), stride_in_bytes(stride_in_bytes) {}

    ALWAYS_INLINE void init(const int8_t* ptr) {
        this->ptr = ptr;
        _bytes = 0;
    }

    ALWAYS_INLINE void prefetch() {
        if (!ptr || !on || _bytes >= size) [[unlikely]] return;

        #pragma GCC unroll 4
        for (int i = 0; i < step; i++) {
            _mm_prefetch(ptr, HINT);
            _bytes += CACHELINE_SIZE;
            if (stride_in_bytes > 0 && _bytes % bytes_per_row == 0) {
                ptr += stride_in_bytes - bytes_per_row; // move to the beginning of next row
            } else {
                ptr += CACHELINE_SIZE;
            }
        }
    }
};




// AMX GEMM Kernel for int8
class GEMMKernelInt8 {
public:

    GEMMKernelInt8(int M, int N, int K,
                   int lda, int ldb, int ldc,
                   const void* RESTRICT A,
                   const void* RESTRICT B,
                   void* RESTRICT C,
                   const GEMMParams& params = GEMMParams(),
                   const BlockingConfig& blocking = BlockingConfig())
        : M(M), N(N), K(K),
          lda(lda), ldb(ldb), ldc(ldc),
          A(static_cast<const int8_t*>(A)),
          B(static_cast<const int8_t*>(B)),
          C(static_cast<int32_t*>(C)),
          params(params),
          MC(round_down(blocking.MC, MR)),
          NC(round_down(blocking.NC, NR)),
          KC(round_down(blocking.KC, KR))
    {
        if (M % MR != 0 || N % NR != 0 || K % KR != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }
        if (blocking.MC < MR || blocking.NC < NR || blocking.KC < KR) {
            std::cerr << "[Error] Blocking sizes must be at least " << MR << "x" << NR << "x" << KR << "!\n";
            std::abort();
        }

        find_best_routine();
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


private:
    // shape
    const int M, N, K;       // Matrix size
    const int lda, ldb, ldc; // Leading dimensions
    // data
    const int8_t* RESTRICT A;
    const int8_t* RESTRICT B;
    int32_t* RESTRICT C;

    const GEMMParams params; // GEMM parameters
    const int MC, NC, KC; // Cache blocking sizes
    KernelRoutine routine; // GEMM routine to use

    // data relayout for whole matrix A, B, C
    BufferA<int8_t> bufA;
    BufferB<int8_t> bufB;
    BufferC<int32_t> bufC;
    // packing utilities
    void pack_matrix_a_3d(int8_t* dst, PackDirection dir);
    void pack_matrix_b_3d(int8_t* dst, PackDirection dir);
    void unpack_matrix_c_3d(const int32_t* src, PackDirection dir);

    // shape check for GEMM
    void check_shape_health(const Buffer<int8_t>& bufA, 
                            const Buffer<int8_t>& bufB, 
                            const Buffer<int32_t>& bufC)
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

    void find_best_routine();

    // compute kernels
    void GEMM_();
    void GEPP_();
    void GEMP_();
    void GEPM_();
    void GEPB_();
    void GEBP_();
    void GEPDOT_();
    using FuncPtr = void (GEMMKernelInt8::*)();
    FuncPtr compute_func = nullptr;

    void GEBP_kernel(BufferA<int8_t>& blockA, BufferB<int8_t>& panelB, BufferC<int32_t>& panelC, 
                     bool acc = true, const int8_t* B_kc_base = nullptr); 

    template <bool B_is_dense, bool C_is_dense>
    void GEBP_kernel_impl(BufferA<int8_t>& blockA, BufferB<int8_t>& panelB, BufferC<int32_t>& panelC, 
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

    void GEPB_kernel(BufferA<int8_t>& panelA, BufferB<int8_t>& blockB, BufferC<int32_t>& panelC, 
                     const GEPBKernelConfig& cfg);

    template <bool fuse_packA, bool overwrite_C>
    void GEPB_kernel_impl_denseABC(
        BufferA<int8_t>& panelA,
        BufferB<int8_t>& blockB, 
        BufferC<int32_t>& panelC,
        const int8_t* originA);

    template <bool fuse_packA, bool overwrite_C>
    void GEPB_kernel_impl_denseAB_stridedC(
        BufferA<int8_t>& panelA,
        BufferB<int8_t>& blockB, 
        BufferC<int32_t>& panelC,
        const int8_t* originA);

    template <bool overwrite_C>
    void GEPB_kernel_impl_denseBC_stridedA(
        BufferA<int8_t>& panelA, 
        BufferB<int8_t>& blockB, 
        BufferC<int32_t>& panelC);

    template <bool overwrite_C>
    void GEPB_kernel_impl_denseB_stridedAC(
        BufferA<int8_t>& panelA, 
        BufferB<int8_t>& blockB, 
        BufferC<int32_t>& panelC);
    
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
                      const GEMMParams& gemm_params = GEMMParams(),
                      const BlockingConfig& blocking = BlockingConfig())
        : M(M), N(N), K(K),
          lda(lda), ldb(ldb), ldc(ldc),
          A(static_cast<const int8_t*>(A)),
          B(static_cast<const int8_t*>(B)),
          C(static_cast<int32_t*>(C)),
          params(params),
          gemm_params(gemm_params),
          MC(round_down(blocking.MC, MR)),
          NC(round_down(blocking.NC, NR)),
          KC(round_down(blocking.KC, KR))
    {
        if (M % MR != 0 || N % NR != 0 || K % KR != 0) {
            std::cerr << "[Error] Matrix dimensions must be multiples of micro-kernel sizes(" << MR << "x" << NR << "x" << KR << ")!\n";
            std::abort();
        }
        if (blocking.MC < MR || blocking.NC < NR || blocking.KC < KR) {
            std::cerr << "[Error] Blocking sizes must be at least " << MR << "x" << NR << "x" << KR << "!\n";
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

    // Top-level AMX GEMM function
    void GEMM();


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
