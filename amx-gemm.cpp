#include "amx-gemm.hpp"

namespace amx {
using Kernel = GEMMKernelInt8;
using KernelMT = GEMMKernelInt8MT;

void pack_tile_a(const int8_t* src, int8_t* dst, int lda) {
    #pragma GCC unroll 16
    for (int r = 0; r < MAX_ROWS; ++r) {
        __m512i v = _mm512_loadu_si512(src); // loadu (unaligned) 因为源地址 lda 不一定 64B 对齐
        _mm512_store_si512(dst, v); // packed 内存已 64B 对齐
        src += lda;
        dst += MIN_STRIDE;
    }
}


// change the data layout of A matrix to enable dense tileload
void Kernel::BufferA::pack() {
    // allocate packed A buffer
    void* raw_ptr = std::aligned_alloc(CACHELINE_SIZE, context->M * context->K * sizeof(int8_t));
    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    data.reset(static_cast<int8_t*>(raw_ptr));

    // pack A matrix
    for (int tk = 0; tk < context->K; tk += TK) {
        const int8_t *src = &context->A[OFFSET2D(0, tk, context->lda)];
        int8_t *dst = data.get() + tk * context->M;

        for (int m = 0; m < context->M; m += M_STEP) {
            for (int k = 0; k < MIN(TK, context->K - tk); k += K_STEP) {
                // pack 2 tiles of A
                pack_tile_a(&src[OFFSET2D(m, k, context->lda)], dst, context->lda);
                dst += TILE_SIZE_i8;
                pack_tile_a(&src[OFFSET2D(m + MAX_ROWS, k, context->lda)], dst, context->lda);
                dst += TILE_SIZE_i8;
            }
        }
    }
}


/**
 * @brief 将 4 行 B 矩阵数据（每行 16 个 int8）混洗成 VNNI 格式
 * * 输入：4 行，每行 16 个元素 (N=16)
 * Row 0: a0 a1 ... a15
 * Row 1: b0 b1 ... b15
 * Row 2: c0 c1 ... c15
 * Row 3: d0 d1 ... d15
 * * 输出：1 行 ZMM (64 字节)，内容为 VNNI 格式
 * {a0 b0 c0 d0}, {a1 b1 c1 d1}, ... {a15 b15 c15 d15}
 */
ALWAYS_INLINE void pack_rows_4_vnni(const int8_t* src, int8_t* dst, int ldb) {
    // 1. 加载 4 行数据 (每行 16 字节 -> XMM 寄存器)
    // 使用 loadu 因为源地址可能不是 16 字节对齐
    __m128i r0 = _mm_loadu_si128((const __m128i*)(src));
    __m128i r1 = _mm_loadu_si128((const __m128i*)(src + ldb));
    __m128i r2 = _mm_loadu_si128((const __m128i*)(src + 2 * ldb));
    __m128i r3 = _mm_loadu_si128((const __m128i*)(src + 3 * ldb));

    // 2. 第一轮交织 (Interleave 8-bit to 16-bit)
    // unpckl: a0 b0 a1 b1 ... a7 b7
    // unpckh: a8 b8 a9 b9 ... a15 b15
    __m128i t0 = _mm_unpacklo_epi8(r0, r1); 
    __m128i t1 = _mm_unpackhi_epi8(r0, r1);
    __m128i t2 = _mm_unpacklo_epi8(r2, r3);
    __m128i t3 = _mm_unpackhi_epi8(r2, r3);

    // 3. 第二轮交织 (Interleave 16-bit to 32-bit) -> 形成最终 VNNI 格式
    // 此时 t0: a0 b0 ...   t2: c0 d0 ...
    // 结果: a0 b0 c0 d0, a1 b1 c1 d1 ...
    __m128i v0 = _mm_unpacklo_epi16(t0, t2); // 前 4 列
    __m128i v1 = _mm_unpackhi_epi16(t0, t2); // 次 4 列
    __m128i v2 = _mm_unpacklo_epi16(t1, t3); // 次 4 列
    __m128i v3 = _mm_unpackhi_epi16(t1, t3); // 后 4 列

    // 4. 将 4 个 XMM 拼成一个 ZMM 并存储
    // 我们凑齐了 64 字节，可以直接用一条 AVX-512 指令存入 Packed Buffer
    // 这里的 insert 逻辑是将 4 个 128位 插入到 512位 寄存器中
    __m512i zmm_out = _mm512_castsi128_si512(v0);
    zmm_out = _mm512_inserti32x4(zmm_out, v1, 1);
    zmm_out = _mm512_inserti32x4(zmm_out, v2, 2);
    zmm_out = _mm512_inserti32x4(zmm_out, v3, 3);

    // 存储到对齐的 Packed 内存
    _mm512_store_si512(dst, zmm_out);
}

void pack_tile_b(const int8_t* src, int8_t* dst, int ldb) {
    for (int r = 0; r < MAX_ROWS; r++) {
        pack_rows_4_vnni(src, dst, ldb); // 4rows * 16B => 64B VNNI
        src += KPACK_b8 * ldb;
        dst += MIN_STRIDE;
    }
}


// change the data layout of B matrix to enable dense tileload
void Kernel::BufferB::pack() {
    // allocate packed B buffer
    void* raw_ptr = std::aligned_alloc(CACHELINE_SIZE, context->N * context->K * sizeof(int8_t));
    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    data.reset(static_cast<int8_t*>(raw_ptr));

    // pack B matrix
    for (int tk = 0; tk < context->K; tk += TK) {
        const int8_t *src = &context->B[OFFSET2D(tk, 0, context->ldb)];
        int8_t *dst = data.get() + tk * context->N;

        for (int n = 0; n < context->N; n += N_STEP) {
            for (int k = 0; k < MIN(TK, context->K - tk); k += K_STEP) {
                // pack 2 tiles of B
                pack_tile_b(&src[OFFSET2D(k, n, context->ldb)], dst, context->ldb);
                dst += TILE_SIZE_i8;
                pack_tile_b(&src[OFFSET2D(k, n + MAX_ROWS, context->ldb)], dst, context->ldb);
                dst += TILE_SIZE_i8;
            }
        }
    }
}


void pack_tile_c(const int32_t* src, int32_t* dst, int ldc) {
    #pragma GCC unroll 16
    for (int r = 0; r < MAX_ROWS; ++r) {
        __m512i v = _mm512_loadu_si512(src);
        _mm512_store_si512(dst, v);
        src += ldc;
        dst += MAX_COLS_i32;
    }
}


// change the data layout of C matrix to enable dense tilestore
void Kernel::BufferC::pack() {
    // allocate packed C buffer
    void* raw_ptr = std::aligned_alloc(CACHELINE_SIZE, context->M * context->N * sizeof(int32_t));
    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    data.reset(static_cast<int32_t*>(raw_ptr));

    // pack C matrix
    if (context->M >= context->N) {
        for (int tn = 0; tn < context->N; tn += TN) { // split on N dimension to have bigger M/TN
            const int32_t *src = &context->C[OFFSET2D(0, tn, context->ldc)];
            int32_t *dst = data.get() + tn * context->M;
            for (int m = 0; m < context->M; m += M_STEP) {
                for (int n = 0; n < MIN(TN, context->N - tn); n += N_STEP) {
                    // pack 4 tiles of C
                    pack_tile_c(&src[OFFSET2D(m, n, context->ldc)], dst, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m, n + MAX_ROWS, context->ldc)], dst + TILE_SIZE_i32, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m + MAX_ROWS, n, context->ldc)], dst + 2 * TILE_SIZE_i32, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m + MAX_ROWS, n + MAX_ROWS, context->ldc)], dst + 3 * TILE_SIZE_i32, context->ldc);
                    dst += 4 * TILE_SIZE_i32;
                }
            }
        }
    }
    else {
        for (int tm = 0; tm < context->M; tm += TM) { // split on M dimension
            const int32_t *src = &context->C[OFFSET2D(tm, 0, context->ldc)];
            int32_t *dst = data.get() + tm * context->N;

            for (int n = 0; n < context->N; n += N_STEP) {
                for (int m = 0; m < MIN(TM, context->M - tm); m += M_STEP) {
                    // pack 4 tiles of C
                    pack_tile_c(&src[OFFSET2D(m, n, context->ldc)], dst, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m, n + MAX_ROWS, context->ldc)], dst + TILE_SIZE_i32, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m + MAX_ROWS, n, context->ldc)], dst + 2 * TILE_SIZE_i32, context->ldc);
                    pack_tile_c(&src[OFFSET2D(m + MAX_ROWS, n + MAX_ROWS, context->ldc)], dst + 3 * TILE_SIZE_i32, context->ldc);
                    dst += 4 * TILE_SIZE_i32;
                }
            }

        }
    }
}


// restore C matrix from packed layout
void Kernel::BufferC::unpack() {
    // TODO: implement unpack
}



///////////////////////////////////////////////////////
// AMX GEMM compute kernel
///////////////////////////////////////////////////////

void Kernel::cpu_gemm_ref() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = C[OFFSET2D(i, j, ldc)];
            for (int k = 0; k < K; k++) 
                sum += A[OFFSET2D(i, k, lda)] * B[OFFSET2D(k, j, ldb)];
            C[OFFSET2D(i, j, ldc)] = sum;
        }
    }
}

// dummy amx implementation
void Kernel::amx_gemm_naive(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    for (int i = 0; i < task->M; i += MAX_ROWS) {
        for (int j = 0; j < task->N; j += MAX_ROWS) {
            _tile_stream_loadd(0, &task->C[OFFSET2D(i, j, ldc)], ldc * sizeof(int32_t));
            for (int k = 0; k < task->K; k += MAX_COLS_i8) {
                _tile_loadd(1, &task->A[OFFSET2D(i, k, lda)], lda * sizeof(int8_t));
                _tile_loadd(2, &task->B[OFFSET2D(k, j, ldb)], ldb * sizeof(int8_t));
                _tile_dpbusd(0, 1, 2);
            }
            _tile_stored(0, &task->C[OFFSET2D(i, j, ldc)], ldc * sizeof(int32_t));
        }
    }
}


/// @brief AMX GEMM Core **2A2B4C** Tiling with origin data layout
///
/// **Cache Policy**:
/// - Stream `A`, `C` into `L1D`
/// - Stream `B` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    for (int i = 0; i < task->M; i += M_STEP) {
        for (int j = 0; j < task->N; j += N_STEP) {
            load_4_tileC_l2(&task->C[OFFSET2D(i, j, ldc)], ldc);
            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                // load_2_tileA_l1(&task->A[OFFSET2D(i, k, lda)], lda);
                // load_2_tileB_l2(&task->B[OFFSET2D(k, j, ldb)], ldb);
                load_tileB_l2(B0, task->B, k, j, ldb);
                load_tileA_l1(A0, task->A, i, k, lda);
                load_tileB_l2(B1, task->B, k, j + MAX_ROWS, ldb);
                load_tileA_l1(A1, task->A, i + MAX_ROWS, k, lda);
                run_4_tdp();
            }
            store_4_tileC_l1(&task->C[OFFSET2D(i, j, ldc)], ldc);
        }
    }
}


/// @brief AMX GEMM Core **2A2B4C** Tiling with packed B
///
/// **Cache Policy**:
/// - Stream `A`, `C` into `L1D`
/// - Stream `B` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core_packB(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    const int8_t *B_ptr;
    for (int i = 0; i < task->M; i += M_STEP) {
        B_ptr = task->B; // packed B block address
        for (int j = 0; j < task->N; j += N_STEP) {
            load_4_tileC_l2(&task->C[OFFSET2D(i, j, ldc)], ldc);
            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                load_2_tileA_l1(&task->A[OFFSET2D(i, k, lda)], lda);
                load_2_tileB_l2(B_ptr);
                run_4_tdp();
                B_ptr += 2 * TILE_SIZE_i8;
            }
            store_4_tileC_l1(&task->C[OFFSET2D(i, j, ldc)], ldc);
        }
    }
}


template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
void Kernel::amx_gemm_core_packAB_v1_template(taskSize *task) {
    const int8_t *A_ptr, *B_ptr;

    using FetcherA_t = std::conditional_t<SWPF_A, SWPFetcher, NoOpFetcher>;
    using FetcherB_t = std::conditional_t<SWPF_B, SWPFetcher, NoOpFetcher>;
    using FetcherC_t = std::conditional_t<SWPF_C, SWPFetcher, NoOpFetcher>;
    FetcherA_t swpfetcher_A;
    FetcherB_t swpfetcher_B;
    FetcherC_t swpfetcher_C;

    if constexpr (SWPF_B) {
        const int8_t *next_B_ptr = task->B + task->N * task->K; // prefetch next TN×TK blockB
        swpfetcher_B = FetcherB_t(task->N * task->K, 1, next_B_ptr);
    }
    for (int i = 0; i < task->M; i += M_STEP) {
        B_ptr = task->B; // packed B block address

        if constexpr (SWPF_A) {
            const int8_t *next_A_ptr = task->A + (i + M_STEP) * task->K; // prefetch next 32×TK blockA
            swpfetcher_A = FetcherA_t(M_STEP * task->K, 2, next_A_ptr);
            swpfetcher_A.on = (i + M_STEP < task->M);
        }
        for (int j = 0; j < task->N; j += N_STEP) {
            A_ptr = task->A + i * task->K; // packed A block address
            load_4_tileC_l2(&task->C[OFFSET2D(i, j, ldc)], ldc);

            if constexpr (SWPF_C) {
                int next_i = (j + N_STEP == task->N)? i + M_STEP : i;
                int next_j = (j + N_STEP == task->N)? 0 : j + N_STEP;
                const int8_t *next_C_ptr = reinterpret_cast<const int8_t*>(&task->C[OFFSET2D(next_i, next_j, ldc)]);
                swpfetcher_C = FetcherC_t(M_STEP * N_STEP * sizeof(int32_t), 2, next_C_ptr);
                swpfetcher_C.on = (next_i < task->M);
            }
            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                _tile_stream_loadd(6, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B0
                _tile_stream_loadd(7, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B1
                _tile_loadd(4, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A0
                _tile_loadd(5, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A1
                run_4_tdp();

                if constexpr (SWPF_A) swpfetcher_A.prefetch();
                if constexpr (SWPF_B) swpfetcher_B.prefetch();
                if constexpr (SWPF_C) {
                    swpfetcher_C.prefetch();
                    swpfetcher_C.ptr += (ldc * sizeof(int32_t) - 2 * CACHELINE_SIZE); // move to next row in C
                    swpfetcher_C.prefetch();
                    swpfetcher_C.ptr += (ldc * sizeof(int32_t) - 2 * CACHELINE_SIZE);
                }
            }
            store_4_tileC_l1(&task->C[OFFSET2D(i, j, ldc)], ldc);
        }
    }
}


/// @brief AMX GEMM Core **2A2B4C** Tiling with packed A & B
///
/// **Cache Policy**:
/// - Stream `A`, `C` into `L1D`
/// - Stream `B` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core_packAB_v1(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    int swpf_choice = params.swpfA << 2 | params.swpfB << 1 | params.swpfC;
    switch (swpf_choice) {
        case 0b000:
            amx_gemm_core_packAB_v1_template<false, false, false>(task);
            break;
        case 0b100:
            amx_gemm_core_packAB_v1_template<true, false, false>(task);
            break;
        case 0b101:
            amx_gemm_core_packAB_v1_template<true, false, true>(task);
            break;
        case 0b111:
            amx_gemm_core_packAB_v1_template<true, true, true>(task);
            break;
        default:
            throw std::runtime_error("Unsupported SWPF configuration!");
    }
}


template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
void Kernel::amx_gemm_core_packAB_v2_template(taskSize *task) {
    const int8_t *A_ptr, *B_ptr;

    using FetcherA_t = std::conditional_t<SWPF_A, SWPFetcher, NoOpFetcher>;
    using FetcherB_t = std::conditional_t<SWPF_B, SWPFetcher, NoOpFetcher>;
    using FetcherC_t = std::conditional_t<SWPF_C, SWPFetcher, NoOpFetcher>;
    FetcherA_t swpfetcher_A;
    FetcherB_t swpfetcher_B;
    FetcherC_t swpfetcher_C;

    if constexpr (SWPF_A) {
        const int8_t *next_A_ptr = task->A + task->M * task->K; // prefetch next TM×TK blockA
        swpfetcher_A = FetcherA_t(task->M * task->K, 1, next_A_ptr);
    }
    for (int j = 0; j < task->N; j += N_STEP) {
        A_ptr = task->A;

        if constexpr (SWPF_B) {
            const int8_t *next_B_ptr = task->B + (j + N_STEP) * task->K; // prefetch next 32×TK blockB
            swpfetcher_B = FetcherB_t(N_STEP * task->K, 2, next_B_ptr);
            swpfetcher_B.on = (j + N_STEP < task->N);
        }
        for (int i = 0; i < task->M; i += M_STEP) {
            B_ptr = task->B + j * task->K;
            load_4_tileC_l2(&task->C[OFFSET2D(i, j, ldc)], ldc);

            if constexpr (SWPF_C) {
                int next_i = (i + M_STEP == task->M)? 0 : i + M_STEP;
                int next_j = (i + M_STEP == task->M)? (j + N_STEP) : j;
                const int8_t *next_C_ptr = reinterpret_cast<const int8_t*>(&task->C[OFFSET2D(next_i, next_j, ldc)]);
                swpfetcher_C = FetcherC_t(M_STEP * N_STEP * sizeof(int32_t), 2, next_C_ptr);
                swpfetcher_C.on = (next_j < task->N);
            }
            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                _tile_stream_loadd(6, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B0
                _tile_stream_loadd(7, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B1
                _tile_loadd(4, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A0
                _tile_loadd(5, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A1
                run_4_tdp();

                if constexpr (SWPF_B) swpfetcher_B.prefetch();
                if constexpr (SWPF_A) swpfetcher_A.prefetch();
                if constexpr (SWPF_C) {
                    swpfetcher_C.prefetch();
                    swpfetcher_C.ptr += (ldc * sizeof(int32_t) - 2 * CACHELINE_SIZE); // move to next row in C
                    swpfetcher_C.prefetch();
                    swpfetcher_C.ptr += (ldc * sizeof(int32_t) - 2 * CACHELINE_SIZE);
                }
            }
            store_4_tileC_l1(&task->C[OFFSET2D(i, j, ldc)], ldc);
        }
    }
}


/// @brief AMX GEMM Core **2A2B4C** Tiling with packed A & B
///
/// **Cache Policy**:
/// - Stream `B`, `C` into `L1D`
/// - Stream `A` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core_packAB_v2(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    int swpf_choice = params.swpfA << 2 | params.swpfB << 1 | params.swpfC;
    switch (swpf_choice) {
        case 0b000:
            amx_gemm_core_packAB_v2_template<false, false, false>(task);
            break;
        case 0b010:
            amx_gemm_core_packAB_v2_template<false, true, false>(task);
            break;
        case 0b011:
            amx_gemm_core_packAB_v2_template<false, true, true>(task);
            break;
        case 0b111:
            amx_gemm_core_packAB_v2_template<true, true, true>(task);
            break;
        default:
            throw std::runtime_error("Unsupported SWPF configuration!");
    }
}


template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
void Kernel::amx_gemm_core_packABC_v1_template(taskSize *task) {
    const int8_t *A_ptr, *B_ptr;
    int32_t *C_ptr = task->C;

    using FetcherA_t = std::conditional_t<SWPF_A, SWPFetcher, NoOpFetcher>;
    using FetcherB_t = std::conditional_t<SWPF_B, SWPFetcher, NoOpFetcher>;
    using FetcherC_t = std::conditional_t<SWPF_C, SWPFetcher, NoOpFetcher>;
    FetcherA_t swpfetcher_A;
    FetcherB_t swpfetcher_B;
    FetcherC_t swpfetcher_C;

    if constexpr (SWPF_B) {
        const int8_t *next_B_ptr = task->B + task->N * task->K; // prefetch next TN×TK blockB
        swpfetcher_B = FetcherB_t(task->N * task->K, 1, next_B_ptr);
    }
    for (int i = 0; i < task->M; i += M_STEP) {
        B_ptr = task->B;

        if constexpr (SWPF_A) {
            const int8_t *next_A_ptr = task->A + (i + M_STEP) * task->K; // prefetch next 32×TK blockA
            swpfetcher_A = FetcherA_t(M_STEP * task->K, 2, next_A_ptr);
            swpfetcher_A.on = (i + M_STEP < task->M);
        }
        for (int j = 0; j < task->N; j += N_STEP) {
            A_ptr = task->A + i * task->K;
            load_4_tileC_l2(C_ptr);

            if constexpr (SWPF_C) {
                const int8_t *next_C_ptr = reinterpret_cast<const int8_t*>(C_ptr + M_STEP * N_STEP);
                swpfetcher_C = FetcherC_t(M_STEP * N_STEP * sizeof(int32_t), 4, next_C_ptr);
                swpfetcher_C.on = !((i + M_STEP == task->M) && (j + N_STEP == task->N)); // not last block
            }

            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                _tile_stream_loadd(6, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B0
                _tile_stream_loadd(7, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B1
                _tile_loadd(4, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A0
                _tile_loadd(5, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A1
                run_4_tdp();

                if constexpr (SWPF_B) swpfetcher_B.prefetch();
                if constexpr (SWPF_A) swpfetcher_A.prefetch();
                if constexpr (SWPF_C) swpfetcher_C.prefetch();
            } // end for k
            store_4_tileC_l1(C_ptr);
            C_ptr += 4 * TILE_SIZE_i32;
        }
    }
}


/// @brief AMX GEMM Core **2A2B4C** Tiling with packed A & B & C
///
/// **Cache Policy**:
/// - Stream `A`, `C` into `L1D`
/// - Stream `B` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core_packABC_v1(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    int swpf_choice = params.swpfA << 2 | params.swpfB << 1 | params.swpfC;
    switch (swpf_choice) {
        case 0b000:
            amx_gemm_core_packABC_v1_template<false, false, false>(task);
            break;
        case 0b100:
            amx_gemm_core_packABC_v1_template<true, false, false>(task);
            break;
        case 0b101:
            amx_gemm_core_packABC_v1_template<true, false, true>(task);
            break;
        case 0b111:
            amx_gemm_core_packABC_v1_template<true, true, true>(task);
            break;
        default:
            throw std::runtime_error("Unsupported SWPF configuration!");
    }
}



template <bool SWPF_A, bool SWPF_B, bool SWPF_C>
void Kernel::amx_gemm_core_packABC_v2_template(taskSize *task) {
    const int8_t *A_ptr, *B_ptr;
    int32_t *C_ptr = task->C;

    using FetcherA_t = std::conditional_t<SWPF_A, SWPFetcher, NoOpFetcher>;
    using FetcherB_t = std::conditional_t<SWPF_B, SWPFetcher, NoOpFetcher>;
    using FetcherC_t = std::conditional_t<SWPF_C, SWPFetcher, NoOpFetcher>;
    FetcherA_t swpfetcher_A;
    FetcherB_t swpfetcher_B;
    FetcherC_t swpfetcher_C;

    if constexpr (SWPF_A) {
        const int8_t *next_A_ptr = task->A + task->M * task->K; // prefetch next TM×TK blockA
        swpfetcher_A = FetcherA_t(task->M * task->K, 1, next_A_ptr);
    }
    for (int j = 0; j < task->N; j += N_STEP) {
        A_ptr = task->A;

        if constexpr (SWPF_B) {
            const int8_t *next_B_ptr = task->B + (j + N_STEP) * task->K; // prefetch next 32×TK blockB
            swpfetcher_B = FetcherB_t(N_STEP * task->K, 2, next_B_ptr);
            swpfetcher_B.on = (j + N_STEP < task->N);
        }
        for (int i = 0; i < task->M; i += M_STEP) {
            B_ptr = task->B + j * task->K;
            load_4_tileC_l2(C_ptr);

            if constexpr (SWPF_C) {
                const int8_t *next_C_ptr = reinterpret_cast<const int8_t*>(C_ptr + M_STEP * N_STEP);
                swpfetcher_C = FetcherC_t(M_STEP * N_STEP * sizeof(int32_t), 4, next_C_ptr);
                swpfetcher_C.on = !((i + M_STEP == task->M) && (j + N_STEP == task->N)); // not last block
            }
            #pragma GCC unroll 20 // GEMM core loop, TK / K_STEP = 20
            for (int k = 0; k < task->K; k += K_STEP) {
                _tile_stream_loadd(4, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A0
                _tile_stream_loadd(5, A_ptr, MIN_STRIDE);
                A_ptr += TILE_SIZE_i8; // tileload A1
                _tile_loadd(6, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B0
                _tile_loadd(7, B_ptr, MIN_STRIDE);
                B_ptr += TILE_SIZE_i8; // tileload B1
                run_4_tdp();

                if constexpr (SWPF_A) swpfetcher_A.prefetch();
                if constexpr (SWPF_B) swpfetcher_B.prefetch();
                if constexpr (SWPF_C) swpfetcher_C.prefetch();
            } // end for k
            store_4_tileC_l1(C_ptr);
            C_ptr += 4 * TILE_SIZE_i32;
        }
    }
}



/// @brief AMX GEMM Core **2A2B4C** Tiling with packed A & B & C
///
/// **Cache Policy**:
/// - Stream `B`, `C` into `L1D`
/// - Stream `A` into `L2`
/// 
/// @param task Configuration for the current matrix block.
void Kernel::amx_gemm_core_packABC_v2(taskSize *task) {
    auto full_task = get_full_task();
    if (task == nullptr) task = &full_task;

    int swpf_choice = params.swpfA << 2 | params.swpfB << 1 | params.swpfC;
    switch (swpf_choice) {
        case 0b000:
            amx_gemm_core_packABC_v2_template<false, false, false>(task);
            break;
        case 0b010:
            amx_gemm_core_packABC_v2_template<false, true, false>(task);
            break;
        case 0b011:
            amx_gemm_core_packABC_v2_template<false, true, true>(task);
            break;
        case 0b111:
            amx_gemm_core_packABC_v2_template<true, true, true>(task);
            break;
        default:
            throw std::runtime_error("Unsupported SWPF configuration!");
    }
}


/// @brief AMX GEMM L2 blocking wrapper, computes the entire MxN matrix block
void Kernel::amx_gemm_blocking() {
    void (Kernel::*gemm_core)(taskSize *task) = nullptr;
    // select gemm core based on packing options
    if (params.packA && params.packB && params.packC) {
        gemm_core = (M >= N) ? 
            &Kernel::amx_gemm_core_packABC_v1 : &Kernel::amx_gemm_core_packABC_v2;
    } else if (params.packA && params.packB) {
        gemm_core = (M >= N) ? 
            &Kernel::amx_gemm_core_packAB_v1 : &Kernel::amx_gemm_core_packAB_v2;
    }
    else if (params.packB) {
        gemm_core = &Kernel::amx_gemm_core_packB;
    }
    else {
        gemm_core = &Kernel::amx_gemm_core;
    }
    for (int tk = 0; tk < K; tk += TK) {
        if (M >= N) {
            for (int tn = 0; tn < N; tn += TN) {
                taskSize task = {
                    .A = bufferA.get_block(tk, 0, lda),
                    .B = bufferB.get_block(tk, tn, ldb),
                    .C = bufferC.get_block(0, tn, ldc),
                    .M = M,
                    .N = MIN(TN, N - tn),
                    .K = MIN(TK, K - tk)
                };
                (this->*gemm_core)(&task);
            }
        }
        else { // M < N
            for (int tm = 0; tm < M; tm += TM) {
                taskSize task = {
                    .A = bufferA.get_block(tk, tm, lda),
                    .B = bufferB.get_block(tk, 0, ldb),
                    .C = bufferC.get_block(tm, 0, ldc),
                    .M = MIN(TM, M - tm),
                    .N = N,
                    .K = MIN(TK, K - tk)
                };
                (this->*gemm_core)(&task);
            }
        }
        
    }
}


void Kernel::amx_gemm_compute() {
    amx_gemm_blocking(); // launch AMX GEMM with L2 blocking
}


void KernelMT::work_per_thread(int tid, int core_id) {
    try {
        bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
    } catch (const std::exception& e) {
        std::cerr << "Thread bind failed: " << e.what() << std::endl;
        return;
    }

    Kernel::amx_init(); // 初始化 AMX

    int blocks_m = CEIL(M, TM);
    int blocks_n = CEIL(N, TN);
    int total_blocks = blocks_m * blocks_n;
    int num_threads = params.core_list.size();

    // 使用简单的 Round-Robin 分配任务块
    for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
        int bm = (block_id / blocks_n) * TM;
        int bn = (block_id % blocks_n) * TN;
        // 启动计算内核实例
        Kernel kernel(
            MIN(TM, M - bm), MIN(TN, N - bn), K,
            lda, ldb, ldc,
            &A[OFFSET2D(bm, 0, lda)],
            &B[OFFSET2D(0, bn, ldb)],
            &C[OFFSET2D(bm, bn, ldc)]
        );

        kernel.prepare_packed_data();
        kernel.amx_gemm_compute();
        kernel.restore_packed_data();
    }
}



void KernelMT::amx_gemm() {
    int num_threads = params.core_list.size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&KernelMT::work_per_thread, this, i, params.core_list[i]);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}


} // namespace amx