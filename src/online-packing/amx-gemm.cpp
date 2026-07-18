#include "amx-gemm.hpp"

namespace amx {
using Kernel = GEMMKernelInt8;
using KernelMT = GEMMKernelInt8MT;

// Packing Utils

void pack_tile_a(const int8_t *src, int8_t *dst, int lda) {
#pragma GCC unroll 16
  for (int r = 0; r < MAX_ROWS; ++r) {
    __m512i v = _mm512_loadu_si512(
        src); // loadu (unaligned) 因为源地址 lda 不一定 64B 对齐
    _mm512_store_si512(dst, v); // packed 内存已 64B 对齐
    src += lda;
    dst += MIN_STRIDE;
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
ALWAYS_INLINE void pack_rows_4_vnni(const int8_t *src, int8_t *dst, int ldb) {
  // 1. 加载 4 行数据 (每行 16 字节 -> XMM 寄存器)
  // 使用 loadu 因为源地址可能不是 16 字节对齐
  __m128i r0 = _mm_loadu_si128((const __m128i *)(src));
  __m128i r1 = _mm_loadu_si128((const __m128i *)(src + ldb));
  __m128i r2 = _mm_loadu_si128((const __m128i *)(src + 2 * ldb));
  __m128i r3 = _mm_loadu_si128((const __m128i *)(src + 3 * ldb));

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

// Candidate 2: AVX2 256-bit route
// Same 4x16B -> 64B VNNI semantics as pack_rows_4_vnni.
// Saves 2 XMM unpacklo/hi_epi16 instructions by widening to YMM for that step,
// at the cost of 2 permute2x128 to fix lane order before the final store.
ALWAYS_INLINE void pack_rows_4_vnni_avx2(const int8_t *src, int8_t *dst,
                                         int ldb) {
  __m128i r0 = _mm_loadu_si128((const __m128i *)(src));
  __m128i r1 = _mm_loadu_si128((const __m128i *)(src + ldb));
  __m128i r2 = _mm_loadu_si128((const __m128i *)(src + 2 * ldb));
  __m128i r3 = _mm_loadu_si128((const __m128i *)(src + 3 * ldb));

  // Step 1: XMM byte-interleave within each row pair → 16-bit {ab} and {cd}
  // halves. Must be done at XMM level: AVX2 unpack works lane-locally, so doing
  // this directly on YMMs would pair (r0,r2) and (r1,r3) per lane instead of
  // (r0,r1).
  __m128i t0 = _mm_unpacklo_epi8(r0, r1); // [a0,b0,...,a7,b7]
  __m128i t1 = _mm_unpackhi_epi8(r0, r1); // [a8,b8,...,a15,b15]
  __m128i t2 = _mm_unpacklo_epi8(r2, r3); // [c0,d0,...,c7,d7]
  __m128i t3 = _mm_unpackhi_epi8(r2, r3); // [c8,d8,...,c15,d15]

  // Step 2: pack into YMM so each lane holds matching column ranges for ab and
  // cd. A.low=t0(ab col0-7),  A.high=t1(ab col8-15) B.low=t2(cd col0-7),
  // B.high=t3(cd col8-15)
  __m256i A = _mm256_set_m128i(t1, t0);
  __m256i B = _mm256_set_m128i(t3, t2);

  // Step 3: 16-bit interleave across the two row-pair groups (one YMM op = two
  // XMM ops). u0.low  = unpacklo(t0,t2) → {a0,b0,c0,d0,...,a3,b3,c3,d3}  cols
  // 0-3 u0.high = unpacklo(t1,t3) → {a8,b8,c8,d8,...,a11,...}        cols  8-11
  // u1.low  = unpackhi(t0,t2) → {a4,b4,c4,d4,...,a7,...}         cols  4-7
  // u1.high = unpackhi(t1,t3) → {a12,...,a15,b15,c15,d15}        cols 12-15
  __m256i u0 = _mm256_unpacklo_epi16(A, B);
  __m256i u1 = _mm256_unpackhi_epi16(A, B);

  // Step 4: fix lane order → sequential column groups for the 64B store.
  // lo = [u0.low(col0-3)  | u1.low(col4-7)]
  // hi = [u0.high(col8-11)| u1.high(col12-15)]
  __m256i lo = _mm256_permute2x128_si256(u0, u1, 0x20);
  __m256i hi = _mm256_permute2x128_si256(u0, u1, 0x31);

  __m512i zmm_out = _mm512_castsi256_si512(lo);
  zmm_out = _mm512_inserti64x4(zmm_out, hi, 1);
  _mm512_store_si512((__m512i *)dst, zmm_out);
}

// Candidate 3: AVX-512 route
// Idea:
// 1. Load the 4 input rows into one ZMM register.
// 2. Use a single byte-level permute table to produce the VNNI layout directly.
// 3. Avoid incremental 128-bit lane assembly entirely.
//
// Notes:
// - This version is a sketch that assumes AVX-512 VBMI-style byte permute
// support.
// - The index table encodes:
//   {r0[0], r1[0], r2[0], r3[0], r0[1], r1[1], r2[1], r3[1], ... }.
// - If your target CPU does not support VBMI/VBMI2, this route is not
// applicable.
ALWAYS_INLINE void pack_rows_4_vnni_avx512(const int8_t *src, int8_t *dst,
                                           int ldb) {
  alignas(64) int8_t tmp[64];

  _mm_storeu_si128((__m128i *)(tmp + 0),
                   _mm_loadu_si128((const __m128i *)(src)));
  _mm_storeu_si128((__m128i *)(tmp + 16),
                   _mm_loadu_si128((const __m128i *)(src + ldb)));
  _mm_storeu_si128((__m128i *)(tmp + 32),
                   _mm_loadu_si128((const __m128i *)(src + 2 * ldb)));
  _mm_storeu_si128((__m128i *)(tmp + 48),
                   _mm_loadu_si128((const __m128i *)(src + 3 * ldb)));

  const __m512i rows = _mm512_load_si512((const __m512i *)tmp);

  // _mm512_setr_epi8 (low-to-high arg order) is absent in GCC immintrin.h.
  // Use a static aligned table instead: avoids synthesizing 64 scalar inserts
  // and ensures the constant stays in .rodata / gets cached across calls.
  static const int8_t idx_data[64] alignas(64) = {
      0,  16, 32, 48, 1,  17, 33, 49, 2,  18, 34, 50, 3,  19, 35, 51,
      4,  20, 36, 52, 5,  21, 37, 53, 6,  22, 38, 54, 7,  23, 39, 55,
      8,  24, 40, 56, 9,  25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
      12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63};
  const __m512i idx = _mm512_load_si512((const __m512i *)idx_data);

  const __m512i zmm_out = _mm512_permutexvar_epi8(idx, rows);
  _mm512_store_si512((__m512i *)dst, zmm_out);
}

void pack_tile_b(const int8_t *src, int8_t *dst, int ldb) {
  for (int r = 0; r < MAX_ROWS; r++) {
    pack_rows_4_vnni(src, dst, ldb); // 4rows * 16B => 64B VNNI
    // pack_rows_4_vnni_avx2(src, dst, ldb);
    // pack_rows_4_vnni_avx512(src, dst, ldb);
    src += KPACK_b8 * ldb;
    dst += MIN_STRIDE;
  }
}

// acc=false: NT-store path (16 × load + streamstore, no branch).
//   Requires dst 64B-aligned per row (ldc multiple of 16 and base ptr
//   64B-aligned). Bypasses L2 write-allocate; caller must issue _mm_sfence()
//   after the sweep.
// acc=true:  accumulate path (16 × loadu + load + add + storeu, no branch).
template <bool acc>
ALWAYS_INLINE void unpack_tile_c(const int32_t *src, int32_t *dst, int ldc) {
#pragma GCC unroll 16
  for (int r = 0; r < MAX_ROWS; ++r) {
    __m512i v_packed = _mm512_load_si512(src);
    if constexpr (acc) {
      __m512i v_dst = _mm512_loadu_si512(dst);
      _mm512_storeu_si512(dst, _mm512_add_epi32(v_dst, v_packed));
    } else {
      _mm512_stream_si512((__m512i *)dst, v_packed);
    }
    src += MAX_COLS_i32;
    dst += ldc;
  }
}

// pack 2 tiles of A
void pack_2_tile_a(const int8_t *src, int8_t *&dst, int lda) {
  pack_tile_a(src, dst, lda);
  dst += TILE_SIZE_i8;
  pack_tile_a(src + MAX_ROWS * lda, dst, lda);
  dst += TILE_SIZE_i8;
}

// pack 2 tiles of B
void pack_2_tile_b(const int8_t *src, int8_t *&dst, int ldb) {
  pack_tile_b(src, dst, ldb);
  dst += TILE_SIZE_i8;
  pack_tile_b(src + (MAX_COLS_i8 / KPACK_b8), dst, ldb);
  dst += TILE_SIZE_i8;
}

// unpack 4 tiles of C — branch on acc once here, not 16×4 times inside the
// tiles. acc=false: issues _mm_sfence() after the 4 NT stores to enforce write
// visibility.
void unpack_4_tile_c(const int32_t *&src, int32_t *dst, int ldc, bool acc) {
  auto call4 = [&]<bool A>() {
    unpack_tile_c<A>(src, dst, ldc);
    src += TILE_SIZE_i32;
    unpack_tile_c<A>(src, dst + MAX_COLS_i32, ldc);
    src += TILE_SIZE_i32;
    unpack_tile_c<A>(src, dst + MAX_ROWS * ldc, ldc);
    src += TILE_SIZE_i32;
    unpack_tile_c<A>(src, dst + MAX_ROWS * ldc + MAX_COLS_i32, ldc);
    src += TILE_SIZE_i32;
  };
  if (acc) {
    call4.template operator()<true>();
  } else {
    call4.template operator()<false>();
    _mm_sfence();
  }
}

template <typename T>
void BufferA<T>::pack_from(const T *src, int stride, int cache_blocking_sz) {
  if (!this->is_dense())
    throw std::logic_error("BufferA must be dense layout to pack!");

  auto pack_2D_block_row_major = [](const T *src, T *dst, int rows, int cols,
                                    int stride) {
    for (int i = 0; i < rows; i += MR) {
      for (int k = 0; k < cols; k += KR) {
        // pack 2 tiles of A
        pack_2_tile_a(&src[OFFSET2D(i, k, stride)], dst, stride);
      }
    }
  };
  auto pack_2D_block_col_major = [](const T *src, T *dst, int rows, int cols,
                                    int stride) {
    for (int k = 0; k < cols; k += KR) {
      for (int i = 0; i < rows; i += MR) {
        // pack 2 tiles of A
        pack_2_tile_a(&src[OFFSET2D(i, k, stride)], dst, stride);
      }
    }
  };

  if (cache_blocking_sz > 0) {
    // 3D packing for whole matrix A
    if (this->direction() == PackDirection::RowMajor) {
      for (int mc = 0; mc < this->rows(); mc += cache_blocking_sz) {
        const T *src_block = &src[OFFSET2D(mc, 0, stride)];
        T *dst_block = this->data() + mc * this->cols();
        pack_2D_block_col_major(src_block, dst_block,
                                std::min(cache_blocking_sz, this->rows() - mc),
                                this->cols(), stride);
      }
    } else { // ColMajor
      for (int kc = 0; kc < this->cols(); kc += cache_blocking_sz) {
        const T *src_block = &src[OFFSET2D(0, kc, stride)];
        T *dst_block = this->data() + kc * this->rows();
        pack_2D_block_row_major(src_block, dst_block, this->rows(),
                                std::min(cache_blocking_sz, this->cols() - kc),
                                stride);
      }
    }
  } else {
    // 2D packing for panel/block A
    if (this->direction() == PackDirection::RowMajor) {
      pack_2D_block_row_major(src, this->data(), this->rows(), this->cols(),
                              stride);
    } else { // ColMajor
      pack_2D_block_col_major(src, this->data(), this->rows(), this->cols(),
                              stride);
    }
  }
}

template <typename T>
void BufferB<T>::pack_from(const T *src, int stride, int cache_blocking_sz) {
  if (!this->is_dense())
    throw std::logic_error("BufferB must be dense layout to pack!");

  auto pack_2D_block_row_major = [](const T *src, T *dst, int rows, int cols,
                                    int stride) {
    for (int k = 0; k < rows; k += KR) {
      for (int j = 0; j < cols; j += NR) {
        // pack 2 tiles of B
        pack_2_tile_b(&src[OFFSET2D(k, j, stride)], dst, stride);
      }
    }
  };
  auto pack_2D_block_col_major = [](const T *src, T *dst, int rows, int cols,
                                    int stride) {
    for (int j = 0; j < cols; j += NR) {
      for (int k = 0; k < rows; k += KR) {
        // pack 2 tiles of B
        pack_2_tile_b(&src[OFFSET2D(k, j, stride)], dst, stride);
      }
    }
  };

  if (cache_blocking_sz > 0) {
    // 3D packing for whole matrix B
    if (this->direction() == PackDirection::RowMajor) {
      for (int kc = 0; kc < this->rows(); kc += cache_blocking_sz) {
        const T *src_block = &src[OFFSET2D(kc, 0, stride)];
        T *dst_block = this->data() + kc * this->cols();
        pack_2D_block_col_major(src_block, dst_block,
                                std::min(cache_blocking_sz, this->rows() - kc),
                                this->cols(), stride);
      }
    } else { // ColMajor
      for (int nc = 0; nc < this->cols(); nc += cache_blocking_sz) {
        const T *src_block = &src[OFFSET2D(0, nc, stride)];
        T *dst_block = this->data() + nc * this->rows();
        pack_2D_block_row_major(src_block, dst_block, this->rows(),
                                std::min(cache_blocking_sz, this->cols() - nc),
                                stride);
      }
    }
  } else {
    if (this->direction() == PackDirection::RowMajor) {
      pack_2D_block_row_major(src, this->data(), this->rows(), this->cols(),
                              stride);
    } else { // ColMajor
      pack_2D_block_col_major(src, this->data(), this->rows(), this->cols(),
                              stride);
    }
  }
}

template <typename T>
void BufferC<T>::unpack_to(T *dst, int stride, bool acc,
                           int cache_blocking_sz) const {
  if (!this->is_dense())
    throw std::logic_error("BufferC must be dense layout to pack!");

  auto unpack_2D_block_row_major = [acc](const T *src, T *dst, int rows,
                                         int cols, int stride) {
    for (int i = 0; i < rows; i += MR) {
      for (int j = 0; j < cols; j += NR) {
        // unpack 4 tiles of C
        unpack_4_tile_c(src, &dst[OFFSET2D(i, j, stride)], stride, acc);
      }
    }
  };
  auto unpack_2D_block_col_major = [acc](const T *src, T *dst, int rows,
                                         int cols, int stride) {
    for (int j = 0; j < cols; j += NR) {
      for (int i = 0; i < rows; i += MR) {
        // unpack 4 tiles of C
        unpack_4_tile_c(src, &dst[OFFSET2D(i, j, stride)], stride, acc);
      }
    }
  };

  if (cache_blocking_sz > 0) { // 3D unpacking for whole matrix C
    if (this->direction() == PackDirection::RowMajor) {
      for (int mc = 0; mc < this->rows(); mc += cache_blocking_sz) {
        const T *src_block = this->data() + mc * this->cols();
        T *dst_block = &dst[OFFSET2D(mc, 0, stride)];
        unpack_2D_block_col_major(
            src_block, dst_block,
            std::min(cache_blocking_sz, this->rows() - mc), this->cols(),
            stride);
      }
    } else { // ColMajor
      for (int nc = 0; nc < this->cols(); nc += cache_blocking_sz) {
        const T *src_block = this->data() + nc * this->rows();
        T *dst_block = &dst[OFFSET2D(0, nc, stride)];
        unpack_2D_block_row_major(
            src_block, dst_block, this->rows(),
            std::min(cache_blocking_sz, this->cols() - nc), stride);
      }
    }
  } else {
    if (this->direction() == PackDirection::RowMajor) {
      unpack_2D_block_row_major(this->data(), dst, this->rows(), this->cols(),
                                stride);
    } else { // ColMajor
      unpack_2D_block_col_major(this->data(), dst, this->rows(), this->cols(),
                                stride);
    }
  }
}

template class BufferA<int8_t>;
template class BufferB<int8_t>;
template class BufferC<int32_t>;

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

// Heuristic selection of different kernel routines based on matrix shapes and
// parameters
void Kernel::find_best_routine() {
  // int8_t magic_number = large_enough_m() << 2 | large_enough_n() << 1 |
  // large_enough_k();
  int8_t magic_number = 0b100;
  switch (magic_number) {
  case 0b111: // large M, N, K
    compute_func = &Kernel::GEMM_;
    if (M >= N) {
      routine = KernelRoutine::Routine1_GEPB;
      bufA.allocate_dense(M, KC, PackDirection::RowMajor);  // panel A
      bufB.allocate_dense(KC, NC, PackDirection::ColMajor); // block B
      bufC.allocate_dense(M, N, PackDirection::ColMajor);   // matrix C
    } else {
      routine = KernelRoutine::Routine2_GEBP;
      bufA.allocate_dense(MC, KC, PackDirection::RowMajor); // block A
      bufB.allocate_dense(KC, N, PackDirection::ColMajor);  // panel B
      bufC.allocate_dense(M, N, PackDirection::RowMajor);   // matrix C
    }
    break;
  case 0b110: // large M, N, small K
    compute_func = &Kernel::GEPP_;
    if (M >= N) {
      routine = KernelRoutine::Routine1_GEPB;
      bufA.allocate_dense(M, KC, PackDirection::RowMajor);  // panel A
      bufB.allocate_dense(KC, NC, PackDirection::ColMajor); // block B
    } else {
      routine = KernelRoutine::Routine2_GEBP;
      bufA.allocate_dense(MC, KC, PackDirection::RowMajor); // block A
      bufB.allocate_dense(KC, N, PackDirection::ColMajor);  // panel B
    }
    break;
  case 0b101: // large M, K, small N
    compute_func = &Kernel::GEMP_;
    routine = KernelRoutine::Routine1_GEPB;
    bufB.allocate_dense(KC, NC, PackDirection::ColMajor); // block B
    bufC.allocate_dense(M, N, PackDirection::ColMajor);   // matrix C
    break;
  case 0b100: // large M, small N, K
    compute_func = &Kernel::GEPB_;
    routine = KernelRoutine::Routine1_GEPB;
    bufB.allocate_dense(KC, NC, PackDirection::ColMajor); // block B
    break;
  case 0b011: // large N, K, small M
    compute_func = &Kernel::GEPM_;
    routine = KernelRoutine::Routine2_GEBP;
    bufA.allocate_dense(MC, KC, PackDirection::RowMajor); // block A
    bufB.allocate_dense(KC, N, PackDirection::ColMajor);  // panel B
    bufC.allocate_dense(M, N, PackDirection::RowMajor);   // matrix C
    break;
  case 0b010: // large N, small M, K
    compute_func = &Kernel::GEBP_;
    routine = KernelRoutine::Routine2_GEBP;
    bufA.allocate_dense(MC, KC, PackDirection::RowMajor); // block A
    bufB.allocate_dense(KC, N, PackDirection::ColMajor);  // panel B
    break;
  case 0b001: // large K, small M, N
    compute_func = &Kernel::GEPDOT_;
    routine = KernelRoutine::Routine3_GEPDOT;
    bufB.allocate_dense(K, N, PackDirection::ColMajor); // whole matrix B
    bufC.allocate_dense(M, N, PackDirection::ColMajor); // whole matrix C
    break;
  default: // small M, N, K
    compute_func = &Kernel::GEPB_;
    routine = KernelRoutine::Routine1_GEPB;               // default routine
    bufB.allocate_dense(KC, NC, PackDirection::ColMajor); // block B
    break;
  }

  // std::cout << "Selected Kernel Routine: \n";
  // if (routine == KernelRoutine::Routine1_GEPB) std::cout <<
  // routine_graphs[0]; else if (routine == KernelRoutine::Routine2_GEBP)
  // std::cout << routine_graphs[1]; else if (routine ==
  // KernelRoutine::Routine3_GEPDOT) std::cout << routine_graphs[2];
}

void Kernel::pack_A() {
  if (routine == KernelRoutine::Routine1_GEPB ||
      routine == KernelRoutine::Routine2_GEBP) {
    bufA.pack_from(A, lda, KC); // 3D packing whole matrix A
  } else if (routine == KernelRoutine::Routine3_GEPDOT) {
    bufA.pack_from(A,
                   lda); // 2D packing panel/block A, no 3D packing for GEPDOT
  }
}

void Kernel::pack_B() {
  if (routine == KernelRoutine::Routine1_GEPB ||
      routine == KernelRoutine::Routine2_GEBP) {
    bufB.pack_from(B, ldb, KC); // 3D packing whole matrix B
  } else if (routine == KernelRoutine::Routine3_GEPDOT) {
    bufB.pack_from(B,
                   ldb); // 2D packing panel/block B, no 3D packing for GEPDOT
  }
}

void Kernel::unpack_C() {
  if (routine == KernelRoutine::Routine1_GEPB ||
      routine == KernelRoutine::Routine2_GEBP) {
    bufC.unpack_to(C, ldc, params.beta == 0.0f ? false : true,
                   bufC.direction() == PackDirection::RowMajor ? MC : NC);
  } else if (routine == KernelRoutine::Routine3_GEPDOT) {
    bufC.unpack_to(C, ldc, params.beta == 0.0f ? false : true);
  }
}

// 重新分配 packed buffer 给整个矩阵 A, B, C，并进行数据打包
void Kernel::alloc_buffers() {
  PackDirection dirA, dirB, dirC;
  if (routine == KernelRoutine::Routine1_GEPB) {
    dirA = PackDirection::ColMajor;
    dirB = PackDirection::RowMajor;
    dirC = PackDirection::ColMajor;
  } else if (routine == KernelRoutine::Routine2_GEBP) {
    dirA = PackDirection::ColMajor;
    dirB = PackDirection::RowMajor;
    dirC = PackDirection::RowMajor;
  } else if (routine == KernelRoutine::Routine3_GEPDOT) {
    dirA = PackDirection::RowMajor;
    dirB = PackDirection::ColMajor;
    dirC = PackDirection::ColMajor;
  }

  bufA.allocate_dense(M, K, dirA);
  bufB.allocate_dense(K, N, dirB);
  bufC.allocate_dense(M, N, dirC);
}

void Kernel::prepare_packed_data() {
  alloc_buffers();
  pack_A();
  pack_B();
}

// write back packed C matrix to original layout
void Kernel::restore_packed_data() {
  if (!bufA.valid() || !bufB.valid() || !bufC.valid()) {
    std::cerr << __func__ << ": "
              << "Please call prepare_packed_data() first!\n";
    return;
  }
  unpack_C();
  // bufA.release_owned_data();
  // bufB.release_owned_data();
  // bufC.release_owned_data();
}

void Kernel::GEMM_compute() {
  if (!bufA.valid() || !bufB.valid() || !bufC.valid()) {
    std::cerr << __func__ << ": "
              << "Please call prepare_packed_data() first!\n";
    return;
  }

  if (routine == KernelRoutine::Routine1_GEPB) {
    int8_t *A_ptr = bufA.data();
    int8_t *B_ptr = bufB.data();
    GEPBKernelConfig cfg;

    for (int kc = 0; kc < K; kc += KC) {
      BufferA<int8_t> panelA(A_ptr, M, min(KC, K - kc), 0, DataLayout::Dense);

      int32_t *C_ptr = bufC.data();
      cfg.overwrite_C = (kc == 0); // for the first kc block, we overwrite C,
                                   // for the rest, we accumulate into C

      for (int nc = 0; nc < N; nc += NC) {
        BufferB<int8_t> blockB(B_ptr, min(KC, K - kc), min(NC, N - nc), 0,
                               DataLayout::Dense);
        BufferC<int32_t> panelC(C_ptr, M, min(NC, N - nc), 0,
                                DataLayout::Dense);
        GEPB_kernel(panelA, blockB, panelC, cfg);
        B_ptr += blockB.size();
        C_ptr += panelC.size();
      }
      A_ptr += panelA.size();
    }
  } else if (routine == KernelRoutine::Routine2_GEBP) {

    int8_t *A_ptr = bufA.data();
    int8_t *B_ptr = bufB.data();
    for (int kc = 0; kc < K; kc += KC) {
      BufferB<int8_t> panelB(B_ptr, min(KC, K - kc), N, 0, DataLayout::Dense);

      int32_t *C_ptr = bufC.data();
      bool acc = kc != 0; // for the first kc block, we overwrite C, for the
                          // rest, we accumulate into C
      for (int mc = 0; mc < M; mc += MC) {
        BufferA<int8_t> blockA(A_ptr, min(MC, M - mc), min(KC, K - kc), 0,
                               DataLayout::Dense);
        BufferC<int32_t> panelC(C_ptr, min(MC, M - mc), N, 0,
                                DataLayout::Dense);
        GEBP_kernel(blockA, panelB, panelC, acc); // compute kernel
        A_ptr += blockA.size();
        C_ptr += panelC.size();
      }
      B_ptr += panelB.size();
    }
  } else if (routine == KernelRoutine::Routine3_GEPDOT) {

    int32_t *C_ptr = bufC.data();
    for (int j = 0; j < N; j += NR) {
      const int8_t *B_panel_ptr = bufB.packed_ptr(0, j);

      for (int i = 0; i < M; i += MR) {
        const int8_t *B_ptr = B_panel_ptr;
        const int8_t *A_ptr = bufA.packed_ptr(i, 0);
        clear_4_tileC(); // clear tile C

        for (int k = 0; k < K; k += KR) {
          load_2_tileA_l1(A_ptr); // load tile A0, A1
          load_2_tileB_l2(B_ptr); // load tile B0, B1
          run_4_tdp();            // compute tile C
          B_ptr += NR * KR;
          A_ptr += MR * KR;
        }

        store_4_tileC_l1(C_ptr); // store tile C
        C_ptr += MR * NR;
      }
    }
  }
}

void Kernel::GEMM() { (this->*compute_func)(); }

// large M, large N, large K
void Kernel::GEMM_() {
  if (routine == KernelRoutine::Routine1_GEPB) {
    assert(bufA.valid() && bufB.valid() && bufC.valid());
    GEPBKernelConfig cfg;

    for (int kc = 0; kc < K; kc += KC) {
      bufA.reset_cols(min(KC, K - kc));
      bufB.reset_rows(min(KC, K - kc));

      int32_t *C_ptr = bufC.data();
      cfg.originA = &A[OFFSET2D(0, kc, lda)];
      cfg.overwrite_C = (kc == 0);

      for (int nc = 0; nc < N; nc += NC) {
        bufB.reset_cols(min(NC, N - nc));
        bufB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
        BufferC<int32_t> panelC(C_ptr, M, min(NC, N - nc), MIN_STRIDE,
                                DataLayout::Dense);

        cfg.fuse_packA = nc == 0;
        GEPB_kernel(bufA, bufB, panelC, cfg);

        C_ptr += panelC.size();
      }
    }
    unpack_C();
  }

  else if (routine == KernelRoutine::Routine2_GEBP) {
    assert(bufA.valid() && bufB.valid() && bufC.valid());

    for (int kc = 0; kc < K; kc += KC) {
      bufA.reset_cols(min(KC, K - kc));
      bufB.reset_rows(min(KC, K - kc));

      int32_t *C_ptr = bufC.data();
      bool acc = kc != 0; // for the first kc block, we overwrite C, for the
                          // rest, we accumulate into C
      const int8_t *B_kc_base = &B[OFFSET2D(kc, 0, ldb)]; // base addr of panelB

      for (int mc = 0; mc < M; mc += MC) {
        bufA.reset_rows(min(MC, M - mc));
        bufA.pack_from(&A[OFFSET2D(mc, kc, lda)], lda); // pack block A
        BufferC<int32_t> panelC(C_ptr, min(MC, M - mc), N, MIN_STRIDE,
                                DataLayout::Dense);
        if (mc == 0) {
          GEBP_kernel(bufA, bufB, panelC, acc, B_kc_base);
        } else {
          GEBP_kernel(bufA, bufB, panelC, acc);
        }
        C_ptr += panelC.size();
      }
    }
    unpack_C();
  }
}

// large M, large N, small K
void Kernel::GEPP_() {
  if (routine == KernelRoutine::Routine1_GEPB) {
    assert(bufA.valid() && bufB.valid());
    assert(!bufC.valid()); // no buffer for C

    GEPBKernelConfig cfg;

    for (int kc = 0; kc < K; kc += KC) {
      bufA.reset_cols(min(KC, K - kc));
      bufB.reset_rows(min(KC, K - kc));

      cfg.overwrite_C = (kc == 0) && (params.beta == 0.0f);
      cfg.originA = &A[OFFSET2D(0, kc, lda)];

      for (int nc = 0; nc < N; nc += NC) {
        bufB.reset_cols(min(NC, N - nc));
        bufB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
        BufferC<int32_t> panelC(&C[OFFSET2D(0, nc, ldc)], M, min(NC, N - nc),
                                ldc, DataLayout::Strided);

        cfg.fuse_packA = nc == 0;
        GEPB_kernel(bufA, bufB, panelC, cfg);
      }
    }

  }

  else if (routine == KernelRoutine::Routine2_GEBP) {
    assert(bufA.valid() && bufB.valid());
    assert(!bufC.valid()); // no buffer for C

    for (int kc = 0; kc < K; kc += KC) {
      bufA.reset_cols(min(KC, K - kc));
      bufB.reset_rows(min(KC, K - kc));

      const int8_t *B_kc_base = &B[OFFSET2D(kc, 0, ldb)]; // base addr of panelB
      for (int mc = 0; mc < M; mc += MC) {
        bufA.reset_rows(min(MC, M - mc));
        bufA.pack_from(&A[OFFSET2D(mc, kc, lda)], lda); // pack block A
        BufferC<int32_t> panelC(&C[OFFSET2D(mc, 0, ldc)], min(MC, M - mc), N,
                                ldc, DataLayout::Strided);
        if (mc == 0) {
          GEBP_kernel(bufA, bufB, panelC, true, B_kc_base);
        } else {
          GEBP_kernel(bufA, bufB, panelC, true);
        }
      }
    }
  }
}

// large M, small N, large K
void Kernel::GEMP_() {
  assert(routine == KernelRoutine::Routine1_GEPB);
  assert(!bufA.valid());                // no buffer for A
  assert(bufB.valid() && bufC.valid()); // block B and matrix C are packed

  GEPBKernelConfig cfg;

  for (int kc = 0; kc < K; kc += KC) {
    bufB.reset_rows(min(KC, K - kc));
    BufferA<int8_t> panelA(const_cast<int8_t *>(&A[OFFSET2D(0, kc, lda)]), M,
                           min(KC, K - kc), lda, DataLayout::Strided);

    cfg.overwrite_C = kc == 0;
    int32_t *C_ptr = bufC.data();

    for (int nc = 0; nc < N; nc += NC) {
      bufB.reset_cols(min(NC, N - nc));
      bufB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
      BufferC<int32_t> panelC(C_ptr, M, min(NC, N - nc), MIN_STRIDE,
                              DataLayout::Dense);

      GEPB_kernel(bufA, bufB, panelC, cfg);
      C_ptr += panelC.size();
    }
  }
  unpack_C();
}

// small M, large N, large K
void Kernel::GEPM_() {
  assert(routine == KernelRoutine::Routine2_GEBP);
  GEMM_();
}

// large M, small N, small K
void Kernel::GEPB_() {
  assert(routine == KernelRoutine::Routine1_GEPB);
  assert(!bufA.valid()); // no buffer for A
  assert(bufB.valid());  // block B is packed
  assert(!bufC.valid()); // no buffer for C

  GEPBKernelConfig cfg;

  for (int kc = 0; kc < K; kc += KC) {
    bufB.reset_rows(min(KC, K - kc));
    BufferA<int8_t> panelA(const_cast<int8_t *>(&A[OFFSET2D(0, kc, lda)]), M,
                           min(KC, K - kc), lda, DataLayout::Strided);

    cfg.overwrite_C = (kc == 0) && (params.beta == 0.0f);

    for (int nc = 0; nc < N; nc += NC) {
      bufB.reset_cols(min(NC, N - nc));
      bufB.pack_from(&B[OFFSET2D(kc, nc, ldb)], ldb); // pack block B
      BufferC<int32_t> panelC(&C[OFFSET2D(0, nc, ldc)], M, min(NC, N - nc), ldc,
                              DataLayout::Strided);
      GEPB_kernel(panelA, bufB, panelC, cfg);
    }
  }

}

// small M, large N, small K
void Kernel::GEBP_() {
  assert(routine == KernelRoutine::Routine2_GEBP);
  GEPP_();
}

void Kernel::GEPDOT_() {
  assert(routine == KernelRoutine::Routine3_GEPDOT);
  assert(!bufA.valid());                // no buffer for A
  assert(bufB.valid() && bufC.valid()); // matrix B and matrix C are packed

  bufB.pack_from(B, ldb); // pack whole matrix B into packed buffer
  // micro-kernel
  int32_t *C_ptr = bufC.data();
  for (int j = 0; j < N; j += NR) {
    const int8_t *B_panel_ptr = bufB.packed_ptr(0, j);

    for (int i = 0; i < M; i += MR) {
      const int8_t *B_ptr = B_panel_ptr;
      clear_4_tileC(); // clear tile C

      for (int k = 0; k < K; k += KR) {
        load_2_tileA_l1(&A[OFFSET2D(i, k, lda)]); // load tile A0, A1
        load_2_tileB_l2(B_ptr);                   // load tile B0, B1
        run_4_tdp();                              // compute tile C
        B_ptr += NR * KR;
      }

      store_4_tileC_l1(C_ptr); // store tile C
      C_ptr += MR * NR;
    }
  }

  bufC.unpack_to(C, ldc); // C += packed result
}

template <bool fuse_packA, bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseABC(BufferA<int8_t> &panelA,
                                       BufferB<int8_t> &blockB,
                                       BufferC<int32_t> &panelC,
                                       const int8_t *originA) {

  int32_t *C_ptr = panelC.data();
  SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                        2,          // step
                        _MM_HINT_T1 // hint: prefetch to L2
  );
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                        4,          // step
                        _MM_HINT_T0 // hint: prefetch to L1
  );

  for (int i = 0; i < panelA.rows(); i += MR) {

    const int8_t *cur_stripA_ptr = panelA.packed_ptr(i, 0);

    if constexpr (fuse_packA) { // pack stripA (MR x KC)
      int8_t *dst = const_cast<int8_t *>(cur_stripA_ptr);
      for (int k = 0; k < panelA.cols(); k += KR) {
        pack_2_tile_a(&originA[OFFSET2D(i, k, lda)], dst, lda);
      }
      swpf_ctx_a.on = false; // when packing into bufferA, no need to prefetch
    } else { // if A_kc_base is nullptr, it means the stripA is already packed,
             // we just need to prefetch it
      const int8_t *next_stripA_ptr = panelA.packed_ptr(i + MR, 0);
      swpf_ctx_a.init(next_stripA_ptr);
    }

    const int8_t *B_ptr = blockB.data();

    // GESB
    for (int j = 0; j < blockB.cols(); j += NR) {

      const int8_t *A_ptr = cur_stripA_ptr;
      if constexpr (overwrite_C) {
        clear_4_tileC();
      } else {
        load_4_tileC_l1(C_ptr);
      }

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockB.rows(); k += KR) {
        load_2_tileB_l2(B_ptr); // tileload B0, B1
        load_2_tileA_l1(A_ptr); // tileload A0, A1
        run_4_tdp();
        B_ptr += NR * KR;
        A_ptr += MR * KR;

        swpf_ctx_a.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(C_ptr);
      C_ptr += MR * NR;
    }
  }
}

template <bool fuse_packA, bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseAB_stridedC(BufferA<int8_t> &panelA,
                                               BufferB<int8_t> &blockB,
                                               BufferC<int32_t> &panelC,
                                               const int8_t *originA) {

  SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                        2,          // step
                        _MM_HINT_T1 // hint: prefetch to L2
  );
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                        4, // step
                        NR * sizeof(int32_t), panelC.stride() * sizeof(int32_t),
                        _MM_HINT_T0 // hint: prefetch to L1
  );

  for (int i = 0; i < panelA.rows(); i += MR) {

    const int8_t *cur_stripA_ptr = panelA.packed_ptr(i, 0);

    if constexpr (fuse_packA) { // pack stripA (MR x KC)
      int8_t *dst = const_cast<int8_t *>(cur_stripA_ptr);
      for (int k = 0; k < panelA.cols(); k += KR) {
        pack_2_tile_a(&originA[OFFSET2D(i, k, lda)], dst, lda);
      }
      swpf_ctx_a.on = false; // when packing into bufferA, no need to prefetch
    } else { // if A_kc_base is nullptr, it means the stripA is already packed,
             // we just need to prefetch it
      const int8_t *next_stripA_ptr = panelA.packed_ptr(i + MR, 0);
      swpf_ctx_a.init(next_stripA_ptr);
    }

    const int8_t *B_ptr = blockB.data();

    // GESB
    for (int j = 0; j < blockB.cols(); j += NR) {

      const int8_t *A_ptr = cur_stripA_ptr;
      if constexpr (overwrite_C) {
        clear_4_tileC();
      } else {
        load_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
      }

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(panelC.next_tile_ptr(i, j));
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockB.rows(); k += KR) {
        load_2_tileB_l2(B_ptr); // tileload B0, B1
        load_2_tileA_l1(A_ptr); // tileload A0, A1
        run_4_tdp();
        B_ptr += NR * KR;
        A_ptr += MR * KR;

        swpf_ctx_a.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
    }
  }
}

template <bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseBC_stridedA(BufferA<int8_t> &panelA,
                                               BufferB<int8_t> &blockB,
                                               BufferC<int32_t> &panelC) {

  int32_t *C_ptr = panelC.data();
  SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                        2, // step
                        panelA.cols() * sizeof(int8_t),
                        panelA.stride() * sizeof(int8_t),
                        _MM_HINT_T1 // hint: prefetch to L2
  );
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                        4,          // step
                        _MM_HINT_T0 // hint: prefetch to L1
  );

  for (int i = 0; i < panelA.rows(); i += MR) {

    const int8_t *next_stripA_ptr = panelA.ptr(i + MR, 0);
    swpf_ctx_a.init(next_stripA_ptr);

    const int8_t *B_ptr = blockB.data();

    // GESB
    for (int j = 0; j < blockB.cols(); j += NR) {

      if constexpr (overwrite_C) {
        clear_4_tileC();
      } else {
        load_4_tileC_l1(C_ptr);
      }

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockB.rows(); k += KR) {
        load_2_tileB_l2(B_ptr);                             // tileload B0, B1
        load_2_tileA_l1(panelA.ptr(i, k), panelA.stride()); // tileload A0, A1
        run_4_tdp();
        B_ptr += NR * KR;

        swpf_ctx_a.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(C_ptr);
      C_ptr += MR * NR;
    }
  }
}

template <bool overwrite_C>
void Kernel::GEPB_kernel_impl_denseB_stridedAC(BufferA<int8_t> &panelA,
                                               BufferB<int8_t> &blockB,
                                               BufferC<int32_t> &panelC) {

  SWPFHelper swpf_ctx_a(MR * panelA.cols() * sizeof(int8_t),
                        2, // step
                        panelA.cols() * sizeof(int8_t),
                        panelA.stride() * sizeof(int8_t),
                        _MM_HINT_T1 // hint: prefetch to L2
  );
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t),
                        4, // step
                        NR * sizeof(int32_t), panelC.stride() * sizeof(int32_t),
                        _MM_HINT_T0 // hint: prefetch to L1
  );

  for (int i = 0; i < panelA.rows(); i += MR) {

    const int8_t *next_stripA_ptr = panelA.ptr(i + MR, 0);
    swpf_ctx_a.init(next_stripA_ptr);

    const int8_t *B_ptr = blockB.data();

    // GESB
    for (int j = 0; j < blockB.cols(); j += NR) {

      if constexpr (overwrite_C) {
        clear_4_tileC();
      } else {
        load_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
      }

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(panelC.next_tile_ptr(i, j));
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockB.rows(); k += KR) {
        load_2_tileB_l2(B_ptr);                             // tileload B0, B1
        load_2_tileA_l1(panelA.ptr(i, k), panelA.stride()); // tileload A0, A1
        run_4_tdp();
        B_ptr += NR * KR;

        swpf_ctx_a.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
    }
  }
}

void Kernel::GEPB_kernel(BufferA<int8_t> &panelA, BufferB<int8_t> &blockB,
                         BufferC<int32_t> &panelC,
                         const GEPBKernelConfig &cfg) {
  using DenseABCFn = void (Kernel::*)(BufferA<int8_t> &, BufferB<int8_t> &,
                                      BufferC<int32_t> &, const int8_t *);
  static constexpr DenseABCFn dense_abc_dispatch[] = {
      &Kernel::GEPB_kernel_impl_denseABC<false, false>,
      &Kernel::GEPB_kernel_impl_denseABC<false, true>,
      &Kernel::GEPB_kernel_impl_denseABC<true, false>,
      &Kernel::GEPB_kernel_impl_denseABC<true, true>,
  };

  using DenseABStridedCFn = void (Kernel::*)(
      BufferA<int8_t> &, BufferB<int8_t> &, BufferC<int32_t> &, const int8_t *);
  static constexpr DenseABStridedCFn dense_ab_strided_c_dispatch[] = {
      &Kernel::GEPB_kernel_impl_denseAB_stridedC<false, false>,
      &Kernel::GEPB_kernel_impl_denseAB_stridedC<false, true>,
      &Kernel::GEPB_kernel_impl_denseAB_stridedC<true, false>,
      &Kernel::GEPB_kernel_impl_denseAB_stridedC<true, true>,
  };

  using DenseBCStridedAFn = void (Kernel::*)(
      BufferA<int8_t> &, BufferB<int8_t> &, BufferC<int32_t> &);
  static constexpr DenseBCStridedAFn dense_bc_strided_a_dispatch[] = {
      &Kernel::GEPB_kernel_impl_denseBC_stridedA<false>,
      &Kernel::GEPB_kernel_impl_denseBC_stridedA<true>,
  };

  using DenseBStridedACFn = void (Kernel::*)(
      BufferA<int8_t> &, BufferB<int8_t> &, BufferC<int32_t> &);
  static constexpr DenseBStridedACFn dense_b_strided_ac_dispatch[] = {
      &Kernel::GEPB_kernel_impl_denseB_stridedAC<false>,
      &Kernel::GEPB_kernel_impl_denseB_stridedAC<true>,
  };

  check_shape_health(panelA, blockB, panelC);

  assert(blockB.is_dense() && "GEPB_kernel requires dense layout for blockB");
  if (panelA.is_dense() && panelC.is_dense()) {
    (this->*dense_abc_dispatch[cfg.dense_ab_index()])(panelA, blockB, panelC,
                                                      cfg.originA);
  } else if (panelA.is_dense() && panelC.is_strided()) {
    (this->*dense_ab_strided_c_dispatch[cfg.dense_ab_index()])(
        panelA, blockB, panelC, cfg.originA);
  } else if (panelA.is_strided() && panelC.is_dense()) {
    (this->*dense_bc_strided_a_dispatch[cfg.overwrite_c_index()])(
        panelA, blockB, panelC);
  } else {
    (this->*dense_b_strided_ac_dispatch[cfg.overwrite_c_index()])(
        panelA, blockB, panelC);
  }
}

template <>
void Kernel::GEBP_kernel_impl<true, true>(BufferA<int8_t> &blockA,
                                          BufferB<int8_t> &panelB,
                                          BufferC<int32_t> &panelC, bool acc,
                                          const int8_t *B_kc_base) {

  int32_t *C_ptr = panelC.data();
  SWPFHelper swpf_ctx_b(panelB.rows() * NR * sizeof(int8_t), 2, _MM_HINT_T1);
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t), 4, _MM_HINT_T0);

  for (int j = 0; j < panelB.cols(); j += NR) {

    const int8_t *cur_stripB_ptr = panelB.packed_ptr(0, j);
    if (B_kc_base != nullptr) { // pack stripB (KC x NR)
      int8_t *dst = const_cast<int8_t *>(cur_stripB_ptr);
      for (int k = 0; k < panelB.rows(); k += KR) {
        pack_2_tile_b(&B_kc_base[OFFSET2D(k, j, ldb)], dst, ldb);
      }
    } else {
      const int8_t *next_stripB_ptr = panelB.packed_ptr(0, j + NR);
      swpf_ctx_b.init(next_stripB_ptr);
    }

    const int8_t *A_ptr = blockA.data();

    // GEBS
    for (int i = 0; i < blockA.rows(); i += MR) {

      const int8_t *B_ptr = cur_stripB_ptr;
      if (acc)
        load_4_tileC_l1(C_ptr);
      else
        clear_4_tileC();

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(C_ptr + MR * NR);
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockA.cols(); k += KR) {
        load_2_tileA_l2(A_ptr); // tileload A0, A1
        load_2_tileB_l1(B_ptr); // tileload B0, B1
        run_4_tdp();
        A_ptr += MR * KR;
        B_ptr += NR * KR;

        swpf_ctx_b.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(C_ptr);
      C_ptr += MR * NR;
    }
  }
}

template <>
void Kernel::GEBP_kernel_impl<true, false>(BufferA<int8_t> &blockA,
                                           BufferB<int8_t> &panelB,
                                           BufferC<int32_t> &panelC, bool acc,
                                           const int8_t *B_kc_base) {

  SWPFHelper swpf_ctx_b(panelB.rows() * NR * sizeof(int8_t), 2, _MM_HINT_T1);
  SWPFHelper swpf_ctx_c(MR * NR * sizeof(int32_t), 4, NR * sizeof(int32_t),
                        panelC.stride() * sizeof(int32_t), _MM_HINT_T0);

  for (int j = 0; j < panelB.cols(); j += NR) {

    const int8_t *cur_stripB_ptr = panelB.packed_ptr(0, j);
    if (B_kc_base != nullptr) { // pack stripB (KC x NR)
      int8_t *dst = const_cast<int8_t *>(cur_stripB_ptr);
      for (int k = 0; k < panelB.rows(); k += KR) {
        pack_2_tile_b(&B_kc_base[OFFSET2D(k, j, ldb)], dst, ldb);
      }
    } else {
      const int8_t *next_stripB_ptr = panelB.packed_ptr(0, j + NR);
      swpf_ctx_b.init(next_stripB_ptr);
    }

    const int8_t *A_ptr = blockA.data();

    // GEBS
    for (int i = 0; i < blockA.rows(); i += MR) {

      const int8_t *B_ptr = cur_stripB_ptr;
      if (acc)
        load_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
      else
        clear_4_tileC();

      const int8_t *next_C_ptr =
          reinterpret_cast<const int8_t *>(panelC.next_tile_ptr(i, j));
      swpf_ctx_c.init(next_C_ptr);

      for (int k = 0; k < blockA.cols(); k += KR) {
        load_2_tileA_l2(A_ptr); // tileload A0, A1
        load_2_tileB_l1(B_ptr); // tileload B0, B1
        run_4_tdp();
        A_ptr += MR * KR;
        B_ptr += NR * KR;

        swpf_ctx_b.prefetch();
        swpf_ctx_c.prefetch();
      } // end for k loop

      store_4_tileC_l1(panelC.ptr(i, j), panelC.stride());
    }
  }
}

void Kernel::GEBP_kernel(BufferA<int8_t> &blockA, BufferB<int8_t> &panelB,
                         BufferC<int32_t> &panelC, bool acc,
                         const int8_t *B_kc_base) {
  check_shape_health(blockA, panelB, panelC);

  using Impl = void (Kernel::*)(BufferA<int8_t> &, BufferB<int8_t> &,
                                BufferC<int32_t> &, bool, const int8_t *);
  Impl impl;

  assert(blockA.is_dense() && "GEBP_kernel requires dense layout for blockA");
  assert(panelB.is_dense() && "GEBP_kernel requires dense layout for panelB");

  if (panelC.is_dense()) {
    impl = &Kernel::GEBP_kernel_impl<true, true>;
  } else {
    impl = &Kernel::GEBP_kernel_impl<true, false>;
  }

  (this->*impl)(blockA, panelB, panelC, acc, B_kc_base);
}

///////////////////////////////////////////////////////
// Multi-threaded Kernel Management
///////////////////////////////////////////////////////

// 找到每个线程对应的 Kernel 实例并初始化
void KernelMT::init_kernel_per_thread(int tid, int core_id) {
  try {
    bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
  } catch (const std::exception &e) {
    std::cerr << "Thread bind failed: " << e.what() << std::endl;
    return;
  }

  int blocks_m = ceil_div(M, MC);
  int blocks_n = ceil_div(N, NC);
  int total_blocks = blocks_m * blocks_n;
  int num_threads = params.core_list.size();

  // 使用简单的 Round-Robin 分配任务块
  for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
    int bm = (block_id / blocks_n) * MC;
    int bn = (block_id % blocks_n) * NC;

    // 在 Kernel_pool 中创建 Kernel 实例
    auto kernel_ptr = std::make_unique<Kernel>(
        min(MC, M - bm), min(NC, N - bn), K, lda, ldb, ldc,
        &A[OFFSET2D(bm, 0, lda)], &B[OFFSET2D(0, bn, ldb)],
        &C[OFFSET2D(bm, bn, ldc)], gemm_params, BlockingConfig(MC, NC, KC));

    kernel_pool[block_id] = std::move(kernel_ptr);
  }
}

void KernelMT::init_kernels() {
  int num_threads = params.core_list.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  int blocks_m = ceil_div(M, MC);
  int blocks_n = ceil_div(N, NC);
  int total_blocks = blocks_m * blocks_n;
  kernel_pool.resize(
      total_blocks); // 调整 kernel_pool 大小以容纳所有线程的 Kernel 实例

  // 启动线程
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(&KernelMT::init_kernel_per_thread, this, i,
                         params.core_list[i]);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

void KernelMT::GEMM_per_thread(int tid, int core_id) {
  try {
    bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
  } catch (const std::exception &e) {
    std::cerr << "Thread bind failed: " << e.what() << std::endl;
    return;
  }

  Kernel::amx_init(); // 初始化 AMX

  // 找到对应的 Kernel 实例
  int num_threads = params.core_list.size();
  int total_blocks = kernel_pool.size();
  for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
    auto &kernel = kernel_pool[block_id];
    if (kernel) {
      kernel->GEMM();
    }
  }
}

void KernelMT::GEMM() {
  int num_threads = params.core_list.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // 启动线程
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(&KernelMT::GEMM_per_thread, this, i,
                         params.core_list[i]);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

void KernelMT::prepare_packed_data_per_thread(int tid, int core_id) {
  try {
    bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
  } catch (const std::exception &e) {
    std::cerr << "Thread bind failed: " << e.what() << std::endl;
    return;
  }

  // 找到对应的 Kernel 实例
  int num_threads = params.core_list.size();
  int total_blocks = kernel_pool.size();
  for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
    auto &kernel = kernel_pool[block_id];
    if (kernel) {
      kernel->prepare_packed_data();
    }
  }
}

void KernelMT::prepare_packed_data() {
  int num_threads = params.core_list.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // 启动线程
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(&KernelMT::prepare_packed_data_per_thread, this, i,
                         params.core_list[i]);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

void KernelMT::GEMM_compute_per_thread(int tid, int core_id) {
  try {
    bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
  } catch (const std::exception &e) {
    std::cerr << "Thread bind failed: " << e.what() << std::endl;
    return;
  }

  Kernel::amx_init(); // 初始化 AMX

  // 找到对应的 Kernel 实例
  int num_threads = params.core_list.size();
  int total_blocks = kernel_pool.size();
  for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
    auto &kernel = kernel_pool[block_id];
    if (kernel) {
      kernel->GEMM_compute();
    }
  }
}

void KernelMT::GEMM_compute() {
  int num_threads = params.core_list.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // 启动线程
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(&KernelMT::GEMM_compute_per_thread, this, i,
                         params.core_list[i]);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

void KernelMT::restore_packed_data_per_thread(int tid, int core_id) {
  try {
    bind_thread_to_cpu(core_id); // 绑定当前线程到物理核心
  } catch (const std::exception &e) {
    std::cerr << "Thread bind failed: " << e.what() << std::endl;
    return;
  }

  // 找到对应的 Kernel 实例
  int num_threads = params.core_list.size();
  int total_blocks = kernel_pool.size();
  for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
    auto &kernel = kernel_pool[block_id];
    if (kernel) {
      kernel->restore_packed_data();
    }
  }
}

void KernelMT::restore_packed_data() {
  int num_threads = params.core_list.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // 启动线程
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(&KernelMT::restore_packed_data_per_thread, this, i,
                         params.core_list[i]);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

} // namespace amx
