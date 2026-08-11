#pragma once
// packing.hpp — 基于 Buffer/View 的统一 pack/unpack(从 amx-gemm.cpp 移植)。
//
// 结构沿用原实现的三层:
//   L1 tile 原语 : pack_tile_a(直拷) / pack_tile_b(VNNI) / unpack_tile_c(累加|NT)
//   L2 分组      : pack_2_tile_a / pack_2_tile_b / unpack_4_tile_c
//   顶层         : pack(View) / unpack(View) —— 按 view.role() 选原语,按 view 的
//                  shape/stride 遍历 tile 网格并定落点(遍历顺序不影响正确性)。
//
// 这里的 pack/unpack 只处理「一个 2 层 tile panel」;整矩阵的 cache-block 层由调用方
// 循环(对 C 即逐 panel 用带 offset 的 make_view),与原 pack_from 的 3D 模式等价。
//
// TODO(集成): 下面几个 tile 常量 mirror amx-gemm.hpp;集成时统一到 buffer.hpp 并删重复。

#include "buffer.hpp"
#include <immintrin.h>

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#endif

namespace amx {

inline constexpr int MAX_COLS_i8   = 64;                       // == KR,单个 int8 tile 的列数
inline constexpr int MAX_COLS_i32  = 16;                       // == MAX_ROWS,单个 int32 tile 的列数
inline constexpr int TILE_SIZE_i8  = MAX_ROWS * MAX_COLS_i8;   // 1024,单个 int8 tile
inline constexpr int TILE_SIZE_i32 = MAX_ROWS * MAX_COLS_i32;  // 256, 单个 int32 tile

// =============================================================================
// L1 — tile 原语
// =============================================================================

// A: 16 行 × 64B 拷贝,strided(lda)→ dense(MIN_STRIDE)
inline void pack_tile_a(const int8_t *src, int8_t *dst, int lda) {
#pragma GCC unroll 16
    for (int r = 0; r < MAX_ROWS; ++r) {
        __m512i v = _mm512_loadu_si512(src); // src 不保证 64B 对齐
        _mm512_store_si512(dst, v);          // packed 已 64B 对齐
        src += lda;
        dst += MIN_STRIDE;
    }
}

// B: 把 4 行 × 16 列 int8 混洗成 1 行 VNNI(64B):{a0b0c0d0, a1b1c1d1, ...}。
ALWAYS_INLINE void pack_rows_4_vnni(const int8_t *src, int8_t *dst, int ldb) {
    __m128i r0 = _mm_loadu_si128((const __m128i *)(src));
    __m128i r1 = _mm_loadu_si128((const __m128i *)(src + ldb));
    __m128i r2 = _mm_loadu_si128((const __m128i *)(src + 2 * ldb));
    __m128i r3 = _mm_loadu_si128((const __m128i *)(src + 3 * ldb));
    __m128i t0 = _mm_unpacklo_epi8(r0, r1);
    __m128i t1 = _mm_unpackhi_epi8(r0, r1);
    __m128i t2 = _mm_unpacklo_epi8(r2, r3);
    __m128i t3 = _mm_unpackhi_epi8(r2, r3);
    __m128i v0 = _mm_unpacklo_epi16(t0, t2);
    __m128i v1 = _mm_unpackhi_epi16(t0, t2);
    __m128i v2 = _mm_unpacklo_epi16(t1, t3);
    __m128i v3 = _mm_unpackhi_epi16(t1, t3);
    __m512i zmm_out = _mm512_castsi128_si512(v0);
    zmm_out = _mm512_inserti32x4(zmm_out, v1, 1);
    zmm_out = _mm512_inserti32x4(zmm_out, v2, 2);
    zmm_out = _mm512_inserti32x4(zmm_out, v3, 3);
    _mm512_store_si512(dst, zmm_out);
}

// B: 16 个输出行,每行由 4 个 K-行交织(共覆盖 64 K-行 × 16 N-列)
inline void pack_tile_b(const int8_t *src, int8_t *dst, int ldb) {
    for (int r = 0; r < MAX_ROWS; ++r) {
        pack_rows_4_vnni(src, dst, ldb);
        src += KPACK_b8 * ldb;
        dst += MIN_STRIDE;
    }
}

// C: 16 行 × 16 int32 dense → strided(ldc)。acc=true 累加;false 走 NT stream(需外层 sfence)。
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

// =============================================================================
// L2 — 分组(dst/src 指针在组内步进,tile 连续码放)
// =============================================================================

// A 的一个 MR×KR 块 = 2 个竖排 tile
inline void pack_2_tile_a(const int8_t *src, int8_t *&dst, int lda) {
    pack_tile_a(src, dst, lda);
    dst += TILE_SIZE_i8;
    pack_tile_a(src + MAX_ROWS * lda, dst, lda);
    dst += TILE_SIZE_i8;
}

// B 的一个 KR×NR 块 = 2 个横排 tile(N 方向 +16 列)
inline void pack_2_tile_b(const int8_t *src, int8_t *&dst, int ldb) {
    pack_tile_b(src, dst, ldb);
    dst += TILE_SIZE_i8;
    pack_tile_b(src + (MAX_COLS_i8 / KPACK_b8), dst, ldb);
    dst += TILE_SIZE_i8;
}

// C 的一个 MR×NR tile-group = 2×2 tile(C00/C01/C10/C11)
inline void unpack_4_tile_c(const int32_t *&src, int32_t *dst, int ldc, bool acc) {
    if (acc) {
        unpack_tile_c<true>(src, dst, ldc);                                 src += TILE_SIZE_i32;
        unpack_tile_c<true>(src, dst + MAX_COLS_i32, ldc);                  src += TILE_SIZE_i32;
        unpack_tile_c<true>(src, dst + MAX_ROWS * ldc, ldc);                src += TILE_SIZE_i32;
        unpack_tile_c<true>(src, dst + MAX_ROWS * ldc + MAX_COLS_i32, ldc); src += TILE_SIZE_i32;
    } else {
        unpack_tile_c<false>(src, dst, ldc);                                 src += TILE_SIZE_i32;
        unpack_tile_c<false>(src, dst + MAX_COLS_i32, ldc);                  src += TILE_SIZE_i32;
        unpack_tile_c<false>(src, dst + MAX_ROWS * ldc, ldc);                src += TILE_SIZE_i32;
        unpack_tile_c<false>(src, dst + MAX_ROWS * ldc + MAX_COLS_i32, ldc); src += TILE_SIZE_i32;
        _mm_sfence(); // NT store 后强制写可见
    }
}

// =============================================================================
// 顶层 — 统一 pack / unpack(按 role 分派)
// =============================================================================

// 遍历 View tile 网格的步长:A=MR×KR, B=KR×NR, C=MR×NR。
struct TileStep { int rows, cols; };
inline TileStep tile_step(Role role) {
    switch (role) {
    case Role::A: return {MR, KR};
    case Role::B: return {KR, NR};
    case Role::C: return {MR, NR};
    }
    return {MR, NR};
}


// View<int8_t>::pack_from —— strided 源(row-major,前导维 src_ld)→ 本 packed view(role A/B)。
// 落点由 tile(i,j) 的 stride 决定,遍历顺序不影响正确性;role 分支循环不变,编译器外提。
template <typename T>
inline void View<T>::pack_from(const T *src, int src_ld) {
    assert(is_dense() && role_ != Role::C);
    const TileStep st = tile_step(role_);
    for (int i = 0; i < rows(); i += st.rows) {
        for (int j = 0; j < cols(); j += st.cols) {
            T *dptr = tile(i, j).data;             // packed 落点(stride 定位)
            const T *sptr = src + i * src_ld + j;  // strided 源 tile 左上角
            if (role_ == Role::A)
                pack_2_tile_a(sptr, dptr, src_ld);
            else
                pack_2_tile_b(sptr, dptr, src_ld);
        }
    }
}

// View<int32_t>::unpack_to —— 本 packed view(role C)→ strided 目标(前导维 dst_ld)。
// acc=true 累加进 dst,false NT 覆盖写。
template <typename T>
inline void View<T>::unpack_to(T *dst, int dst_ld, bool acc) const {
    assert(is_dense() && role_ == Role::C);
    const TileStep st = tile_step(Role::C);
    for (int i = 0; i < rows(); i += st.rows) {
        for (int j = 0; j < cols(); j += st.cols) {
            const T *sptr = tile(i, j).data;   // packed tile-group 基址
            T *dptr = dst + i * dst_ld + j;     // strided 落点
            unpack_4_tile_c(sptr, dptr, dst_ld, acc);
        }
    }
}

} // namespace amx
