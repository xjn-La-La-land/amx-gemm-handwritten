#pragma once
// TODO(集成): tile 几何常量本文件先自带,便于独立 review;集成时由 amx-gemm.hpp
//   include 本文件并删其重复定义(本文件作为 tile 几何唯一真源)。

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>

namespace amx {

// ---- tile 几何常量 (mirror amx-gemm.hpp,集成时统一到此) ----------------------
inline constexpr int MAX_ROWS       = 16;
inline constexpr int MR             = MAX_ROWS * 2; // 32, micro-kernel M
inline constexpr int NR             = MAX_ROWS * 2; // 32, micro-kernel N
inline constexpr int KR             = 64;           // micro-kernel K (= MAX_COLS_i8)
inline constexpr int MIN_STRIDE     = 64;           // packed tile 的行字节跨度
inline constexpr int CACHELINE_SIZE = 64;
inline constexpr int KPACK_b8       = 4;            // VNNI: K 方向 4 个 int8 交织

// shape / stride 固定 rank-2,零堆、平凡可拷贝
using Shape   = std::array<int, 2>; // {rows, cols}
using Strides = std::array<int, 2>; // {s0, s1}

// ---- 布局标签 / 角色 ----------------------------------------------------------
enum class Layout : uint8_t { Dense, Strided };
enum class Role   : uint8_t { A, B, C };

// 直接喂给 _tile_loadd / _tile_stored 的 tile 句柄
template <typename T>
struct Tile {
    T*      data;      // tile 的起始地址
    int32_t row_bytes; // 传给 AMX load/store 的 stride(字节)
};

// -----------------------------------------------------------------------------
// View<T>: 非拥有的寻址窗口。平凡可拷贝、值传递、无所有权,生命周期绑它借用的 Buffer。
// 所有的寻址与切片都在这里进行
// -----------------------------------------------------------------------------
template <typename T>
struct View {
    T*      data_   = nullptr;
    Shape   shape_  = {0, 0};       // {rows, cols}(逻辑矩阵形状)
    Strides stride_ = {0, 0};       // {s0, s1}
    Layout  layout_ = Layout::Dense;
    Role    role_   = Role::A;      // A/B/C —— 让统一的 pack/unpack 能据此分派 tile 几何

    // ---- 构造 ----
    // 唯一全量构造(由 make_view 工厂调用);block() 等内部走默认构造 + 逐字段赋值。
    View() = default;
    View(Role role, T* data, Shape shape, Strides stride, Layout layout)
        : data_(data), shape_(shape), stride_(stride), layout_(layout), role_(role) {}

    // ---- 基本查询 ----
    int  rows()   const { return shape_[0]; }
    int  cols()   const { return shape_[1]; }
    int  size()   const { return shape_[0] * shape_[1]; } // 元素个数(推进 running 指针用)
    T*   data()   const { return data_; }
    bool valid()  const { return data_ != nullptr; }
    bool is_dense()   const { return layout_ == Layout::Dense; }
    bool is_strided() const { return layout_ == Layout::Strided; }
    int  row_stride() const { return stride_[0]; } // leading dim / panel 宽
    Role role()       const { return role_; }

    // 仿射地址,无边界检查 
    T* operator()(int r, int c) const { return data_ + r * stride_[0] + c * stride_[1]; }

    // tile 基址 + AMX load 所需字节行跨度
    Tile<T> tile(int r, int c) const {
        assert(r >= 0 && c >= 0 && r < shape_[0] && c < shape_[1]);
        const int32_t row_bytes = is_dense() ? MIN_STRIDE
                                : static_cast<int32_t>(stride_[0] * sizeof(T));
        return Tile<T>{data_ + r * stride_[0] + c * stride_[1], row_bytes};
    }

    // 子矩形:data 偏移到 (r0,c0) 的 tile 基址,shape 收缩,stride/layout/role 继承。
    // Dense 下调用方需保证 (r0,c0) tile 对齐。
    View block(int r0, int c0, int rows, int cols) const {
        assert(r0 >= 0 && c0 >= 0 && r0 + rows <= shape_[0] && c0 + cols <= shape_[1]);
        View v;
        v.data_   = data_ + r0 * stride_[0] + c0 * stride_[1];
        v.shape_  = {rows, cols};
        v.stride_ = stride_;
        v.layout_ = layout_;
        v.role_   = role_;
        return v;
    }

    // pack/unpack —— 定义在 packing.hpp
    void pack_from(const T* src, int src_ld);            // role A/B:strided 源 → 本 packed view
    void unpack_to(T* dst, int dst_ld, bool acc) const;  // role C  :本 packed view → strided 目标
};

// -----------------------------------------------------------------------------
// Buffer<T>: 唯一持有对齐内存的 owner(move-only)。纯 1D 存储，不会感知矩阵形状/stride/layout/role
// -----------------------------------------------------------------------------
template <typename T>
class Buffer {
public:
    Buffer() = default;

    // 分配 n_elems 个 T 的对齐内存(64B 对齐)。Buffer 只关心「有多少个元素」。
    static Buffer allocate(size_t n_elems) {
        const size_t bytes = round_up_cacheline(n_elems * sizeof(T));
        void* raw = std::aligned_alloc(CACHELINE_SIZE, bytes);
        if (!raw) throw std::bad_alloc();

        Buffer buf;
        buf.storage_.reset(static_cast<T*>(raw));
        buf.size_ = n_elems;
        return buf;
    }

    bool   valid()  const { return static_cast<bool>(storage_); }
    T*     data()   const { return storage_.get(); }
    size_t size()   const { return size_; }              // 元素个数
    size_t nbytes() const { return size_ * sizeof(T); }

private:
    struct AlignedFree { void operator()(T* p) const noexcept { std::free(p); } };

    static size_t round_up_cacheline(size_t n) {
        return (n + (CACHELINE_SIZE - 1)) & ~static_cast<size_t>(CACHELINE_SIZE - 1);
    }

    std::unique_ptr<T[], AlignedFree> storage_;
    size_t size_ = 0; // 元素个数
};

// -----------------------------------------------------------------------------
// storage → view 工厂。三个重载覆盖三种场景,尽量自动推断 layout / stride / offset:
//   - Strided    : 借用外部 row-major 矩阵(原始 A/C),给 ld → stride = {ld, 1}。
//   - Dense 自动 : A/B,stride 由 role 推(A 行优先 {cols,MR}、B 列优先 {NR,rows})。
//   - Dense 显式 : C 必走(其 panel 朝向随 loop_order 变,推不出);A/B 需非默认布局时亦可。
// offset 是偏移的元素个数
// -----------------------------------------------------------------------------

// 从原始 RowMajor 矩阵 A/B/C 构造 View
template <typename T>
inline View<T> make_view(Role role, T* ptr, Shape shape, int ld) {
    return View<T>(role, ptr, shape, {ld, 1}, Layout::Strided);
}

// 从 Buffer 构造 View, stride 由 role 自动推导（A -> RowGrid, B -> ColGrid）
template <typename T>
inline View<T> make_view(Role role, Buffer<T>& buf, Shape shape, int offset = 0) {
    assert(buf.valid() && offset >= 0);
    if (offset + shape[0] * shape[1] > buf.size()) {
        throw std::out_of_range("make_view: offset + shape exceeds buffer size");
    }

    Strides stride;
    if (role == Role::A) {
        stride = {shape[1], MR}; // {cols, MR}
    } else if (role == Role::B) {
        stride = {NR, shape[0]}; // {NR, rows}
    } else if (role == Role::C) {
        std::cerr << "[Error] make_view(role=C,Buffer,Shape) is ambiguous! Use make_view(role=C,Buffer,Shape,Strides) instead." << std::endl;
    }
    return View<T>(role, buf.data() + offset, shape, stride, Layout::Dense);
}

// 从 Buffer 构造 View
template <typename T>
inline View<T> make_view(Role role, Buffer<T>& buf, Shape shape, Strides stride,
                         int offset = 0) {
    assert(buf.valid() && offset >= 0);
    if (offset + shape[0] * shape[1] > buf.size()) {
        throw std::out_of_range("make_view: offset + shape exceeds buffer size");
    }
    return View<T>(role, buf.data() + offset, shape, stride, Layout::Dense);
}

} // namespace amx
