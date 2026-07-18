#pragma once
#include <type_traits>
#include <stdexcept>

// 二维行主序索引: 第 (x,y) 个元素在 leading dimension 为 ld 的数组中的偏移。
// 保留为宏: 每个参数只求值一次(无重复求值缺陷)，且 101+ 处热路径使用，做成函数无收益。
#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))

namespace amx {

// 用 common_type_t 支持混合类型实参(如 min(int, constexpr int))，保留宏原有的隐式转换语义。

template <typename A, typename B>
constexpr std::common_type_t<A, B> min(A a, B b) {
    return a < b ? a : b;
}

template <typename A, typename B>
constexpr std::common_type_t<A, B> max(A a, B b) {
    return a > b ? a : b;
}

// 向上取整除法: ceil(x / y)。要求 y > 0。
template <typename A, typename B>
constexpr std::common_type_t<A, B> ceil_div(A x, B y) {
    return (x + y - 1) / y;
}

// 向下/向上取整到 step 的整数倍
template <typename T>
constexpr T round_down(T a, T b) {
    if (b <= 0) throw std::invalid_argument("b must be positive");
    return (a / b) * b;
}

template <typename T>
constexpr T round_up(T a, T b) {
    if (b <= 0) throw std::invalid_argument("b must be positive");
    return a == 0 ? 0 : ((a - 1) / b + 1) * b;
}

} // namespace amx
