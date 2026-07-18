#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <type_traits>
#include <iomanip>
#include <sstream>
#include <algorithm>

// 调试用: 打印矩阵内容、GEMM routine ASCII 示意图。

// 打印矩阵(左上角 max_print_rows × max_print_cols 子块)
template <typename T>
void print_matrix(const std::string& name,
                  const T* data,
                  int rows,
                  int cols,
                  int ld,
                  int max_print_rows = 8,
                  int max_print_cols = 8,
                  bool hex = true) {
    int pr = std::min(rows, max_print_rows);
    int pc = std::min(cols, max_print_cols);

    // 将元素转为字符串(统一处理格式问题)
    auto format_elem = [&](T v) {
        std::ostringstream oss;

        if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(2) << v;
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            // & 0xff 避免负数被符号扩展成 0xffffffab
            if (hex)
                oss << "0x" << std::hex << (static_cast<int>(v) & 0xff);
            else
                oss << static_cast<int>(v);
        }
        else if constexpr (std::is_same_v<T, uint8_t>) {
            if (hex)
                oss << "0x" << std::hex << static_cast<unsigned int>(v);
            else
                oss << static_cast<unsigned int>(v);
        }
        else if constexpr (std::is_integral_v<T>) {
            if (hex)
                oss << "0x" << std::hex << v;
            else
                oss << v;
        }
        else {
            oss << v;
        }

        return oss.str();
    };

    // 预扫描，计算列宽
    int width = 0;
    for (int i = 0; i < pr; ++i) {
        for (int j = 0; j < pc; ++j) {
            width = std::max(width, static_cast<int>(format_elem(data[i * ld + j]).size()));
        }
    }
    width += 1;  // 留一点间距

    // 打印矩阵
    std::cout << name << " = \n";
    for (int i = 0; i < pr; ++i) {
        std::cout << name << "[ ";
        for (int j = 0; j < pc; ++j) {
            std::cout << std::setw(width) << std::right << format_elem(data[i * ld + j]);
        }
        std::cout << " ]\n";
    }

    // 截断提示
    if (rows > pr || cols > pc) {
        std::cout << name << " (showing "
                  << pr << "x" << pc
                  << " of " << rows << "x" << cols << ")\n";
    }
}

// GEMM kernel routine 的 ASCII 示意图 (Routine1/2/3)
inline const std::vector<std::string> routine_graphs = {
    R"(
        +--------------+      +--------------+  +--------------+
        |              |      |              |  |              |
        |              |      |              |  |              |
        |     MxN      | :+=  |     MxK      |  |     KxN      |   GEMM
        |              |      |              |  |              |
        |              |      |              |  |              |
        +--------------+      +--------------+  +--------------+

        +--------------+      +------+  +--------------+
        |              |      |      |  |     KCxN     |
        |              |      |      |  +--------------+
        |     MxN      | :+=  | MxKC |                             GEPP
        |              |      |      |
        |              |      |      |
        +--------------+      +------+

        +------+      +------+  +-----+
        |      |      |      |  |KCxNC|
        |      |      |      |  +-----+
        | MxNC | :+=  | MxKC |                                     GEPB
        |      |      |      |
        |      |      |      |
        +------+      +------+
    )",
    R"(
        +--------------+      +--------------+  +--------------+
        |              |      |              |  |              |
        |              |      |              |  |              |
        |     MxN      | :+=  |     MxK      |  |     KxN      |   GEMM
        |              |      |              |  |              |
        |              |      |              |  |              |
        +--------------+      +--------------+  +--------------+

        +--------------+      +------+  +--------------+
        |              |      |      |  |     KCxN     |
        |              |      |      |  +--------------+
        |     MxN      | :+=  | MxKC |                             GEPP
        |              |      |      |
        |              |      |      |
        +--------------+      +------+

        +--------------+      +------+  +--------------+
        |     MCxN     | :+=  |MCxKC |  |     KCxN     |           GEBP
        +--------------+      +------+  +--------------+

    )",
    R"(
        +--------------+      +--------------+  +--------------+
        |              |      |              |  |              |
        |              |      |              |  |              |
        |     MxN      | :+=  |     MxK      |  |     KxN      |   GEMM
        |              |      |              |  |              |
        |              |      |              |  |              |
        +--------------+      +--------------+  +--------------+

        +--------------+      +--------------+  +--------------+
        |     MCxN     | :+=  |     MCxK     |  |              |   GEPM
        +--------------+      +--------------+  |              |
                                                |     KxN      |
                                                |              |
                                                |              |
                                                +--------------+

        +------+      +--------------+  +------+
        |MCxNC | :+=  |     MCxK     |  |      |                   GEPDOT
        +------+      +--------------+  |      |
                                        | KxNC |
                                        |      |
                                        |      |
                                        +------+

    )"
};
