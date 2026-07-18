cmake_minimum_required(VERSION 3.16)
project(amx-gemm LANGUAGES CXX)

# ---------------------------------------------------------------------------
# 顶层公共配置。子目录(offline-packing / online-packing)各自定义可执行文件，
# 并链接下面的 INTERFACE 库 amx_common 继承所有公共编译/链接设置。
# 编译语义忠实复刻原 Makefile: -O3 -march=native -ffast-math -funroll-loops
#   -g -DNDEBUG -Wall -Wextra -fno-strict-aliasing -flto=auto, 链接 -lnuma。
# ---------------------------------------------------------------------------

# 禁止 in-source 构建，保持源码树干净。用法: cmake -B build && cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Use: cmake -B build")
endif()

# 给 clangd/LazyVim 生成 compile_commands.json(本项目引入 CMake 的主要动机之一)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 可执行文件直接输出到 build/ 根，不按 variant 建子目录(每个 variant 仅一个可执行文件)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# C++ 标准: 代码使用 designated initializers 和 [[unlikely]](均 C++20)。
# 保留 GNU 扩展(gnu++20)以匹配 __attribute__ / __builtin_expect / #pragma GCC。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# 目标架构。默认 native(Sapphire Rapids)，跨机构建时可覆盖，如 -DAMX_ARCH=sapphirerapids
set(AMX_ARCH "native" CACHE STRING "Value passed to -march= (e.g. native, sapphirerapids)")

# ---------------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)   # std::thread 的真实依赖(原 Makefile -fopenmp 顺带链接的 pthread)

# libnuma 没有官方 CMake 包，手动定位并封装成 imported target NUMA::NUMA
find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARY numa)
if(NOT NUMA_INCLUDE_DIR OR NOT NUMA_LIBRARY)
  message(FATAL_ERROR "libnuma not found. Install it: sudo apt install libnuma-dev")
endif()
add_library(NUMA::NUMA UNKNOWN IMPORTED)
set_target_properties(NUMA::NUMA PROPERTIES
  IMPORTED_LOCATION "${NUMA_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# amx_common: 承载所有公共编译/链接设置的 INTERFACE 库
# ---------------------------------------------------------------------------
add_library(amx_common INTERFACE)

# 公共头(utils.hpp / bench_harness.hpp)在 src/common，第三方库(CLI11.hpp)在 third_party。
# 统一暴露到 include 路径，变体源码即可直接 #include "utils.hpp" 等(无需 ../)。
target_include_directories(amx_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_compile_options(amx_common INTERFACE
  -O3 -march=${AMX_ARCH}
  -ffast-math -funroll-loops       # 数学与循环优化(注: -ffast-math 对 int8 GEMM 无实际影响，保留以对齐原 flags)
  -g -Wall -Wextra                 # 调试符号 + 警告(不影响性能，方便 perf/vtune)
  -fno-strict-aliasing
  -flto=auto)                      # 并行 LTO
target_link_options(amx_common INTERFACE -flto=auto)

target_compile_definitions(amx_common INTERFACE NDEBUG)  # 禁用 assert，对齐原 -DNDEBUG

target_link_libraries(amx_common INTERFACE NUMA::NUMA Threads::Threads)

# ---------------------------------------------------------------------------
# 子目录(后续步骤逐个加入其 CMakeLists.txt；未就绪的先跳过，保证本文件可独立配置)
# 注意: 两个可执行文件不能都叫 gemm-test，统一构建时需区分命名(如 gemm-offline/gemm-online)。
# ---------------------------------------------------------------------------
foreach(variant offline-packing online-packing)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${variant}/CMakeLists.txt")
    add_subdirectory(src/${variant})
  else()
    message(STATUS "Skipping '${variant}': no CMakeLists.txt yet (added in a later step).")
  endif()
endforeach()

# 在源码根放一个指向构建目录 compile_commands.json 的符号链接，方便 clangd 发现。
# (记得把 /compile_commands.json 加入 .gitignore)
file(CREATE_LINK
  "${CMAKE_BINARY_DIR}/compile_commands.json"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json"
  SYMBOLIC)
