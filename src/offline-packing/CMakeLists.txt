# offline-packing 变体: 计时前一次性预打包整个 A/B/C。
# 公共编译/链接设置(flags、C++ 标准、include 路径、numa/pthread)全部来自
# 顶层 INTERFACE 库 amx_common。
add_executable(gemm-offline
  amx-gemm.cpp
  gemm-test.cpp)

target_link_libraries(gemm-offline PRIVATE amx_common)
