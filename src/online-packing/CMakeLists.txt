# online-packing 变体: 打包融合进计算循环，按 M/N/K 自动分派 routine。
# 公共编译/链接设置(flags、C++ 标准、include 路径、numa/pthread)全部来自
# 顶层 INTERFACE 库 amx_common。
add_executable(gemm-online
  amx-gemm.cpp
  gemm-test.cpp)

target_link_libraries(gemm-online PRIVATE amx_common)
