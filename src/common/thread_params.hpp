#pragma once
#include <vector>

namespace amx {

// 多线程 GEMM 的线程/NUMA 配置。
struct ThreadParams {
    std::vector<int> core_list = {0};
    bool numa_aware = false;
    int num_numa_node = 1;
};

} // namespace amx
