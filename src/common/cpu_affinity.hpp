#pragma once
#include <vector>
#include <string>
#include <sched.h>
#include <numa.h>       // libnuma
#include <system_error>
#include <stdexcept>

// CPU 亲和性与 NUMA 环境初始化。

/**
 * @brief 将当前线程绑定到指定的 CPU 核心
 * @param cpu_id 逻辑核心 ID
 * @throws std::system_error 如果绑定失败
 */
inline void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    // 用 system_error 包装 errno，是最标准的 C++ 系统错误抛出方式
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::system_error(errno, std::generic_category(),
            "Failed to bind thread to CPU " + std::to_string(cpu_id));
    }
}

/**
 * @brief 检查并初始化 NUMA 环境，收集前 required_nodes 个节点的核心到 core_list
 * @param required_nodes 需要使用的 NUMA 节点数量
 * @param core_list [out] 被清空并填入这些节点包含的所有 CPU ID
 * @throws std::runtime_error / std::range_error 如果 NUMA 不可用或节点数不足
 */
inline void init_numa(int required_nodes, std::vector<int> &core_list) {
    if (numa_available() < 0) {
        throw std::runtime_error("NUMA support is not available on this system.");
    }

    int available_nodes = numa_max_node() + 1;

    if (required_nodes > available_nodes) {
        throw std::range_error(
            "Requested NUMA nodes (" + std::to_string(required_nodes) +
            ") exceed system availability (" + std::to_string(available_nodes) + ")."
        );
    }

    core_list.clear();
    struct bitmask* mask = numa_allocate_cpumask();
    // 遍历 0 ... required_nodes - 1 NUMA Node, 获取每个节点的 CPU 列表
    for (int node_id = 0; node_id < required_nodes; ++node_id) {
        if (numa_node_to_cpus(node_id, mask) != 0) {
            numa_free_cpumask(mask);
            throw std::runtime_error("Failed to get CPUs for NUMA node " + std::to_string(node_id));
        }

        for (unsigned long cpu_id = 0; cpu_id < mask->size; ++cpu_id) {
            if (numa_bitmask_isbitset(mask, cpu_id)) {
                core_list.push_back(static_cast<int>(cpu_id));
            }
        }
    }
    numa_free_cpumask(mask);
}
