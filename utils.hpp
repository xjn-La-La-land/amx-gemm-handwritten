#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <cstring>  // for strerror
#include <cerrno>   // for errno
#include <fcntl.h>  // for open
#include <unistd.h> // for pread, pwrite, close
#include <sched.h>
#include <numa.h>       // libnuma
#include <system_error> // C++ 标准错误处理
#include <stdexcept>

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET3D(x, y, z, ld1, ld2) ((x) * (ld1) * (ld2) + (y) * (ld2) + (z))

#if !defined(likely)
#define likely(cond) __builtin_expect(cond, 1)
#define unlikely(cond) __builtin_expect(cond, 0)
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define ROUNDUP(x, y) (CEIL(x, y) * (y))
#define ROUNDDOWN(x, y) ((x) / (y) * (y))


namespace HWPFCtrl {

// 使用命名空间内的 constexpr 替代宏，类型更安全
namespace MSR {
    constexpr uint32_t MISC_FEATURE_CONTROL = 0x1A4;

    constexpr uint64_t DISABLE_L2           = (1ULL << 0);
    constexpr uint64_t DISABLE_L2_ADJACENT  = (1ULL << 1);
    constexpr uint64_t DISABLE_DCU          = (1ULL << 2);
    constexpr uint64_t DISABLE_DCU_IP       = (1ULL << 3);
    
    // 默认禁用的配置 (根据你的需求调整)
    constexpr uint64_t DEFAULT_DISABLE_MASK = DISABLE_L2 | DISABLE_L2_ADJACENT | DISABLE_DCU | DISABLE_DCU_IP;
}

class FileDescriptor {
    int fd_;
public:
    FileDescriptor(const std::string& path, int flags) {
        fd_ = ::open(path.c_str(), flags);
    }
    ~FileDescriptor() { // 利用 RAII 自动关闭文件描述符
        if (fd_ >= 0) ::close(fd_);
    }
    bool is_valid() const { return fd_ >= 0; }
    int get() const { return fd_; }
};

inline bool read_msr(int cpu, uint32_t msr, uint64_t& value) {
    std::string path = "/dev/cpu/" + std::to_string(cpu) + "/msr";
    FileDescriptor fd(path, O_RDONLY);

    if (!fd.is_valid()) {
        std::cerr << "[Error] Open MSR (read) CPU " << cpu << ": " << std::strerror(errno) << std::endl;
        return false;
    }

    if (::pread(fd.get(), &value, sizeof(value), msr) != sizeof(value)) {
        std::cerr << "[Error] Pread MSR CPU " << cpu << " reg 0x" << std::hex << msr << ": " 
                    << std::strerror(errno) << std::dec << std::endl;
        return false;
    }
    return true;
}

inline bool write_msr(int cpu, uint32_t msr, uint64_t value) {
    std::string path = "/dev/cpu/" + std::to_string(cpu) + "/msr";
    // 注意：有些系统需要 O_RDWR 才能写入，但在 Linux msr 驱动中 O_WRONLY 通常足够
    FileDescriptor fd(path, O_WRONLY); 

    if (!fd.is_valid()) {
        std::cerr << "[Error] Open MSR (write) CPU " << cpu << ": " << std::strerror(errno) << std::endl;
        return false;
    }

    if (::pwrite(fd.get(), &value, sizeof(value), msr) != sizeof(value)) {
        std::cerr << "[Error] Pwrite MSR CPU " << cpu << " reg 0x" << std::hex << msr << ": " 
                    << std::strerror(errno) << std::dec << std::endl;
        return false;
    }
    return true;
}


/**
 * @brief 禁用指定的硬件预取器
 * @param cpus CPU 核心 ID 列表
 * @param mask (可选) 要禁用的位掩码，默认为 L2 + L2 Adjacent
 * @return 成功返回 true，失败返回 false
 */
inline bool disable_prefetchers(const std::vector<int>& cpus, uint64_t mask = MSR::DEFAULT_DISABLE_MASK) {
    bool all_success = true;
    for (int cpu : cpus) {
        uint64_t val = 0;
        if (!read_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) { // 读取当前值
            all_success = false;
            continue;
        }

        uint64_t original = val;
        val |= mask; // 修改位 (将 mask 中的位设置为 1)

        // 3. 如果值有变化，则写回
        if (val != original) {
            if (!write_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
                all_success = false;
            }
        }
    }
    return all_success;
}

/**
 * @brief 恢复（启用）指定的硬件预取器
 * @param cpus CPU 核心 ID 列表
 * @param mask (可选) 要恢复的位掩码，默认为 L2 + L2 Adjacent
 * @return 成功返回 true，失败返回 false
 */
inline bool enable_prefetchers(const std::vector<int>& cpus, uint64_t mask = MSR::DEFAULT_DISABLE_MASK) {
    bool all_success = true;
    for (int cpu : cpus) {
        uint64_t val = 0;
        if (!read_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
            all_success = false;
            continue;
        }

        // 修改位 (设置 mask 中的位为 0)
        uint64_t original = val;
        val &= ~mask;

        if (val != original) {
            if (!write_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
                all_success = false;
            }
        }
    }
    return all_success;
}
}


/**
 * @brief 将当前线程绑定到指定的 CPU 核心
 * @param cpu_id 逻辑核心 ID
 * @throws std::system_error 如果绑定失败
 */
inline void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    // 使用 system_error 包装 errno，这是最标准的 C++ 系统错误抛出方式
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::system_error(errno, std::generic_category(), 
            "Failed to bind thread to CPU " + std::to_string(cpu_id));
    }
}

/**
 * @brief 检查并初始化 NUMA 环境
 * @param required_nodes 需要使用的 NUMA 节点数量
 * @throws std::runtime_error 如果 NUMA 不可用或节点数不足
 */
inline void init_numa(int required_nodes, std::vector<int> &core_list) {
    // 检查 NUMA 可用性
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
        if (numa_node_to_cpus(node_id, mask) != 0) { // 获取该 node 下的所有 CPU 到 mask 中
            numa_free_cpumask(mask);
            throw std::runtime_error("Failed to get CPUs for NUMA node " + std::to_string(node_id));
        }

        for (int cpu_id = 0; cpu_id < mask->size; ++cpu_id) {
            if (numa_bitmask_isbitset(mask, cpu_id)) {
                core_list.push_back(cpu_id);
            }
        }
    }
    numa_free_cpumask(mask);
}


/**
 * @brief 将当前线程绑定到指定 NUMA 节点的所有核心上
 * @note 使用 libnuma 查询真实的 CPU 拓扑，而不是假设 ID 是连续的
 */
inline void bind_thread_to_numa_node(int node_id) {
    struct bitmask* mask = numa_allocate_cpumask();
    
    // 获取该 NUMA 节点实际包含哪些 CPU (解决核心 ID 不连续的问题)
    if (numa_node_to_cpus(node_id, mask) != 0) {
        numa_free_cpumask(mask);
        throw std::system_error(errno, std::generic_category(), 
            "Failed to query CPUs for NUMA node " + std::to_string(node_id));
    }

    // 将 bitmask 转换为 cpu_set_t
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (unsigned int i = 0; i < mask->size; i++) {
        if (numa_bitmask_isbitset(mask, i)) {
            CPU_SET(i, &cpuset);
        }
    }

    // 绑定
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        numa_free_cpumask(mask);
        throw std::system_error(errno, std::generic_category(), 
            "Failed to bind thread to NUMA node " + std::to_string(node_id));
    }

    numa_free_cpumask(mask);
}


#include <type_traits>
#include <iomanip>
#include <sstream>
#include <algorithm>

// Matrix data output format
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

    // 将元素转为字符串（统一处理格式问题）
    auto format_elem = [&](T v) {
        std::ostringstream oss;

        if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(2) << v;
        } 
        else if constexpr (std::is_same_v<T, int8_t>) {
            if (hex)
                oss << "0x" << std::hex << static_cast<int>(v);
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