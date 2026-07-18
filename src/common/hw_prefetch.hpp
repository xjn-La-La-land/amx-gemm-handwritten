#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>  // strerror
#include <cerrno>   // errno
#include <fcntl.h>  // open
#include <unistd.h> // pread, pwrite, close

// 硬件预取器控制: 通过读写 MSR 0x1A4 (MISC_FEATURE_CONTROL) 开关 L1/L2 硬件预取器。
// 用途: 性能测试时禁用硬件预取器，隔离软件预取/blocking 的效果。需要 root + msr 内核模块。
namespace HWPFCtrl {

// 命名空间内的 constexpr 替代宏，类型更安全
namespace MSR {
    constexpr uint32_t MISC_FEATURE_CONTROL = 0x1A4;

    constexpr uint64_t DISABLE_L2           = (1ULL << 0);
    constexpr uint64_t DISABLE_L2_ADJACENT  = (1ULL << 1);
    constexpr uint64_t DISABLE_DCU          = (1ULL << 2);
    constexpr uint64_t DISABLE_DCU_IP       = (1ULL << 3);

    // 默认禁用全部四个预取器 (L2 + L2 Adjacent + DCU + DCU IP)
    constexpr uint64_t DEFAULT_DISABLE_MASK = DISABLE_L2 | DISABLE_L2_ADJACENT | DISABLE_DCU | DISABLE_DCU_IP;
}

// RAII 文件描述符。禁用拷贝(避免 double-close)；允许移动。
class FileDescriptor {
    int fd_;
public:
    FileDescriptor(const std::string& path, int flags) {
        fd_ = ::open(path.c_str(), flags);
    }
    ~FileDescriptor() { // RAII 自动关闭
        if (fd_ >= 0) ::close(fd_);
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
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
    // 注意: 有些系统需要 O_RDWR 才能写入，但 Linux msr 驱动 O_WRONLY 通常足够
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
 * @brief 禁用指定 CPU 的硬件预取器
 * @param cpus CPU 核心 ID 列表
 * @param mask (可选) 要禁用的位掩码，默认禁用全部四个预取器
 * @return 全部成功返回 true
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
        val |= mask; // 置位 = 禁用

        if (val != original) { // 有变化才写回
            if (!write_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
                all_success = false;
            }
        }
    }
    return all_success;
}

/**
 * @brief 恢复(启用)指定 CPU 的硬件预取器
 * @param cpus CPU 核心 ID 列表
 * @param mask (可选) 要恢复的位掩码，默认恢复全部四个预取器
 * @return 全部成功返回 true
 */
inline bool enable_prefetchers(const std::vector<int>& cpus, uint64_t mask = MSR::DEFAULT_DISABLE_MASK) {
    bool all_success = true;
    for (int cpu : cpus) {
        uint64_t val = 0;
        if (!read_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
            all_success = false;
            continue;
        }

        uint64_t original = val;
        val &= ~mask; // 清位 = 启用

        if (val != original) {
            if (!write_msr(cpu, MSR::MISC_FEATURE_CONTROL, val)) {
                all_success = false;
            }
        }
    }
    return all_success;
}

} // namespace HWPFCtrl
