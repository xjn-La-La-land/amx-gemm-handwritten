#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>

namespace amx {

// Process-wide allocation policy for packed matrix buffers.
enum class BufferAllocationMode : uint8_t {
    Auto,
    Regular,
    HugePage,
};

inline const char* buffer_allocation_mode_str(BufferAllocationMode mode) {
    switch (mode) {
    case BufferAllocationMode::Auto:     return "auto";
    case BufferAllocationMode::Regular:  return "regular";
    case BufferAllocationMode::HugePage: return "huge";
    }
    return "unknown";
}

inline BufferAllocationMode parse_buffer_allocation_mode(const std::string& value) {
    if (value == "auto")    return BufferAllocationMode::Auto;
    if (value == "regular") return BufferAllocationMode::Regular;
    if (value == "huge")    return BufferAllocationMode::HugePage;
    throw std::invalid_argument("invalid buffer allocation mode: " + value);
}

namespace page_alloc {

inline constexpr size_t HUGE_PAGE_SIZE = 2u * 1024u * 1024u; // 2MB
inline constexpr size_t CACHE_LINE_BYTES = 64u;

// First size that exceeds the 64-entry 4KB load DTLB while limiting
// 2MB rounding amplification to 4x.
inline constexpr size_t AUTO_HUGE_PAGE_MIN_BYTES = 512u * 1024u;

class Allocation;
namespace detail {
inline BufferAllocationMode configured_mode = BufferAllocationMode::Auto;
Allocation allocate_regular(size_t bytes);
Allocation try_allocate_huge_page(size_t bytes);
} // namespace detail

// Configure once before creating buffers or starting worker threads.
inline void set_mode(BufferAllocationMode mode) {
    detail::configured_mode = mode;
}

inline BufferAllocationMode mode() {
    return detail::configured_mode;
}

// Move-only owner for either aligned_alloc or mmap-backed memory.
class Allocation {
public:
    Allocation() = default;

    void* data() const { return owner_ ? data_ : nullptr; }
    bool valid() const { return static_cast<bool>(owner_); }

private:
    friend Allocation detail::allocate_regular(size_t);
    friend Allocation detail::try_allocate_huge_page(size_t);

    struct Releaser {
        size_t mmap_bytes = 0;

        void operator()(void* p) const noexcept {
            if (!p) return;
            if (mmap_bytes) {
                ::munmap(p, mmap_bytes);
            } else {
                std::free(p);
            }
        }
    };

    Allocation(void* data, void* release_base, size_t mmap_bytes)
        : owner_(release_base, Releaser{mmap_bytes}), data_(data) {}

    std::unique_ptr<void, Releaser> owner_{nullptr, Releaser{}};
    void* data_ = nullptr;
};

namespace detail {

inline size_t round_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline bool auto_selects_huge_page(size_t bytes) {
    return bytes >= AUTO_HUGE_PAGE_MIN_BYTES;
}

// Allocate cache-line-aligned memory through the C allocator.
inline Allocation allocate_regular(size_t bytes) {
    const size_t alloc_bytes = round_up(bytes, CACHE_LINE_BYTES);
    void* raw = std::aligned_alloc(CACHE_LINE_BYTES, alloc_bytes);
    if (!raw) throw std::bad_alloc();
    return Allocation(raw, raw, 0);
}

// Reserve a mapping containing a complete 2MB-aligned advised region:
//
// mmap reservation
// mapping              aligned_addr                              mapping + reserve_bytes
//   |------ prefix ------|<--------- advised_bytes --------->|------ suffix ------|
//                        |-- data_offset --> data
//
// Return an empty Allocation when mmap or madvise fails.
inline Allocation try_allocate_huge_page(size_t bytes) {
    Allocation allocation;
    const size_t advised_bytes = round_up(bytes, HUGE_PAGE_SIZE);

    // Reserve one extra huge page virtually so an aligned PMD-sized subrange is
    // always available. Only the advised/touched subrange consumes physical RAM.
    const size_t reserve_bytes = advised_bytes + HUGE_PAGE_SIZE;
    void* mapping = ::mmap(nullptr, reserve_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return allocation;

    const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(mapping);
    auto* aligned_addr = reinterpret_cast<uint8_t*>(round_up(raw_addr, HUGE_PAGE_SIZE));

    if (::madvise(aligned_addr, advised_bytes, MADV_HUGEPAGE) != 0) {
        ::munmap(mapping, reserve_bytes);
        return allocation;
    }

    // A 2MB-aligned data pointer makes every buffer start at identical L1/L2
    // cache sets. Keep the mapping PMD-aligned, but rotate the 64B-aligned data
    // offset across the 128KB L2 set-index span when the allocation has slack.
    static std::atomic<size_t> color_ticket{0};
    constexpr size_t L2_SET_SPAN = 128u * 1024u;
    constexpr size_t COLOR_STRIDE = 17u * CACHE_LINE_BYTES;
    const size_t ticket = color_ticket.fetch_add(1, std::memory_order_relaxed) + 1;
    size_t data_offset = (ticket * COLOR_STRIDE) & (L2_SET_SPAN - CACHE_LINE_BYTES);
    if (data_offset > advised_bytes - bytes) data_offset = 0;

    return Allocation(aligned_addr + data_offset, mapping, reserve_bytes);
}

} // namespace detail

// Request a MADV_HUGEPAGE allocation and fail instead of falling back.
inline Allocation allocate_huge_page(size_t bytes) {
    Allocation allocation = detail::try_allocate_huge_page(bytes);
    if (!allocation.valid())
        throw std::runtime_error("failed to create MADV_HUGEPAGE allocation");
    return allocation;
}

// Allocate using the configured policy; Auto falls back to regular allocation.
inline Allocation allocate(size_t bytes) {
    const BufferAllocationMode allocation_mode = mode();
    if (allocation_mode == BufferAllocationMode::Regular)
        return detail::allocate_regular(bytes);

    const bool use_huge_page = allocation_mode == BufferAllocationMode::HugePage ||
                               (allocation_mode == BufferAllocationMode::Auto &&
                                detail::auto_selects_huge_page(bytes));
    if (use_huge_page) {
        Allocation allocation = detail::try_allocate_huge_page(bytes);
        if (allocation.valid()) return allocation;
        if (allocation_mode == BufferAllocationMode::HugePage)
            throw std::runtime_error("failed to create MADV_HUGEPAGE allocation");
    }
    return detail::allocate_regular(bytes);
}

} // namespace page_alloc
} // namespace amx
