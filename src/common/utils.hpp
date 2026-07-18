#pragma once
// umbrella header: 汇总 common 层的公共工具，方便一次性引入。
// 各功能已按职责拆分到独立头文件，需要精确依赖时可直接 include 对应的：
//   - numeric.hpp      OFFSET2D 宏 + amx::min/max/ceil_div/round_up/round_down
//   - hw_prefetch.hpp  namespace HWPFCtrl: MSR 读写 + 硬件预取器开关
//   - cpu_affinity.hpp bind_thread_to_cpu / init_numa
//   - debug_print.hpp  print_matrix + routine_graphs

#include "numeric.hpp"
#include "hw_prefetch.hpp"
#include "cpu_affinity.hpp"
#include "debug_print.hpp"
