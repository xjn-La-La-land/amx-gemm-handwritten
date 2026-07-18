#!/usr/bin/env bash
# freq.sh —— CPU 频率锁定/恢复(从原 Makefile lockfreq/unlockfreq 分离而来)。
# 纯系统状态操作, 与 build 无关。需要 sudo(cpupower)。
#
# 恢复范围与 governor 是实验机专属, 可用环境变量覆盖:
#   FREQ_RESTORE_MIN (默认 1.0GHz)
#   FREQ_RESTORE_MAX (默认 5.0GHz)
#   FREQ_RESTORE_GOV (默认 powersave)
# 锁定用的 governor:
#   FREQ_LOCK_GOV    (默认 performance)
#
# CLI:
#   freq.sh lock <khz>   # 锁定所有核心到 <khz> kHz
#   freq.sh unlock       # 恢复自动调频
#   e.g.: freq.sh lock 3000000

: "${FREQ_RESTORE_MIN:=1.0GHz}"
: "${FREQ_RESTORE_MAX:=5.0GHz}"
: "${FREQ_RESTORE_GOV:=powersave}"
: "${FREQ_LOCK_GOV:=performance}"

# 诊断输出走 stderr, 避免污染可能被管道消费的 stdout
freq_lock() {
    local khz="$1"
    if [[ -z "$khz" ]]; then echo "freq_lock: missing frequency (kHz)" >&2; return 2; fi
    echo "🔒 Locking CPU frequency to ${khz} kHz..." >&2
    sudo cpupower frequency-set -g "$FREQ_LOCK_GOV" > /dev/null
    sudo cpupower frequency-set -d "$khz" -u "$khz" > /dev/null
    echo "✅ CPU frequency locked to ${khz} kHz" >&2
}

freq_unlock() {
    echo "🔓 Restoring CPU frequency policy..." >&2
    sudo cpupower frequency-set -d "$FREQ_RESTORE_MIN" -u "$FREQ_RESTORE_MAX" > /dev/null
    sudo cpupower frequency-set -g "$FREQ_RESTORE_GOV" > /dev/null
    echo "✅ CPU frequency restored (${FREQ_RESTORE_MIN}..${FREQ_RESTORE_MAX}, ${FREQ_RESTORE_GOV})" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        lock)   freq_lock "$2" ;;
        unlock) freq_unlock ;;
        *) echo "usage: $0 {lock <khz>|unlock}" >&2; exit 1 ;;
    esac
fi
