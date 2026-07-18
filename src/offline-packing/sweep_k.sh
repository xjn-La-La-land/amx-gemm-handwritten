#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# 频率锁定/解锁: 复用 scripts/freq.sh(取代原 make lockfreq/unlockfreq)
source "$REPO_ROOT/scripts/freq.sh"

BIN="${BIN:-$REPO_ROOT/build/gemm-offline}"
FREQ="${FREQ:-3000000}"
CORE="${CORE:-0}"
ROUNDS="${ROUNDS:-100000}"
K_START="${K_START:-64}"
K_END="${K_END:-4096}"
K_STEP="${K_STEP:-64}"
OUT="${OUT:-sweep_k_result.txt}"

TMP_OUT="$(mktemp /tmp/sweep_k_perf_XXXXXX.txt)"
ROW_FMT="%-8s %12s %18s %18s %16s\n"
SEP="$(printf '%0.s-' {1..80})"
K_VALUES=()
UTIL_VALUES=()
HIT_RATE_VALUES=()

cleanup() {
    rm -f "$TMP_OUT"
    freq_unlock
}
trap cleanup EXIT

parse_counter() {
    local event="$1"
    local file="$2"

    awk -v event="$event" '
        index($0, event) {
            value = $1
            gsub(/,/, "", value)
            if (value ~ /^[0-9]+$/)
                last = value
        }
        END {
            if (last != "")
                print last
        }
    ' "$file"
}

parse_utilization() {
    local file="$1"

    awk '
        function is_num(s) {
            return s ~ /^[0-9]+([.][0-9]+)?$/
        }

        {
            line = $0
            gsub(/[,=%()]/, " ", line)
            n = split(line, f)

            # Table row:
            #   M N K Time(s) Perf(TOPS) Util(%)
            if (n >= 6 && f[1] == "32" && f[2] == "32" && is_num(f[6]))
                util = f[6]

            # Log-style row:
            #   M N K = 32 32 1024, ..., Utilization = 96.82%
            for (i = 1; i < n; i++) {
                if (f[i] == "Utilization" && is_num(f[i + 1]))
                    util = f[i + 1]
            }
        }

        END {
            if (util != "")
                print util
        }
    ' "$file"
}

calc_hit_rate() {
    local hit="$1"
    local miss="$2"

    if [[ "$hit" =~ ^[0-9]+$ && "$miss" =~ ^[0-9]+$ ]]; then
        awk -v hit="$hit" -v miss="$miss" '
            BEGIN {
                total = hit + miss
                if (total > 0)
                    printf "%.2f", hit * 100.0 / total
                else
                    printf "N/A"
            }
        '
    else
        printf "N/A"
    fi
}

emit_line() {
    printf "%s\n" "$1" | tee -a "$OUT"
}

emit_row() {
    printf "$ROW_FMT" "$@" | tee -a "$OUT"
}

python_list_value() {
    local value="$1"

    if [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf "%s" "$value"
    else
        printf "None"
    fi
}

python_list() {
    local name="$1"
    shift
    local value
    local sep=""

    printf "%s = [" "$name"
    for value in "$@"; do
        printf "%s" "$sep"
        python_list_value "$value"
        sep=", "
    done
    printf "]\n"
}

append_python_lists() {
    {
        echo ""
        echo "# Python lists"
        python_list "K" "${K_VALUES[@]}"
        python_list "Util" "${UTIL_VALUES[@]}"
        python_list "L1D_Hit_Rate" "${HIT_RATE_VALUES[@]}"
    } >> "$OUT"
}

print_header() {
    : > "$OUT"
    emit_line "Sweep K: core=$CORE rounds=$ROUNDS K=$K_START..$K_END step=$K_STEP bin=$BIN"
    emit_line "$SEP"
    emit_row "K" "Util(%)" "L1D_HIT" "L1D_MISS" "L1D_HitRate(%)"
    emit_line "$SEP"
}

print_header

freq_lock "$FREQ"

for k in $(seq "$K_START" "$K_STEP" "$K_END"); do
    : > "$TMP_OUT"

    taskset -c "$CORE" sudo perf stat \
        -e cycles \
        -e instructions \
        -e MEM_LOAD_RETIRED.L1_HIT \
        -e MEM_LOAD_RETIRED.L1_MISS \
        "$BIN" \
            -r "$ROUNDS" \
            --core-list "$CORE" \
            --k "$k" \
        > "$TMP_OUT" 2>&1

    util="$(parse_utilization "$TMP_OUT")"
    l1_hit="$(parse_counter "MEM_LOAD_RETIRED.L1_HIT" "$TMP_OUT")"
    l1_miss="$(parse_counter "MEM_LOAD_RETIRED.L1_MISS" "$TMP_OUT")"

    util="${util:-N/A}"
    l1_hit="${l1_hit:-N/A}"
    l1_miss="${l1_miss:-N/A}"
    hit_rate="$(calc_hit_rate "$l1_hit" "$l1_miss")"

    K_VALUES+=("$k")
    UTIL_VALUES+=("$util")
    HIT_RATE_VALUES+=("$hit_rate")

    emit_row "$k" "$util" "$l1_hit" "$l1_miss" "$hit_rate"
done

emit_line "$SEP"
emit_line "Results saved to $OUT"
append_python_lists
