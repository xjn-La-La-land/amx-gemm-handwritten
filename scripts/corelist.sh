#!/usr/bin/env bash
# corelist.sh —— 把核心规格展开成列表或计数。
# 规格语法(与原 Makefile CORE 变量一致): "0" | "0,1,2" | "0-29" | "0-3,8,12-15"
#
# 既可被 source 使用其函数, 也可作为 CLI:
#   corelist.sh expand 0-3,8   ->  0,1,2,3,8
#   corelist.sh count  0-29    ->  30

# 展开为逗号分隔列表: "0-3,8" -> "0,1,2,3,8"
corelist_expand() {
    local spec="$1" part a b i
    local -a parts=() out=()
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            a="${part%-*}"; b="${part#*-}"
            for ((i = a; i <= b; i++)); do out+=("$i"); done
        else
            out+=("$part")
        fi
    done
    ( IFS=','; echo "${out[*]}" )
}

# 统计核心数量: "0-29" -> 30
corelist_count() {
    local expanded
    expanded="$(corelist_expand "$1")"
    awk -F',' '{ print NF }' <<< "$expanded"
}

# 仅当作为脚本直接执行时才走 CLI 分支(被 source 时跳过)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        expand) corelist_expand "$2" ;;
        count)  corelist_count  "$2" ;;
        *) echo "usage: $0 {expand|count} <core-spec>" >&2; exit 1 ;;
    esac
fi
