#!/usr/bin/env python3
"""toml_get.py <file> <key> [--lines] —— 从 TOML 取顶层键值, 供 bench.sh 读取编排参数。
   标量原样打印; 数组默认逗号分隔(适配 taskset / --core-list),
   带 --lines 则每元素一行(适配含逗号的值, 如 perf 裸事件, 由 readarray 读取)。
   键不存在则打印空串并退出 0(由调用方判断是否必需)。"""
import sys
import tomllib

args = [a for a in sys.argv[1:] if a != "--lines"]
lines = "--lines" in sys.argv
if len(args) != 2:
    print("usage: toml_get.py <file> <key> [--lines]", file=sys.stderr)
    sys.exit(2)

path, key = args
try:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
except (OSError, tomllib.TOMLDecodeError) as e:
    print(f"toml_get: {e}", file=sys.stderr)
    sys.exit(1)

v = cfg.get(key)
if v is None:
    pass                                          # 缺键 → 空输出
elif isinstance(v, list):
    sep = "\n" if lines else ","
    print(sep.join(str(x) for x in v))            # 数组 → 逐行 或 0,1,2
else:
    print(v)                                      # 标量/字符串原样
