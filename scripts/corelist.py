#!/usr/bin/env python3
"""Parse a Linux CPU list and print its expanded form."""

import argparse


def parse_cpulist(spec: str) -> set[int]:
    cpus: set[int] = set()
    try:
        for item in (spec.split(",") if spec else ()):
            first, separator, last = item.partition("-")
            first, last = int(first), int(last) if separator else int(first)
            if first > last:
                raise ValueError
            cpus.update(range(first, last + 1))
    except ValueError as error:
        raise ValueError(f"invalid CPU list: {spec!r}") from error
    return cpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpus", help="CPU list, e.g. 0-3,8")
    args = parser.parse_args()
    try:
        cpus = parse_cpulist(args.cpus)
        if not cpus:
            raise ValueError("CPU list must not be empty")
    except ValueError as error:
        parser.error(str(error))
    print(",".join(map(str, sorted(cpus))))


if __name__ == "__main__":
    main()
