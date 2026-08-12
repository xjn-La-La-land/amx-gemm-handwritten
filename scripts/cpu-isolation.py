#!/usr/bin/env python3
"""Strict, reversible CPU isolation for handwritten-v2 benchmarks.

Runtime isolation uses a cgroup-v2 isolated partition, IRQ affinity changes,
irqbalance suspension, and SMT sibling hotplug.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from corelist import parse_cpulist


SLICE = "benchmark.slice"
CGROUP_ROOT = Path("/sys/fs/cgroup")
SLICE_PATH = CGROUP_ROOT / SLICE
CPU_ROOT = Path("/sys/devices/system/cpu")
STATE_PATH = Path("/run/handwritten-v2-cpu-isolation.json")


class IsolationError(RuntimeError):
    pass


def read(path: Path) -> str:
    return path.read_text().strip()


def write(path: Path, value: str) -> None:
    path.write_text(value + "\n")


@dataclass(frozen=True)
class Plan:
    target: set[int]
    physical: set[int]
    offline: set[int]
    housekeeping: set[int]

    @classmethod
    def build(cls, spec: str) -> "Plan":
        target = parse_cpulist(spec)
        online = parse_cpulist(read(CPU_ROOT / "online"))
        if not target:
            raise IsolationError("CPU list must not be empty")
        if not target <= online:
            raise IsolationError(f"requested CPUs are not all online (online: {format_cpulist(online)})")

        physical: set[int] = set()
        for cpu in target:
            path = CPU_ROOT / f"cpu{cpu}/topology/thread_siblings_list"
            if not path.is_file():
                raise IsolationError(f"CPU {cpu} topology is unavailable")
            siblings = parse_cpulist(read(path))
            if len(target & siblings) != 1:
                raise IsolationError(
                    f"select only one hardware thread from physical core "
                    f"{{{format_cpulist(siblings)}}}"
                )
            physical |= siblings

        if 0 in physical:
            raise IsolationError("the physical core containing CPU 0 cannot be strictly isolated")
        housekeeping = online - physical
        if not housekeeping:
            raise IsolationError("at least one housekeeping CPU is required")
        return cls(target, physical, physical - target, housekeeping)

def format_cpulist(cpus: set[int]) -> str:
    runs: list[list[int]] = []
    for cpu in sorted(cpus):
        if not runs or cpu != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(cpu)
    return ",".join(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs)


def run(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def slice_properties() -> dict[str, str]:
    names = ("ActiveState", "FragmentPath", "DropInPaths", "AllowedCPUs")
    output = run("systemctl", "show", SLICE, *(f"--property={name}" for name in names)).stdout
    return dict(line.split("=", 1) for line in output.splitlines())


def require_root() -> None:
    if os.geteuid() != 0:
        raise IsolationError("this command must run as root")


def save_state(state: dict) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    write(temporary, json.dumps(state, indent=2, sort_keys=True))
    temporary.chmod(0o644)
    temporary.replace(STATE_PATH)


def load_state() -> dict:
    try:
        return json.loads(read(STATE_PATH))
    except (OSError, json.JSONDecodeError) as error:
        raise IsolationError(f"cannot read isolation state: {error}") from error


def configure_partition(plan: Plan, state: dict) -> None:
    if slice_properties() != {
        "ActiveState": "inactive",
        "FragmentPath": "",
        "DropInPaths": "",
        "AllowedCPUs": "",
    }:
        raise IsolationError(f"{SLICE} already has state or configuration; refusing to overwrite it")

    state["slice_configured"] = True
    save_state(state)
    run("systemctl", "set-property", "--runtime", SLICE, f"AllowedCPUs={format_cpulist(plan.physical)}")
    run("systemctl", "start", SLICE)

    partition = SLICE_PATH / "cpuset.cpus.partition"
    if not partition.exists():
        raise IsolationError(f"systemd did not create a cpuset-enabled {SLICE}")
    write(partition, "isolated")
    if read(partition) != "isolated":
        raise IsolationError(f"{SLICE} did not become a valid isolated partition")
    isolated = parse_cpulist(read(CGROUP_ROOT / "cpuset.cpus.isolated"))
    if not plan.physical <= isolated:
        raise IsolationError("kernel did not grant the requested exclusive CPUs")


def redirect_irqs(plan: Plan, state: dict) -> None:
    irqbalance_active = run("systemctl", "is-active", "--quiet", "irqbalance", check=False).returncode == 0
    state["irqbalance_was_active"] = irqbalance_active
    if irqbalance_active:
        save_state(state)
        run("systemctl", "stop", "irqbalance")

    changes: dict[str, str] = {}
    for affinity in Path("/proc/irq").glob("[0-9]*/smp_affinity_list"):
        original = read(affinity)
        if parse_cpulist(original) & plan.physical:
            changes[affinity.parent.name] = original
    state["irq_affinities"] = changes
    save_state(state)

    for irq, original in changes.items():
        irq_dir = Path("/proc/irq") / irq
        affinity_path = irq_dir / "smp_affinity_list"
        allowed = parse_cpulist(original) - plan.physical or plan.housekeeping
        try:
            write(affinity_path, format_cpulist(allowed))
        except OSError as error:
            if read(affinity_path) != original:
                raise IsolationError(f"IRQ {irq} affinity changed after write failed: {error}") from error
            effective_path = irq_dir / "effective_affinity_list"
            effective = parse_cpulist(read(effective_path)) if effective_path.is_file() else set()
            if effective & plan.physical:
                print(
                    f"cpu-isolation: warning: IRQ {irq} cannot be moved from isolated CPUs "
                    f"{format_cpulist(effective)}; skipping ({error})",
                    file=sys.stderr,
                )
            continue
        effective_path = irq_dir / "effective_affinity_list"
        if not effective_path.is_file():
            raise IsolationError(f"cannot verify IRQ {irq} effective affinity")
        effective = parse_cpulist(read(effective_path))
        if effective & plan.physical:
            raise IsolationError(
                f"IRQ {irq} remains effective on isolated CPUs ({format_cpulist(effective)})"
            )


def offline_siblings(plan: Plan, state: dict) -> None:
    offlined = [
        cpu
        for cpu in sorted(plan.offline)
        if read(CPU_ROOT / f"cpu{cpu}/online") == "1"
    ]
    state["offlined_cpus"] = offlined
    save_state(state)
    for cpu in offlined:
        write(CPU_ROOT / f"cpu{cpu}/online", "0")


def restore_file(path: Path, value: str, failures: list[str], label: str) -> None:
    try:
        write(path, value)
    except OSError as error:
        try:
            if read(path) == value:
                return
        except OSError:
            pass
        failures.append(f"{label}: {error}")


def release_state(state: dict) -> bool:
    failures: list[str] = []

    if state.get("slice_configured"):
        run("systemctl", "stop", SLICE, check=False, quiet=True)
        partition = SLICE_PATH / "cpuset.cpus.partition"
        if partition.exists():
            restore_file(partition, "member", failures, "partition")
        for command in (
            ("systemctl", "set-property", "--runtime", SLICE, "AllowedCPUs="),
            ("systemctl", "revert", SLICE),
            ("systemctl", "reset-failed", SLICE),
        ):
            if run(*command, check=False, quiet=True).returncode and command[1] == "set-property":
                failures.append("could not clear slice AllowedCPUs")

    for cpu in state.get("offlined_cpus", []):
        restore_file(CPU_ROOT / f"cpu{cpu}/online", "1", failures, f"CPU {cpu}")

    for irq, original in state.get("irq_affinities", {}).items():
        restore_file(Path("/proc/irq") / irq / "smp_affinity_list", original, failures, f"IRQ {irq}")

    if state.get("irqbalance_was_active"):
        if run("systemctl", "start", "irqbalance", check=False, quiet=True).returncode:
            failures.append("could not restart irqbalance")

    if failures:
        print("cpu-isolation: release incomplete: " + "; ".join(failures), file=sys.stderr)
        return False
    STATE_PATH.unlink(missing_ok=True)
    return True


def acquire(spec: str) -> None:
    require_root()
    if run("stat", "-fc", "%T", str(CGROUP_ROOT)).stdout.strip() != "cgroup2fs":
        raise IsolationError(f"{CGROUP_ROOT} is not a cgroup-v2 mount")
    if "cpuset" not in read(CGROUP_ROOT / "cgroup.controllers").split():
        raise IsolationError("the cgroup-v2 cpuset controller is unavailable")
    if STATE_PATH.exists():
        raise IsolationError(f"isolation state already exists; run: sudo {sys.argv[0]} release")

    plan = Plan.build(spec)
    state = {
        "target_cpus": sorted(plan.target),
        "physical_cpus": sorted(plan.physical),
    }
    save_state(state)
    try:
        configure_partition(plan, state)
        redirect_irqs(plan, state)
        offline_siblings(plan, state)
    except BaseException:
        print("cpu-isolation: warning: acquire failed; restoring partial changes", file=sys.stderr)
        if not release_state(state):
            print(f"cpu-isolation: run manually: sudo {sys.argv[0]} release", file=sys.stderr)
        raise
    print(
        f"cpu-isolation: isolated physical CPUs {format_cpulist(plan.physical)}; "
        f"benchmark CPUs {format_cpulist(plan.target)}",
        file=sys.stderr,
    )


def release() -> None:
    require_root()
    if not STATE_PATH.exists():
        print("cpu-isolation: no active isolation", file=sys.stderr)
        return
    if not release_state(load_state()):
        raise IsolationError(f"runtime isolation was not fully released; state kept in {STATE_PATH}")
    print("cpu-isolation: runtime CPU isolation released", file=sys.stderr)


def status() -> None:
    state = load_state() if STATE_PATH.exists() else {}
    partition_path = SLICE_PATH / "cpuset.cpus.partition"
    partition = read(partition_path) if partition_path.is_file() else "inactive"
    lifecycle = "active" if partition == "isolated" else "partial" if state else "inactive"
    print(f"state     : {lifecycle}")
    print(f"partition : {partition}")
    print(f"target    : {format_cpulist(set(state.get('target_cpus', [])))}")
    print(f"physical  : {format_cpulist(set(state.get('physical_cpus', [])))}")
    print(f"isolated  : {read(CGROUP_ROOT / 'cpuset.cpus.isolated')}")


def print_plan(spec: str) -> None:
    plan = Plan.build(spec)
    print(f"benchmark CPUs : {format_cpulist(plan.target)}")
    print(f"physical CPUs  : {format_cpulist(plan.physical)}")
    print(f"SMT to offline : {format_cpulist(plan.offline)}")
    print(f"housekeeping   : {format_cpulist(plan.housekeeping)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  cpu-isolation.py plan 15
  sudo cpu-isolation.py acquire 15
  sudo cpu-isolation.py release""",
    )
    subparsers = parser.add_subparsers(required=True)
    commands = (
        ("plan", print_plan, "Show CPU topology and planned runtime changes.", True),
        ("acquire", acquire, "Create runtime CPU isolation; requires root.", True),
        ("release", release, "Restore the recorded runtime state; requires root.", False),
        ("status", status, "Show the current isolation state without changing it.", False),
    )
    for name, action, summary, takes_cpus in commands:
        command = subparsers.add_parser(name, help=summary, description=summary)
        if takes_cpus:
            command.add_argument(
                "cpus",
                help="logical CPU list; select one SMT thread per physical core, e.g. 15 or 1-3,8",
            )
        command.set_defaults(action=action)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        arguments = (args.cpus,) if hasattr(args, "cpus") else ()
        args.action(*arguments)
    except KeyboardInterrupt:
        return 130
    except (IsolationError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"cpu-isolation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
