#!/usr/bin/env python3
"""Topology-aware core allocation for benchmark jobs.

The machine layout comes from hwloc's ``lstopo`` (XML), or from /sys when
lstopo is unavailable: physical cores, their hyperthread siblings, the L3
cache they share and their NUMA node.

Allocation rules, chosen so that timings are comparable across batches:

* one PU per physical core; hyperthread siblings of the pool stay idle;
* a job lives inside one L3 domain when it fits, so its threads share the
  fastest common cache (an AMD CCX, or a whole Intel socket);
* each L3 domain runs at most ``jobs_per_l3`` jobs at once. Neighbours share
  that cache and the node's memory bandwidth, so a fixed cap keeps the load a
  job sees the same from batch to batch. ``auto``: one job per small domain
  (<= 16 cores, e.g. an AMD CCX), otherwise one job per 8 cores;
* among domains with room, the one with the fewest running jobs wins;
* a job larger than any domain takes whole idle domains of one NUMA node;
* memory is bound to the job's NUMA node when hwloc-bind or numactl is available.

``python3 topo_alloc.py`` prints the detected layout. Stdlib only.
"""
import argparse
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from pathlib import Path
from _common import core_list, pin_prefix

Core = namedtuple("Core", "pu siblings l3 numa")

HWLOC_BIN = ""


def _tool(name):
    path = os.path.join(HWLOC_BIN, name)
    return path if os.access(path, os.X_OK) else shutil.which(name)


def _bitmap(text):
    # hwloc bitmap: comma-separated 32-bit hex words, most significant first;
    # an empty word is zero.
    value = 0
    for word in text.split(","):
        value = (value << 32) | int(word or "0", 16)
    return {i for i in range(value.bit_length()) if value >> i & 1}


def _cpulist(text):
    cpus = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def from_lstopo(exe):
    xml = subprocess.run([exe, "--no-io", "--of", "xml"], capture_output=True,
                         text=True, check=True).stdout
    root = ET.fromstring(xml)
    numa_of = {}
    for node in root.iter("object"):
        if node.get("type") == "NUMANode" and node.get("cpuset"):
            for pu in _bitmap(node.get("cpuset")):
                numa_of[pu] = int(node.get("os_index"))
    cores = []

    def walk(node, l3):
        kind = node.get("type")
        if kind == "L3Cache":
            l3 = int(node.get("gp_index"))
        if kind == "Core":
            pus = sorted(int(c.get("os_index")) for c in node.iter("object") if c.get("type") == "PU")
            cores.append(Core(pus[0], tuple(pus[1:]), l3, numa_of.get(pus[0], -1)))
            return
        for child in node.findall("object"):
            walk(child, l3)

    walk(root.find("object"), None)
    return cores


def from_sys():
    base = "/sys/devices/system/cpu"
    numa_of = {}
    for node in (os.listdir("/sys/devices/system/node") if os.path.isdir("/sys/devices/system/node") else []):
        if node.startswith("node") and node[4:].isdigit():
            for pu in _cpulist(open(f"/sys/devices/system/node/{node}/cpulist").read()):
                numa_of[pu] = int(node[4:])
    seen, cores = set(), []
    for pu in sorted(_cpulist(open(f"{base}/online").read())):
        if pu in seen:
            continue
        siblings = sorted(_cpulist(open(f"{base}/cpu{pu}/topology/thread_siblings_list").read()))
        seen.update(siblings)
        l3 = None
        cache = f"{base}/cpu{pu}/cache"
        for index in sorted(os.listdir(cache)) if os.path.isdir(cache) else ():
            if index.startswith("index") and open(f"{cache}/{index}/level").read().strip() == "3":
                l3 = min(_cpulist(open(f"{cache}/{index}/shared_cpu_list").read()))
        cores.append(Core(siblings[0], tuple(siblings[1:]), l3, numa_of.get(siblings[0], -1)))
    return cores


def detect():
    exe = _tool("lstopo-no-graphics") or _tool("lstopo")
    if exe:
        try:
            return from_lstopo(exe), "lstopo"
        except (OSError, ValueError, IndexError, subprocess.CalledProcessError, ET.ParseError):
            pass
    try:
        return from_sys(), "/sys"
    except OSError as error:
        raise RuntimeError("Topology discovery requires hwloc or Linux /sys") from error


class Allocator:
    def __init__(self, pool=None, jobs_per_l3="auto", membind=True, hwloc_bin=None):
        global HWLOC_BIN
        HWLOC_BIN = str(Path(hwloc_bin).resolve()) if hwloc_bin else ""
        self.cores, self.source = detect()
        pool = set(core_list() if pool is None else pool)
        selected = []
        for core in self.cores:
            choices = sorted(({core.pu} | set(core.siblings)) & pool)
            if choices:
                selected.append(Core(choices[0], tuple(p for p in choices[1:]), core.l3, core.numa))
        self.cores = selected
        by_domain = defaultdict(list)
        for core in self.cores:
            by_domain[(core.numa, core.l3)].append(core.pu)
        if not by_domain:
            raise ValueError("No physical cores in requested pool")
        if jobs_per_l3 != "auto" and int(jobs_per_l3) < 1:
            raise ValueError("jobs-per-l3 must be positive")
        self.size = {d: len(p) for d, p in by_domain.items()}
        self.domain_pus = {d: set(p) for d, p in by_domain.items()}
        self.free = {d: sorted(p) for d, p in by_domain.items()}
        self.running = {d: 0 for d in by_domain}
        self.limit = {d: self._limit(n, jobs_per_l3) for d, n in self.size.items()}
        self.bind = _tool("hwloc-bind") if membind else None
        self.numactl = shutil.which("numactl") if membind else None

    @staticmethod
    def _limit(size, jobs_per_l3):
        if jobs_per_l3 != "auto":
            return int(jobs_per_l3)
        return 1 if size <= 16 else max(1, size // 8)

    def describe(self):
        lines = [f"topology from {self.source}: {len(self.cores)} physical cores, "
                 f"{sum(len(c.siblings) for c in self.cores)} hyperthread siblings kept idle"]
        for (numa, l3), size in sorted(self.size.items(), key=lambda item: str(item[0])):
            pus = self.free[(numa, l3)]
            lines.append(f"  NUMA {numa} L3 {l3}: {size} cores ({pus[0]}..{pus[-1]}), "
                         f"max {self.limit[(numa, l3)]} concurrent jobs")
        lines.append(f"  memory binding: {'hwloc-bind' if self.bind else ('numactl' if self.numactl else 'off (binding tool unavailable or disabled)')}")
        return "\n".join(lines)

    def take(self, n):
        """Reserve n cores; returns (domains, cores) or None if the job must wait."""
        fits = [d for d in self.free if len(self.free[d]) >= n and self.running[d] < self.limit[d]]
        if fits:
            d = min(fits, key=lambda d: (self.running[d], -len(self.free[d]), str(d)))
            cores, self.free[d] = self.free[d][:n], self.free[d][n:]
            self.running[d] += 1
            return [d], cores
        if n <= max(self.size.values()):
            return None
        # Larger than any domain: whole idle domains from one NUMA node.
        idle = defaultdict(list)
        for d in sorted(self.free, key=str):
            if self.running[d] == 0 and len(self.free[d]) == self.size[d]:
                idle[d[0]].append(d)
        for numa, domains in sorted(idle.items()):
            chosen, total = [], 0
            for d in domains:
                chosen.append(d)
                total += self.size[d]
                if total >= n:
                    cores = [pu for d in chosen for pu in self.free[d]][:n]
                    for d in chosen:
                        self.free[d], self.running[d] = [], 1
                    return chosen, cores
        return None

    def release(self, domains, cores):
        for d in domains:
            if self.running[d] == 1:
                self.free[d] = sorted(self.domain_pus[d])
            else:
                self.free[d] = sorted(set(self.free[d]) | (set(cores) & self.domain_pus[d]))
            self.running[d] -= 1

    def launcher(self, domains, cores):
        """Command prefix pinning a job to its cores and its NUMA node's memory."""
        prefix = pin_prefix(cores)
        numas = {d[0] for d in domains}
        if self.bind and len(numas) == 1 and -1 not in numas:
            prefix = [self.bind, "-p", "--membind", f"node:{numas.pop()}", "--"] + prefix
        elif self.numactl and len(numas) == 1 and -1 not in numas:
            prefix = [self.numactl, "--membind=" + str(next(iter(numas)))] + prefix
        return prefix


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", help="allowed CPU pool, e.g. 0-7,16-23")
    ap.add_argument("--jobs-per-l3", default="auto")
    ap.add_argument("--hwloc-bin", help="optional directory of hwloc executables")
    ap.add_argument("--no-membind", action="store_true")
    opt = ap.parse_args()
    print(Allocator(core_list(opt.cores), opt.jobs_per_l3, not opt.no_membind,
                    opt.hwloc_bin).describe())
