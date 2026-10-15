"""Platform report (Phase 0, Step 0.2): what this Mac and its libraries can do.

Writes outputs/checks/platform.txt and exits with an error if the GPU adapter is not a
real Metal GPU (a CPU/software fallback would make the renderer unusably slow).

Run: uv run python scripts/platform_report.py
"""

import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "platform.txt"

PACKAGES = [
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "matplotlib",
    "wgpu",
    "imageio",
    "imageio-ffmpeg",
    "pytest",
    "ruff",
]

GPU_LIMITS = [
    "max-buffer-size",
    "max-storage-buffer-binding-size",
    "max-texture-dimension-3d",
    "max-compute-workgroup-size-x",
    "max-compute-invocations-per-workgroup",
]

# Features later phases depend on: Phase 8 needs float32-filterable for hardware texture
# filtering (else manual trilinear in the shader); timestamp-query would allow GPU timing.
GPU_FEATURES_OF_INTEREST = ["float32-filterable", "timestamp-query", "shader-f16"]


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def gpu_section() -> tuple[list[str], bool]:
    import wgpu

    lines = []
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        return ["  No GPU adapter found."], False
    info = adapter.info
    lines += [
        f"  Device:        {info['device']}",
        f"  Backend:       {info['backend_type']}",
        f"  Adapter type:  {info['adapter_type']}",
        "  Limits:",
    ]
    lines += [f"    {k:40s} {adapter.limits[k]:>14,}" for k in GPU_LIMITS]
    lines.append("  Features of interest:")
    for f in GPU_FEATURES_OF_INTEREST:
        lines.append(f"    {f:40s} {'yes' if f in adapter.features else 'NO'}")
    lines.append("  All features: " + ", ".join(sorted(adapter.features)))
    ok = info["backend_type"] == "Metal" and info["adapter_type"] in (
        "IntegratedGPU",
        "DiscreteGPU",
    )
    return lines, ok


def main() -> int:
    mem_gb = int(sysctl("hw.memsize")) / 2**30
    lines = [
        f"Platform report, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "System",
        f"  macOS:         {platform.mac_ver()[0]}",
        f"  Machine:       {platform.machine()}",
        f"  CPU:           {sysctl('machdep.cpu.brand_string')}",
        f"  Cores:         {sysctl('hw.physicalcpu')} physical, {os.cpu_count()} logical",
        f"  RAM:           {mem_gb:.1f} GB",
        "",
        "Python",
        f"  Version:       {sys.version.split()[0]} ({sys.executable})",
        "  Packages:",
    ]
    lines += [f"    {p:16s} {version(p)}" for p in PACKAGES]
    lines += ["", "GPU (wgpu, high-performance adapter)"]
    gpu_lines, gpu_ok = gpu_section()
    lines += gpu_lines
    verdict = "PASS" if gpu_ok else "FAIL"
    lines += ["", f"GPU check (Metal + real GPU, not a CPU fallback): {verdict}"]

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Written to {OUT}")
    return 0 if gpu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
