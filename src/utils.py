"""Configuration and device helpers shared by the scenario runners."""

import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml


def get_device():
    """Use the requested device, then CUDA if available, then CPU."""
    requested = os.environ.get("PDSTL_DEVICE")
    if requested:
        return torch.device(requested)
    if (
        torch.cuda.is_available()
        and os.environ.get("PDSTL_USE_CUDA", "1") != "0"
    ):
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path):
    """Load YAML from the requested path, repository configs, or wheel data."""
    candidate = Path(path)
    if candidate.is_file():
        return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if candidate.is_absolute():
        raise FileNotFoundError(candidate)
    repository_file = Path(__file__).resolve().parents[1] / candidate
    if repository_file.is_file():
        return yaml.safe_load(repository_file.read_text(encoding="utf-8"))
    installed_file = (
        Path(sys.prefix) / "share" / "probabilistic-dstl" / candidate
    )
    if installed_file.is_file():
        return yaml.safe_load(installed_file.read_text(encoding="utf-8"))
    raise FileNotFoundError(path)


class _SkippedRun(Exception):
    """Leave a disabled project block without executing its body."""


@contextmanager
def skip_run(flag, label):
    """Keep the project-by-project ``with skip_run(...), check()`` entry point."""
    if flag not in ("run", "skip"):
        raise ValueError("run flag must be 'run' or 'skip'")

    @contextmanager
    def check():
        if flag == "skip":
            print(f"Skipping {label}", flush=True)
            raise _SkippedRun
        print(f"Running {label}", flush=True)
        yield

    try:
        yield check
    except _SkippedRun:
        pass


def cpu_model():
    """The CPU model name from /proc/cpuinfo, else the platform's name."""
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
