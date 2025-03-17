import math
from multiprocessing import cpu_count
from pathlib import Path


def get_cpu_count() -> int:
    """Get the number of CPUs available to the current process.

    This function accounts for the possibility of a Docker container with
    limited CPU resources on a larger machine (which is ignored by
    `multiprocessing.cpu_count()`). This is derived from
    isce-framework/dolphin.

    Returns
    -------
    int
        The number of CPUs available to the current process.
    """

    def get_cpu_quota() -> int:
        return int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())

    def get_cpu_period() -> int:
        return int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())

    try:
        cfs_quota_us = get_cpu_quota()
        cfs_period_us = get_cpu_period()
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return math.ceil(cfs_quota_us / cfs_period_us)
    except Exception:
        pass
    return cpu_count()
