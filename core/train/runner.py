"""Synchronous subprocess orchestration for one training stage: launch, stream stdout
as progress, support cooperative cancellation. Deliberately synchronous - running this
off the main thread is the caller's concern (the GUI layer, in a later batch), not
something to bake in here where it would make testing nondeterministic.
"""

import subprocess
from dataclasses import dataclass, field
from enum import Enum


class RunStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunResult:
    status: RunStatus
    return_code: int | None
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None


def run_stage(command, *, cwd=None, env=None, on_progress=None, popen_factory=None, cancel_check=None) -> RunResult:
    on_progress = on_progress or (lambda line: None)
    cancel_check = cancel_check or (lambda: False)
    popen_factory = popen_factory or subprocess.Popen

    process = popen_factory(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_lines: list[str] = []

    try:
        for line in process.stdout:
            stripped = line.rstrip("\n")
            log_lines.append(stripped)
            on_progress(stripped)
            if cancel_check():
                process.terminate()
                process.wait()
                return RunResult(status=RunStatus.CANCELLED, return_code=process.returncode, log_lines=log_lines)
    except Exception as exc:
        process.kill()
        return RunResult(status=RunStatus.FAILED, return_code=None, log_lines=log_lines, error=str(exc))

    return_code = process.wait()
    if return_code == 0:
        return RunResult(status=RunStatus.COMPLETED, return_code=return_code, log_lines=log_lines)
    return RunResult(
        status=RunStatus.FAILED,
        return_code=return_code,
        log_lines=log_lines,
        error=f"Process exited with code {return_code}",
    )
