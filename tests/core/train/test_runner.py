from core.train.runner import RunStatus, run_stage


def test_run_stage_completed_streams_lines(fake_popen_factory):
    factory = fake_popen_factory(lines=["line1", "line2"], returncode=0)
    seen = []

    result = run_stage(["echo", "hi"], on_progress=seen.append, popen_factory=factory)

    assert result.status == RunStatus.COMPLETED
    assert result.return_code == 0
    assert result.log_lines == ["line1", "line2"]
    assert seen == ["line1", "line2"]


def test_run_stage_failed_on_nonzero_returncode(fake_popen_factory):
    factory = fake_popen_factory(lines=["oops"], returncode=1)

    result = run_stage(["bad", "cmd"], popen_factory=factory)

    assert result.status == RunStatus.FAILED
    assert result.return_code == 1
    assert "1" in result.error


def test_run_stage_passes_command_and_cwd_to_popen_factory(fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)

    run_stage(["my", "command"], cwd="/some/dir", popen_factory=factory)

    assert factory.calls == [{"command": ["my", "command"], "cwd": "/some/dir", "env": None}]


def test_run_stage_passes_env_through_to_popen_factory(fake_popen_factory):
    factory = fake_popen_factory(lines=[], returncode=0)

    run_stage(["x"], env={"FOO": "bar"}, popen_factory=factory)

    assert factory.calls[0]["env"] == {"FOO": "bar"}


def test_run_stage_cancels_mid_stream_and_terminates_process(fake_popen_factory):
    factory = fake_popen_factory(lines=["l1", "l2", "l3", "l4"], returncode=0)
    state = {"count": 0}

    def cancel_check():
        state["count"] += 1
        return state["count"] >= 2

    result = run_stage(["x"], popen_factory=factory, cancel_check=cancel_check)

    assert result.status == RunStatus.CANCELLED
    assert result.log_lines == ["l1", "l2"]
    assert factory.last_process.terminated is True


def test_run_stage_handles_exception_during_streaming_as_failed():
    class ExplodingProcess:
        returncode = None

        @property
        def stdout(self):
            raise RuntimeError("pipe broke")

        def kill(self):
            self.killed = True

    def factory(command, cwd=None, **kwargs):
        return ExplodingProcess()

    result = run_stage(["x"], popen_factory=factory)

    assert result.status == RunStatus.FAILED
    assert "pipe broke" in result.error


def test_run_stage_works_with_default_on_progress_and_cancel_check(fake_popen_factory):
    factory = fake_popen_factory(lines=["a"], returncode=0)

    result = run_stage(["x"], popen_factory=factory)

    assert result.status == RunStatus.COMPLETED
