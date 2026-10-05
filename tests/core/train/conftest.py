import pytest


class FakeProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def wait(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


@pytest.fixture
def fake_popen_factory():
    """Factory fixture: fake_popen_factory(lines=[...], returncode=0) returns a
    popen_factory-compatible callable. `.calls` records every invocation's
    command/cwd; `.last_process` exposes the most recently created FakeProcess
    (e.g. to assert .terminated after a cancel)."""

    def _make(lines=(), returncode=0):
        calls = []

        def factory(command, cwd=None, env=None, **kwargs):
            calls.append({"command": command, "cwd": cwd, "env": env})
            factory.last_process = FakeProcess(list(lines), returncode=returncode)
            return factory.last_process

        factory.calls = calls
        factory.last_process = None
        return factory

    return _make
