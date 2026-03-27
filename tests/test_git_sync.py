import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from src import git_sync


@contextmanager
def pushd(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def run_git(args, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def collect_stream(generator):
    logs = []
    success = False
    try:
        while True:
            logs.append(next(generator))
    except StopIteration as exc:
        success = exc.value
    return success, logs


def init_repo_with_tracked_ignored_file(tmp_path: Path) -> Path:
    run_git(["init"], cwd=tmp_path)
    run_git(["config", "user.name", "Test User"], cwd=tmp_path)
    run_git(["config", "user.email", "test@example.com"], cwd=tmp_path)

    ignored_dir = tmp_path / "data"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "secret.txt"
    ignored_file.write_text("top-secret\n")
    (tmp_path / ".gitignore").write_text("data/\n")

    run_git(["add", ".gitignore"], cwd=tmp_path)
    run_git(["add", "-f", "data/secret.txt"], cwd=tmp_path)
    run_git(["commit", "-m", "Initial commit"], cwd=tmp_path)
    run_git(["branch", "-M", "main"], cwd=tmp_path)
    return ignored_file


def test_remove_ignored_tracked_files_keeps_local_copy(tmp_path):
    ignored_file = init_repo_with_tracked_ignored_file(tmp_path)

    with pushd(tmp_path):
        assert git_sync.get_tracked_ignored_paths() == ["data/secret.txt"]

        success, logs = collect_stream(
            git_sync.remove_ignored_tracked_files_from_remote_stream(push=False)
        )

    assert success
    assert ignored_file.exists()
    assert "Sync Now" in "\n".join(logs)

    tracked = run_git(
        ["ls-files", "--error-unmatch", "data/secret.txt"],
        cwd=tmp_path,
        check=False,
    )
    assert tracked.returncode != 0

    commit_count = run_git(["rev-list", "--count", "HEAD"], cwd=tmp_path)
    assert commit_count.stdout.strip() == "2"


def test_remove_ignored_tracked_files_aborts_with_preexisting_staged_changes(tmp_path):
    init_repo_with_tracked_ignored_file(tmp_path)
    (tmp_path / "notes.txt").write_text("draft\n")
    run_git(["add", "notes.txt"], cwd=tmp_path)

    with pushd(tmp_path):
        success, logs = collect_stream(
            git_sync.remove_ignored_tracked_files_from_remote_stream(push=False)
        )

    assert not success
    assert "Hay cambios ya indexados" in "\n".join(logs)

    tracked = run_git(
        ["ls-files", "--error-unmatch", "data/secret.txt"],
        cwd=tmp_path,
        check=False,
    )
    assert tracked.returncode == 0
