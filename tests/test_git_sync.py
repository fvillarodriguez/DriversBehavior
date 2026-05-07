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


def init_remote_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote_work = tmp_path / "remote_work"
    remote_bare = tmp_path / "remote.git"
    remote_work.mkdir()

    run_git(["init"], cwd=remote_work)
    run_git(["config", "user.name", "Test User"], cwd=remote_work)
    run_git(["config", "user.email", "test@example.com"], cwd=remote_work)

    (remote_work / ".gitignore").write_text(
        "Datos/\nResultados/\nsimulación/\ndocs/\nNLP/\nDRIFT/\n"
    )
    (remote_work / "app.py").write_text("remote v1\n")
    (remote_work / "src").mkdir()
    (remote_work / "src" / "module.py").write_text("VALUE = 'v1'\n")

    run_git(["add", "."], cwd=remote_work)
    run_git(["commit", "-m", "Initial remote commit"], cwd=remote_work)
    run_git(["branch", "-M", "main"], cwd=remote_work)

    run_git(["init", "--bare", str(remote_bare)], cwd=tmp_path)
    run_git(["remote", "add", "origin", str(remote_bare)], cwd=remote_work)
    run_git(["push", "-u", "origin", "main"], cwd=remote_work)
    run_git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=remote_bare)

    return remote_work, remote_bare


def push_remote_change(remote_work: Path, relative_path: str, content: str) -> None:
    target = remote_work / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    run_git(["add", relative_path], cwd=remote_work)
    run_git(["commit", "-m", f"Update {relative_path}"], cwd=remote_work)
    run_git(["push", "origin", "main"], cwd=remote_work)


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


def test_update_local_repo_initializes_missing_git_repo_and_fetches_remote(tmp_path):
    _, remote_bare = init_remote_repo(tmp_path)
    local_repo = tmp_path / "local"
    local_repo.mkdir()
    (local_repo / "stale.py").write_text("remove me\n")
    (local_repo / "Datos").mkdir()
    (local_repo / "Datos" / "flujos.duckdb").write_text("keep local data\n")

    success, message = git_sync.update_local_repo_from_github(
        repo_dir=local_repo,
        remote_url=str(remote_bare),
    )

    assert success, message
    assert (local_repo / ".git").exists()
    assert (local_repo / "app.py").read_text() == "remote v1\n"
    assert (local_repo / "src" / "module.py").read_text() == "VALUE = 'v1'\n"
    assert not (local_repo / "stale.py").exists()
    assert (local_repo / "Datos" / "flujos.duckdb").read_text() == "keep local data\n"
    assert run_git(["branch", "--show-current"], cwd=local_repo).stdout.strip() == "main"
    assert run_git(["remote", "get-url", "origin"], cwd=local_repo).stdout.strip() == str(remote_bare)


def test_update_local_repo_overwrites_modified_tracked_files(tmp_path):
    remote_work, remote_bare = init_remote_repo(tmp_path)
    local_repo = tmp_path / "local"
    run_git(["clone", "-b", "main", str(remote_bare), str(local_repo)], cwd=tmp_path)

    push_remote_change(remote_work, "app.py", "remote v2\n")
    (local_repo / "app.py").write_text("local edit\n")

    success, message = git_sync.update_local_repo_from_github(
        repo_dir=local_repo,
        remote_url=str(remote_bare),
    )

    assert success, message
    assert (local_repo / "app.py").read_text() == "remote v2\n"


def test_update_local_repo_removes_untracked_nonignored_files(tmp_path):
    _, remote_bare = init_remote_repo(tmp_path)
    local_repo = tmp_path / "local"
    run_git(["clone", "-b", "main", str(remote_bare), str(local_repo)], cwd=tmp_path)
    (local_repo / "scratch.py").write_text("temporary\n")
    (local_repo / "nested").mkdir()
    (local_repo / "nested" / "scratch.py").write_text("temporary\n")

    success, message = git_sync.update_local_repo_from_github(
        repo_dir=local_repo,
        remote_url=str(remote_bare),
    )

    assert success, message
    assert not (local_repo / "scratch.py").exists()
    assert not (local_repo / "nested").exists()


def test_update_local_repo_preserves_ignored_local_outputs(tmp_path):
    _, remote_bare = init_remote_repo(tmp_path)
    local_repo = tmp_path / "local"
    run_git(["clone", "-b", "main", str(remote_bare), str(local_repo)], cwd=tmp_path)
    (local_repo / "Datos").mkdir()
    (local_repo / "Datos" / "local.csv").write_text("data\n")
    (local_repo / "Resultados").mkdir()
    (local_repo / "Resultados" / "model.pkl").write_text("model\n")

    success, message = git_sync.update_local_repo_from_github(
        repo_dir=local_repo,
        remote_url=str(remote_bare),
    )

    assert success, message
    assert (local_repo / "Datos" / "local.csv").read_text() == "data\n"
    assert (local_repo / "Resultados" / "model.pkl").read_text() == "model\n"


def test_update_local_repo_invalid_remote_returns_controlled_error(tmp_path):
    local_repo = tmp_path / "local"
    local_repo.mkdir()
    (local_repo / "local.py").write_text("still here\n")

    success, message = git_sync.update_local_repo_from_github(
        repo_dir=local_repo,
        remote_url=str(tmp_path / "missing.git"),
    )

    assert not success
    assert "No se pudo descargar la base de código desde GitHub" in message
    assert (local_repo / "local.py").read_text() == "still here\n"
