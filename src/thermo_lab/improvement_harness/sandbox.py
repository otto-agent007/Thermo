"""Fail-closed bubblewrap sandboxes for every command that executes candidate code."""

import os
import shutil
import sys
from pathlib import Path

_INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "TERM")
_CHECK_ENV = {
    "JAX_PLATFORMS": "cpu",
    "TMPDIR": "/tmp",
    "UV_NO_SYNC": "1",
    "UV_OFFLINE": "1",
    "UV_CACHE_DIR": "/tmp/uv-cache",
    "npm_config_cache": "/tmp/npm-cache",
    "npm_config_update_notifier": "false",
}
# Namespaces: no network (loopback only), no host PIDs, IPC or abstract sockets.
_ISOLATION = ("--unshare-all", "--die-with-parent", "--new-session")


class SandboxUnavailable(RuntimeError):
    """Raised instead of running candidate code without enforced limits."""


def _bwrap() -> str:
    path = shutil.which("bwrap")
    if path is None:
        raise SandboxUnavailable("sandbox unavailable: bubblewrap (bwrap) is not installed")
    return path


def _environment(extra: dict[str, str]) -> list[str]:
    env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
    env["HOME"] = str(Path.home())
    env.update(extra)
    args = ["--clearenv"]
    for name, value in sorted(env.items()):
        args += ["--setenv", name, value]
    return args


# Levels above each resolved executable that form its install root.
_TOOLS = {"uv": 0, "node": 1}


def _toolchain() -> tuple[list[Path], list[str]]:
    """Resolved tool locations, so hidden $HOME symlinks cannot break lookup.

    Returns read-only binds for install roots under $HOME and the resolved
    executable directories to put first on PATH.
    """
    home = Path.home().resolve()
    roots: set[Path] = set()
    path_dirs: list[str] = []
    tools = [(shutil.which(name), levels) for name, levels in _TOOLS.items()]
    # Level 2 for Python: a uv-managed install directory also holds the
    # minor-version alias links that virtual environments point through.
    for found, levels in tools + [(sys.executable, 2)]:
        if found is None:
            continue
        real = Path(found).resolve()
        if str(real.parent) not in path_dirs:
            path_dirs.append(str(real.parent))
        roots.add(real.parents[levels] if levels else real.parent)
    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or str(home / ".cache/ms-playwright")
    roots.add(Path(browsers).resolve())
    binds = sorted(root for root in roots if root.is_relative_to(home) and root.exists())
    return binds, path_dirs


def check_command(argv: list[str], *, worktree: Path, cwd: Path | None = None) -> list[str]:
    """Wrap one catalogued check.

    The host filesystem is read-only, the worktree is the only writable path,
    and $HOME, /run and /tmp are replaced by empty tmpfs mounts. Hiding /run
    and /tmp matters because read-only binds do not block connections to Unix
    sockets such as ssh-agent or Docker.
    """
    worktree = Path(worktree).resolve(strict=True)
    home = str(Path.home())
    args = [_bwrap(), "--ro-bind", "/", "/", *_ISOLATION]
    args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    args += ["--tmpfs", home]
    binds, path_dirs = _toolchain()
    for path in binds:
        args += ["--ro-bind", str(path), str(path)]
    directory = Path(cwd).absolute() if cwd is not None else worktree
    args += ["--bind", str(worktree), str(worktree), "--chdir", str(directory)]
    search = ":".join(path_dirs + [os.environ.get("PATH", "/usr/bin:/bin")])
    return args + _environment({**_CHECK_ENV, "PATH": search}) + ["--", *argv]


def hook_command(stage: Path, module: str) -> list[str]:
    """Run a staged hook with only system libraries and the standard library.

    No repository source, site-packages or home directory is mounted, so the
    hook cannot import the scorer or any numerical package.
    """
    stage = Path(stage).resolve(strict=True)
    python = Path(sys.executable).resolve()
    args = [_bwrap(), *_ISOLATION, "--ro-bind", "/usr", "/usr"]
    for name in ("/bin", "/lib", "/lib32", "/lib64", "/sbin"):
        path = Path(name)
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), name]
        elif path.is_dir():
            args += ["--ro-bind", name, name]
    args += ["--ro-bind", str(python.parents[1]), str(python.parents[1])]
    args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    args += ["--ro-bind", str(stage), "/candidate", "--chdir", "/candidate"]
    # -S: no site-packages. -E: ignore PYTHON* variables. -m prepends /candidate.
    command = [str(python), "-S", "-E", "-s", "-B", "-m", module]
    return args + _environment({"TMPDIR": "/tmp"}) + ["--", *command]
