"""Package and verify a static dashboard; never execute or extract incoming tar members."""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

LIMIT = 100 * 1024 * 1024
TAR_LIMIT = LIMIT + 16 * 1024 * 1024
REQUIRED_STUDIES = {
    "full-row",
    "hop-fidelity",
    "return-fixture",
    "objective-steps",
    "survival-audit",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(sha, repository, run_id, attempt):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full source commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if not all(re.fullmatch(r"[1-9][0-9]*", x) for x in [run_id, attempt]):
        raise ValueError("Run and attempt must be positive integers")
    return {
        "source_commit": sha,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": attempt,
    }


def validate_snapshot(files):
    if "dist/index.html" not in files or not any(x.startswith("dist/assets/") for x in files):
        raise ValueError("Missing compiled dashboard")
    snapshot = json.loads(files["dist/data/project.json"])
    if snapshot.get("mode") != "snapshot" or snapshot.get("availability") != "available":
        raise ValueError("M4G evidence unavailable in static snapshot")
    cells = snapshot.get("study", {}).get("cells", [])
    ids = [c["id"] for c in cells]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("Expected all 60 distinct M4G cells")
    for cell_id in ids:
        if f"dist/data/cells/{cell_id.replace('/', '~')}.json" not in files:
            raise ValueError("Missing cell detail")
    studies = snapshot.get("recentStudies", [])
    study_ids = [s["id"] for s in studies]
    if (
        not REQUIRED_STUDIES.issubset(study_ids)
        or len(set(study_ids)) != len(study_ids)
        or any(s.get("availability") != "available" for s in studies)
    ):
        raise ValueError("Recent study evidence missing or unavailable")


def write_checksums(output):
    output.joinpath("SHA256SUMS").write_text(
        "".join(
            f"{digest(output.joinpath(name).read_bytes())}  {name}\n"
            for name in ["dashboard.tar.gz", "release.json"]
        )
    )


def build(dist, output, sha, repository, run_id, attempt):
    manifest = {"schema": 1, **identity(sha, repository, run_id, attempt)}
    files = {}
    if dist.is_symlink():
        raise ValueError("Build directory cannot be a symlink")
    for path in sorted(dist.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not release files")
        if path.is_file():
            files["dist/" + path.relative_to(dist).as_posix()] = path.read_bytes()
    if sum(map(len, files.values())) > LIMIT:
        raise ValueError("Dashboard exceeds bundle size limit")
    validate_snapshot(files)
    manifest["files"] = {name: digest(data) for name, data in files.items()}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    output.joinpath("release.json").write_bytes(manifest_bytes)
    with output.joinpath("dashboard.tar.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted({**files, "release.json": manifest_bytes}.items()):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o644
                    archive.addfile(member, io.BytesIO(data))
    write_checksums(output)


def verify(output, sha, repository, run_id, attempt):
    expected = identity(sha, repository, run_id, attempt)
    for name in ["dashboard.tar.gz", "release.json", "SHA256SUMS"]:
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
            raise ValueError("Missing, unsafe or oversized bundle file")
    checksums = "".join(
        f"{digest((output / name).read_bytes())}  {name}\n"
        for name in ["dashboard.tar.gz", "release.json"]
    )
    if (output / "SHA256SUMS").read_text() != checksums:
        raise ValueError("Transport checksum mismatch")
    manifest_bytes = (output / "release.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != 1 or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Release identity mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Empty file manifest")
    # Bound the entire decompressed stream, including extension headers that tarfile
    # consumes internally before yielding a TarInfo member.
    decompressed = 0
    with gzip.open(output / "dashboard.tar.gz", "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            decompressed += len(chunk)
            if decompressed > TAR_LIMIT:
                raise ValueError("Decompressed archive exceeds size limit")
    files = {}
    total = 0
    with tarfile.open(output / "dashboard.tar.gz", "r:gz") as archive:
        for member in archive:
            name = member.name
            total += member.size
            if (
                not member.isfile()
                or name in files
                or total > LIMIT
                or len(files) >= 10000
                or member.size < 0
                or PurePosixPath(name).is_absolute()
                or ".." in name.split("/")
                or "\\" in name
                or PurePosixPath(name).as_posix() != name
            ):
                raise ValueError("Unsafe, duplicate or oversized archive member")
            if name != "release.json" and (not name.startswith("dist/") or name not in entries):
                raise ValueError("Unlisted release file")
            files[name] = archive.extractfile(member).read()
    if files.pop("release.json", None) != manifest_bytes:
        raise ValueError("Embedded manifest mismatch")
    if set(files) != set(entries) or any(
        digest(data) != entries[name] for name, data in files.items()
    ):
        raise ValueError("Release file manifest mismatch")
    validate_snapshot(files)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["build", "verify"])
    parser.add_argument("--dist", type=Path, default=Path("dashboard/dist"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    values = (args.output, args.sha, args.repository, args.run_id, args.attempt)
    if args.operation == "build":
        build(args.dist, *values)
    verify(*values)
    print(f"Verified dashboard for {args.sha}, run {args.run_id}, attempt {args.attempt}")


if __name__ == "__main__":
    main()
