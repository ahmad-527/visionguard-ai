"""Read-only focused diff hygiene, not disk cleanup or dataset scanning."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path.cwd())
    a = p.parse_args()
    repo = a.repository

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    base = "3ab2b3bc9135aa1c1c63aef78fd2d88c1552f1f7"
    paths = set(git("diff", "--name-only", base).splitlines()) | set(
        git("ls-files", "--others", "--exclude-standard").splitlines()
    )
    secret = re.compile(
        r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
        r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|sk-[A-Za-z0-9_-]{35,}"
    )
    private = re.compile(
        r"[A-Z]:[\\/]Users[\\/]|/home/[A-Za-z0-9_.-]+/|"
        r"/Users/[A-Za-z0-9_. -]+/|AppData[\\/]",
        re.I,
    )
    failures = []
    total = 0
    for relative in sorted(paths):
        path = repo / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > 2**20:
            failures.append([relative, "unintended-large-file"])
        text = path.read_text(encoding="utf-8")
        if secret.search(text):
            failures.append([relative, "possible-secret"])
        if private.search(text):
            failures.append([relative, "personal-absolute-path"])
    git("diff", "--check", base)
    report = {
        "changed_files_scanned": len(paths),
        "total_bytes": total,
        "secret_large_file_personal_path_whitespace_failures": failures,
        "dataset_roots_scanned": False,
    }
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
