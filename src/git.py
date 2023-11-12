"""Git utilities."""

import subprocess


def get_git_hash():
    """Get the Git commit hash."""

    with subprocess.Popen(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, stdout=subprocess.PIPE
    ) as process:
        return process.communicate()[0].strip().decode("utf-8")


if __name__ == "__main__":
    print(get_git_hash())
