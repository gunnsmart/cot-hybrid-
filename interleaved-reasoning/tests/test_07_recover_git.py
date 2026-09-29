"""The push-rule guard must exist, be executable, and pass on a clean repo."""
import os
import subprocess

from scripts import common


def test_recover_git_script_exists_and_executable():
    p = os.path.join(common.REPO_ROOT, "scripts", "recover_git.sh")
    assert os.path.exists(p)
    assert os.access(p, os.X_OK)


def test_recover_git_passes_on_clean_pushed_repo():
    r = subprocess.run(["scripts/recover_git.sh"], cwd=common.REPO_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"recover_git.sh refused:\n{r.stdout}\n{r.stderr}"
    assert r.stdout.startswith("OK:")


def test_recover_git_refuses_unpushed_commits(tmp_path):
    """Behavioral check in a throwaway repo: unpushed commit -> exit 1."""
    repo = common.REPO_ROOT
    script = f"""
set -e
git init -q -b main
git config user.email t@t
git config user.name t
git init -q --bare origin.git
git remote add origin ./origin.git
echo a > f && git add f && git commit -qm a
git push -q origin main
echo b > f && git add f && git commit -qm b   # unpushed
mkdir -p scripts && cp {repo}/scripts/recover_git.sh scripts/
sh scripts/recover_git.sh
"""
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 1, (
        "recover_git.sh must REFUSE when unpushed commits exist; "
        f"got rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    assert "REFUSING" in (r.stdout + r.stderr)
