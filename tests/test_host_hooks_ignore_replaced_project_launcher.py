"""Host command trust must cover everything executed before the installed Forge guard.

Test-audit: a replaced helper used to run code and override forge without changing the trusted
command. Existing setup and trust tests never execute generated commands with that replacement.
This regression runs the real generated commands concurrently, without a validator stub or
sequential hook short-circuit. It needs no production test seam.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import ROOT
from test_fix_new_repos_get_claude_as_their_worker_by import _new_repo
from test_setup import _hook_commands, _version

STORY = "FIX-TRUSTED-HOOK-LAUNCHER"


@pytest.mark.parametrize("lifecycle", ["new", "adopted-v1.2.2"])
def test_1_generated_host_hooks_run_installed_guards_without_project_launcher(
        repo, gh, tmp_path, claude_payload, codex_payload, lifecycle):
    if lifecycle == "new":
        client = _new_repo(repo, gh, tmp_path)
        made = repo.forge("init", cwd=client)
        assert made.returncode == 0, made.stdout + made.stderr
        repo.git("checkout", "-qb", "fix/check-host-hooks", cwd=client)
    else:
        client = repo.path
        shutil.copytree(ROOT / "tests/fixtures/adopted-v1.2.2/client", client,
                        dirs_exist_ok=True)
        repo.git("add", "-A")
        repo.git("commit", "-qm", "Adopt Forge on the earlier release")
        repo.git("checkout", "-qb", "fix/check-host-hooks")
        config = client / "forge.toml"
        old = config.read_text("utf-8")
        config.write_text(old.replace('version = "v1.2.2"',
                                      f'version = "{_version(repo)}"'), "utf-8")
        made = repo.forge("sync")
        assert made.returncode == 0, made.stdout + made.stderr

    commands = _hook_commands(client)
    assert len(commands) == 8
    assert all("\n" not in command and "\r" not in command
               for _, _, command in commands)
    # The helper stays for Git/Husky consumers, but is outside host command trust. Both a
    # side effect and a function override must be harmless even when every hook runs at once.
    marker = tmp_path / "project-launcher-ran"
    (client / ".forge/hooks.sh").write_text(
        f'echo ran > "{marker.as_posix()}"\n'
        'forge() { echo "overridden guard"; return 0; }\n', "utf-8")
    calls = []
    for host, event, command in commands:
        build = claude_payload if host.startswith(".claude") else codex_payload
        if event == "PreToolUse":
            payload = build(event, "Bash", {"command": "git status"}, cwd=client)
        elif event == "PostToolUse":
            tool = "AskUserQuestion" if host.startswith(".claude") else "request_user_input"
            payload = build(event, tool, {}, {}, cwd=client)
        else:
            payload = build(event, cwd=client)
        calls.append((host, event, command, payload, 0, ""))
        if event == "PreToolUse":
            for blocked, reason in (("git commit --no-verify -m change", "git hooks must run"),
                                    ("gh pr merge 12 --squash", "only a human merges")):
                payload = build(event, "Bash", {"command": blocked}, cwd=client)
                calls.append((host, event, command, payload, 2, reason))
        elif event == "PostToolUse":
            tool = "ExitPlanMode" if host.startswith(".claude") else "request_user_input"
            inputs = {"questions": [{"id": "approve_plan_" + "0" * 64}]}
            payload = build(event, tool, inputs, {"status": "cancelled"}, cwd=client)
            calls.append((host, event, command, payload, 2, "cancelled or failed"))

    def run(call):
        host, event, command, payload, status, reason = call
        done = subprocess.run(["sh", "-c", command], cwd=client, input=json.dumps(payload),
                              capture_output=True, text=True, encoding="utf-8", timeout=60)
        return host, event, status, reason, done

    # Hosts may run ALL matching hooks in parallel. Each invocation must stand on its own.
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        results = list(pool.map(run, calls))
    assert not marker.exists(), "The trusted hook executed the replaced repository helper"
    for host, event, status, reason, done in results:
        assert done.returncode == status, (host, event, done.stdout, done.stderr)
        assert reason in done.stderr, (host, event, done.stderr)
        assert "overridden guard" not in done.stdout + done.stderr


def test_2_host_launcher_upgrade_keeps_earlier_husky_checks_reachable(repo):
    # Changing the host command must not make sync treat its earlier Husky wrapper as a
    # user command: that would run a second Forge check before the team's hook body.
    repo.git("checkout", "-qb", "fix/check-host-hooks")
    repo.write("forge.toml", f'version = "{_version(repo)}"\n')
    (repo.path / ".husky/_").mkdir(parents=True)
    repo.git("config", "core.hooksPath", ".husky/_")
    for hook in ("pre-commit", "pre-push"):
        old = (f"sh -c '. \"$(git rev-parse --show-toplevel)/.forge/hooks.sh\" && "
               f"forge hook {hook} || exit 2' || exit 2")
        repo.write(f".husky/{hook}", f"#!/bin/sh\n{old}\necho team-{hook}\n")
    synced = repo.forge("sync")
    assert synced.returncode == 0, synced.stdout + synced.stderr
    again = repo.forge("sync")
    assert again.returncode == 0 and "Nothing to change" in again.stdout
    for hook, stdin, reason in (
            ("pre-commit", "", "was not started by Forge"),
            ("pre-push", "refs/heads/test " + "1" * 40 + " refs/heads/main " + "0" * 40 + "\n",
             "changes only through a merged pull request")):
        done = subprocess.run(["sh", f".husky/{hook}"], cwd=repo.path, input=stdin,
                              capture_output=True, text=True, timeout=60)
        assert done.stdout == f"team-{hook}\n"
        assert done.returncode != 0 and reason in done.stderr


def test_3_missing_installed_guard_explains_recovery_on_stderr(repo):
    repo.git("checkout", "-qb", "fix/check-missing-host-guard")
    repo.write("forge.toml", f'version = "{_version(repo)}"\n')
    synced = repo.forge("sync")
    assert synced.returncode == 0, synced.stdout + synced.stderr
    command = _hook_commands(repo.path)[0][2]
    script = shlex.split(command)[2]
    # Select the generated no-install fallback without depending on this machine's PATH.
    script = script.replace("if forge_path=$(command -v forge);", "if false;")
    script = script.replace("elif uvx_path=$(command -v uvx);", "elif false;")
    missing = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert missing.stderr.startswith("Forge isn't installed")
    assert "then run forge doctor" in missing.stderr
