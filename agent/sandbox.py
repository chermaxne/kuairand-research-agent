"""Experiment sandbox: subprocess execution with hard timeout, cwd isolation, env stripping and
(on macOS) `sandbox-exec` confinement — no network, writes only inside the workspace, and an explicit
read-deny list (used to hide the full data dir that contains hidden-test rows during the loop).

Hackathon-scale, deliberately simple (spec §10). The generated code never installs packages: the
OS sandbox blocks the network, and `static_code_check` refuses obviously forbidden constructs before
anything is executed (defence in depth).
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

STATUS_OK, STATUS_FAILED, STATUS_TIMEOUT = "ok", "failed", "timeout"


@dataclass
class SandboxResult:
    status: str                       # ok | failed | timeout
    returncode: Optional[int]
    runtime_s: float
    stdout_tail: str
    stderr_tail: str
    cmd: List[str]
    isolation: str
    stdout_path: str = ""
    stderr_path: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def error_excerpt(self, n_lines: int = 60) -> str:
        """Last `n_lines` of stderr (falls back to stdout), plus the reason line."""
        src = self.stderr_tail.strip() or self.stdout_tail.strip()
        lines = src.splitlines()[-n_lines:]
        head = ""
        if self.status == STATUS_TIMEOUT:
            head = f"TIMEOUT: killed after {self.runtime_s:.0f}s ({self.note})\n"
        elif self.status == STATUS_FAILED:
            head = f"exit code {self.returncode}\n"
        return (head + "\n".join(lines)).strip()


def detect_isolation(mode: str) -> str:
    """auto -> sandbox-exec on macOS when available, else none."""
    mode = (mode or "auto").lower()
    if mode == "none":
        return "none"
    have = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
    if mode == "sandbox-exec":
        if not have:
            raise RuntimeError("sandbox.isolation=sandbox-exec requested but sandbox-exec is unavailable")
        return "sandbox-exec"
    return "sandbox-exec" if have else "none"


def _q(p: str) -> str:
    return '"' + p.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(workspace: str, allow_write: Sequence[str] = (), deny_read: Sequence[str] = ()) -> str:
    """Apple sandbox profile: allow everything, then deny network + writes, re-allow writes in the
    workspace / temp dirs, and deny reads of the listed directories. Last matching rule wins."""
    ws = os.path.realpath(workspace)
    write_ok = [ws, "/private/tmp", "/private/var/folders", "/dev", os.path.realpath(os.environ.get("TMPDIR", "/tmp"))]
    write_ok += [os.path.realpath(p) for p in allow_write]
    parts = ["(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)"]
    parts += [f"(allow file-write* (subpath {_q(p)}))" for p in write_ok]
    for p in deny_read:
        rp = os.path.realpath(p)
        parts.append(f"(deny file-read* ({'literal' if os.path.isfile(rp) else 'subpath'} {_q(rp)}))")
    return "".join(parts)


def make_env(sandbox_cfg: Dict, pythonpath: Sequence[str] = ()) -> Dict[str, str]:
    """Minimal environment: only the passthrough list survives (API keys never reach the sandbox)."""
    env: Dict[str, str] = {}
    for k in sandbox_cfg.get("env_passthrough", ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"]):
        if k in os.environ:
            env[k] = os.environ[k]
    env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(os.path.realpath(p) for p in pythonpath)
    for k, v in (sandbox_cfg.get("extra_env") or {}).items():
        env[str(k)] = str(v)
    threads = sandbox_cfg.get("threads")
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(int(threads))
    return env


def _tail(path: str, max_bytes: int = 64_000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_command(cmd: List[str], workspace: str, timeout_s: float, env: Dict[str, str], isolation: str = "none",
                deny_read: Sequence[str] = (), allow_write: Sequence[str] = (), log_prefix: str = "") -> SandboxResult:
    """Run `cmd` inside `workspace`; stdout/stderr go to files in the workspace; kill the whole process
    group on timeout."""
    os.makedirs(workspace, exist_ok=True)
    full_cmd = list(cmd)
    if isolation == "sandbox-exec":
        full_cmd = ["sandbox-exec", "-p", sandbox_profile(workspace, allow_write, deny_read)] + full_cmd
    out_p = os.path.join(workspace, f"{log_prefix}stdout.txt")
    err_p = os.path.join(workspace, f"{log_prefix}stderr.txt")
    t0 = time.time()
    with open(out_p, "wb") as out, open(err_p, "wb") as err:
        proc = subprocess.Popen(full_cmd, cwd=workspace, env=env, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        note = ""
        try:
            rc = proc.wait(timeout=timeout_s)
            status = STATUS_OK if rc == 0 else STATUS_FAILED
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            rc, status, note = None, STATUS_TIMEOUT, f"limit {timeout_s:.0f}s"
    runtime = time.time() - t0
    return SandboxResult(status=status, returncode=rc, runtime_s=runtime, stdout_tail=_tail(out_p), stderr_tail=_tail(err_p),
                         cmd=full_cmd, isolation=isolation, stdout_path=out_p, stderr_path=err_p, note=note)


def run_pipeline(workspace: str, data_dir: str, split: str, out_name: str, timeout_s: float, sandbox_cfg: Dict,
                 pythonpath: Sequence[str] = (), deny_read: Sequence[str] = (), python: Optional[str] = None,
                 log_prefix: str = "", extra_env: Optional[Dict[str, str]] = None) -> SandboxResult:
    """Spec §5.2 invocation: python pipeline.py --data <dir> --split <val|test> --out <preds.csv>.
    `extra_env` adds harness-owned variables (e.g. the leak test's fast-path flag) on top of the stripped environment."""
    py = python or sandbox_cfg.get("python") or sys.executable
    isolation = detect_isolation(sandbox_cfg.get("isolation", "auto"))
    cmd = [py, "pipeline.py", "--data", os.path.realpath(data_dir), "--split", split, "--out", out_name]
    env = make_env(sandbox_cfg, pythonpath)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    return run_command(cmd, workspace, timeout_s, env, isolation=isolation, deny_read=deny_read, log_prefix=log_prefix)


# ---------------------------------------------------------------------------
# static guard on LLM-written code
# ---------------------------------------------------------------------------
def static_code_check(files: Dict[str, str], sandbox_cfg: Dict) -> List[str]:
    """Return a list of policy violations (empty = ok). Checks import statements against the forbidden
    module list and raw text against forbidden patterns."""
    forbidden_mods = set(sandbox_cfg.get("forbidden_imports", []))
    patterns = list(sandbox_cfg.get("forbidden_patterns", []))
    problems: List[str] = []
    imp_re = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w\.]*)", re.M)
    for name, code in files.items():
        if not name.endswith(".py"):
            continue
        for m in imp_re.finditer(code):
            root = m.group(1).split(".")[0]
            if root in forbidden_mods:
                problems.append(f"{name}: forbidden import '{m.group(1)}' (sandbox has no network / package installs)")
        for pat in patterns:
            if pat in code:
                problems.append(f"{name}: forbidden pattern {pat!r}")
        # Import ORDER: on macOS two OpenMP runtimes in one process abort the interpreter (SIGSEGV, no traceback).
        # Measured on this box: `import torch` before `import lightgbm` -> exit 139 / "OMP: Error #179"; the reverse
        # order survives. Refusing it here costs 0s; letting it run costs a full training run and a debug retry.
        for first, second in (sandbox_cfg.get("import_order") or []):
            pos = {}
            for m in imp_re.finditer(code):
                root = m.group(1).split(".")[0]
                if root in (first, second) and root not in pos:
                    pos[root] = m.start()
            if first in pos and second in pos and pos[second] < pos[first]:
                problems.append(f"{name}: `import {second}` appears before `import {first}` — on this machine that combination "
                                f"crashes the process with SIGSEGV (two OpenMP runtimes; 'OMP: Error #179'). Import {first} "
                                f"FIRST (before {second}) at the top of the file, or use only one of the two libraries.")
        # Leak prevention, first line: a same-row feedback column named inside a feature/field list is a label leak by
        # construction (the label long_view is a threshold of play_time_ms; is_click is nested in it). Narrow on purpose:
        # only assignments whose target looks like a field/feature list, so legitimate uses (reading the label column,
        # past-only aggregates built from earlier dates, multi-task targets) are not flagged.
        feedback = list(sandbox_cfg.get("feedback_columns", []))
        if feedback:
            # only list/tuple LITERALS assigned to a field/feature-named variable (not function calls, which build
            # aggregates from a column and are legitimate when past-only)
            for m in re.finditer(r"^[ \t]*((?:[A-Za-z_]\w*?)?(?:FIELD|FEAT|field|feat)\w*)\s*(?:[:=]|\+=)\s*[\[(][^\n]*$", code, re.M):
                line = m.group(0)
                hit = [c for c in feedback if re.search(r"['\"]" + re.escape(c) + r"['\"]", line)]
                if hit:
                    problems.append(f"{name}: feedback column(s) {hit} listed as input field(s) in `{m.group(1)}` — same-row feedback is the label "
                                    f"(long_view is a threshold of play_time_ms; is_click is nested in it). Feedback may only be a training "
                                    f"TARGET or a past-only aggregate from strictly earlier dates.")
        if re.search(r"--split\s+test|['\"]test['\"]\s*\]", code) and re.search(r"long_view|label", code):
            # Not a hard violation (the champion legitimately handles --split test at finalize); flag only.
            pass
    return problems
