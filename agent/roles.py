"""Briefing assembly, role calls and output parsing for the four LLM roles.

Prompt assembly order (spec §3, for provider prompt caching):
    static role prompt -> static knowledge library -> dynamic state block -> dynamic ledger -> task instruction
The LLM only ever returns: a hypothesis JSON (§5.1), whole code files, a debug fix / abandon JSON,
one ≤20-word lesson, and a narrative rendered from harness facts. Nothing it returns is ever
interpreted as a score, decision or streak.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm_client import CallLog, LLMClient, LLMError, LLMResponse
from .memory import one_line, truncate_words
from .schemas import ContractError, HarnessResult, ResearcherPlan, TokenUsage

ROLE_FILES = {"researcher": "researcher.md", "engineer": "engineer.md", "debugger": "debugger.md",
              "scribe_lesson": "scribe_lesson.md", "scribe_logentry": "scribe_logentry.md"}
TASK_MARKER = "<!-- TASK -->"

FILE_START = re.compile(r"^=== FILE: (?P<name>[^\s=]+) ===\s*$", re.M)
FILE_END = re.compile(r"^=== END FILE ===\s*$", re.M)


# ---------------------------------------------------------------------------
# parsing helpers (pure)
# ---------------------------------------------------------------------------
def _strip_fence(body: str) -> str:
    lines = body.strip("\n").splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).rstrip() + "\n"


def parse_file_blocks(text: str) -> Dict[str, str]:
    """Extract `=== FILE: name === ... === END FILE ===` blocks. Falls back to a single ```python fence
    (treated as pipeline.py). Returns {} when nothing usable is found."""
    files: Dict[str, str] = {}
    pos = 0
    while True:
        m = FILE_START.search(text, pos)
        if not m:
            break
        e = FILE_END.search(text, m.end())
        if not e:
            break
        files[m.group("name").strip()] = _strip_fence(text[m.end():e.start()])
        pos = e.end()
    if not files:
        fences = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, flags=re.S)
        if len(fences) == 1 and "argparse" in fences[0]:
            files["pipeline.py"] = fences[0].rstrip() + "\n"
    return files


def render_file_blocks(files: Dict[str, str]) -> str:
    out = []
    for name in sorted(files):
        out.append(f"=== FILE: {name} ===\n```python\n{files[name].rstrip()}\n```\n=== END FILE ===")
    return "\n\n".join(out)


def extract_json(text: str) -> Any:
    """Find the first JSON object in free text (tolerates ```json fences and leading prose)."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise ContractError("no JSON object found in output")


def parse_researcher(text: str) -> ResearcherPlan:
    return ResearcherPlan.from_obj(extract_json(text))


@dataclass
class DebugOutcome:
    action: str                      # fix | abandon
    files: Dict[str, str] = field(default_factory=dict)
    reason: str = ""
    fix_summary: str = ""


def parse_debugger(text: str) -> DebugOutcome:
    files = parse_file_blocks(text)
    if files:
        m = re.search(r"^FIX SUMMARY:\s*(.+)$", text, flags=re.M)
        summary = one_line(m.group(1), 200) if m else one_line(text.strip().splitlines()[0] if text.strip() else "fix", 200)
        return DebugOutcome(action="fix", files=files, fix_summary=summary)
    try:
        obj = extract_json(text)
    except ContractError:
        raise ContractError("debugger output has neither file blocks nor an abandon JSON")
    if isinstance(obj, dict) and str(obj.get("action", "")).lower() == "abandon":
        return DebugOutcome(action="abandon", reason=one_line(obj.get("reason", "no reason given"), 200))
    raise ContractError("debugger JSON must be {\"action\":\"abandon\",\"reason\":...}")


# ---------------------------------------------------------------------------
# default prompts (used only when prompts/<role>.md is missing, e.g. the Phase-1 stub loop)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM = {
    "researcher": "You are the Researcher. Output ONLY the JSON object described in the task.",
    "engineer": "You are the Engineer. Output the full modified file(s) as === FILE: name === blocks.",
    "debugger": "You are the Debugger. Output fixed file(s) as === FILE: name === blocks or {\"action\":\"abandon\",\"reason\":\"...\"}.",
    "scribe_lesson": "You are the Scribe. Output one sentence of at most 20 words.",
    "scribe_logentry": "You are the Scribe. Render the supplied facts as a short markdown narrative. Copy numbers verbatim.",
}
DEFAULT_TASK = {
    "researcher": (
        "Propose the next experiment. Reply with ONLY this JSON object:\n"
        '{"hypothesis": "...", "category": "feature | model | training | multitask | other", '
        '"change_spec": "...", "expected_risk": "low | medium | high", "builds_on": "champion", "rationale": "..."}'),
    "engineer": (
        "Implement the change specification above on the champion files. Output the COMPLETE modified file(s) "
        "using exactly this format for each file:\n=== FILE: pipeline.py ===\n```python\n<full file>\n```\n=== END FILE ===\n"
        "Keep edits minimal and targeted. Do not print mock results. Never install packages or access the network."),
    "debugger": (
        "Fix the failing code so it runs. Output the COMPLETE fixed file(s) in === FILE: name === blocks, preceded by one "
        "line 'FIX SUMMARY: <what you changed>'. If it cannot be fixed within the experiment's intent, output "
        '{"action":"abandon","reason":"..."} instead.'),
    "scribe_lesson": "Write ONE sentence (max 20 words) stating the lesson from this result. Output only the sentence.",
    "scribe_logentry": "Render the facts above as a short markdown narrative (max 120 words). Copy every number verbatim; add nothing not in the facts.",
}
REASK = ("Your previous reply did not satisfy the required output format: {error}\n"
         "Reply again with ONLY the required output, nothing else.")


def load_prompt(prompts_dir: str, role: str) -> Tuple[str, str]:
    """Return (system_prompt, task_template). The file may contain both, separated by `<!-- TASK -->`."""
    p = os.path.join(prompts_dir, ROLE_FILES[role])
    if not os.path.exists(p):
        return DEFAULT_SYSTEM[role], DEFAULT_TASK[role]
    text = open(p, encoding="utf-8").read()
    if TASK_MARKER in text:
        sys_p, task_p = text.split(TASK_MARKER, 1)
        return sys_p.strip(), task_p.strip()
    return text.strip(), DEFAULT_TASK[role]


# ---------------------------------------------------------------------------
# role runner
# ---------------------------------------------------------------------------
class Roles:
    """Calls the LLM for each role, logs prompts/responses per iteration, accounts tokens.
    `iteration_usage` / `role_usage` are read by the harness after each iteration."""

    def __init__(self, client: LLMClient, cfg: Dict[str, Any], prompts_dir: str, knowledge_path: str, call_log: Optional[CallLog] = None):
        self.client = client
        self.cfg = cfg
        self.prompts_dir = prompts_dir
        self.knowledge = open(knowledge_path, encoding="utf-8").read() if knowledge_path and os.path.exists(knowledge_path) else ""
        self.call_log = call_log
        self.log = None                                   # optional console logger (set by the harness)
        self.transcript_dir: Optional[str] = None
        self.iteration = 0
        self.iteration_usage = TokenUsage()
        self.role_usage: Dict[str, int] = {}             # cumulative for this process (diagnostics)
        self.iteration_role_usage: Dict[str, int] = {}   # per iteration (the harness adds these to run_state)
        self.calls_this_iteration = 0
        self.last_error: str = ""

    # -- bookkeeping ---------------------------------------------------------
    def begin_iteration(self, iteration: int, transcript_dir: Optional[str]) -> None:
        self.iteration = iteration
        self.transcript_dir = transcript_dir
        self.iteration_usage = TokenUsage()
        self.iteration_role_usage = {}
        self.calls_this_iteration = 0
        if transcript_dir:
            os.makedirs(transcript_dir, exist_ok=True)

    def _model(self, role: str) -> str:
        key = "scribe" if role.startswith("scribe") else role
        return str(self.cfg["llm"][f"{key}_model"])

    def _max_tokens(self, role: str) -> int:
        key = "scribe" if role.startswith("scribe") else role
        return int(self.cfg["llm"]["max_output_tokens"][key])

    def _call(self, role: str, system_blocks: Sequence[str], messages: List[Dict[str, str]], purpose: str, attempt: int = 1) -> LLMResponse:
        resp = self.client.complete(role=role, model=self._model(role), system_blocks=system_blocks, messages=messages,
                                    max_tokens=self._max_tokens(role))
        self.iteration_usage.add(resp.usage)
        self.role_usage[role] = self.role_usage.get(role, 0) + resp.usage.total
        self.iteration_role_usage[role] = self.iteration_role_usage.get(role, 0) + resp.usage.total
        self.calls_this_iteration += 1
        if self.log:
            self.log(f"[llm] {purpose}: {resp.model} answered in {resp.latency_s:.0f}s "
                     f"({resp.usage.input_tokens + resp.usage.cache_read_input_tokens} in / {resp.usage.output_tokens} out, stop={resp.stop_reason or '?'})")
        if self.call_log:
            self.call_log.record(self.iteration, role, resp, attempt=attempt, purpose=purpose)
        if self.transcript_dir:
            fn = os.path.join(self.transcript_dir, f"{purpose}.md")
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(f"# {role} — {purpose} (model {resp.model}, {resp.usage.total} tokens{', estimated' if resp.estimated_usage else ''})\n\n")
                for note in getattr(resp, "fallback_notes", []) or []:
                    fh.write(f"> FALLBACK: {note}\n\n")
                for i, b in enumerate(system_blocks):
                    fh.write(f"## system block {i + 1}\n\n{b}\n\n")
                for m in messages:
                    fh.write(f"## {m['role']}\n\n{m['content']}\n\n")
                fh.write(f"## assistant (response)\n\n{resp.text}\n")
        return resp

    def _system_blocks(self, role: str) -> List[str]:
        sys_p, _ = load_prompt(self.prompts_dir, role)
        blocks = [sys_p]
        if role == "researcher" and self.knowledge:
            blocks.append("# KNOWLEDGE LIBRARY (domain playbook)\n\n" + self.knowledge)
        return blocks

    # -- researcher ----------------------------------------------------------
    def researcher(self, dynamic_briefing: str) -> Tuple[Optional[ResearcherPlan], str, str]:
        """Returns (plan | None, error, raw_text). One re-ask on malformed output (spec §13 Phase 3)."""
        _, task = load_prompt(self.prompts_dir, "researcher")
        messages = [{"role": "user", "content": dynamic_briefing.rstrip() + "\n\n# TASK\n" + task}]
        try:
            resp = self._call("researcher", self._system_blocks("researcher"), messages, "researcher")
        except LLMError as e:
            return None, f"researcher_llm_error: {e}", ""
        try:
            return parse_researcher(resp.text), "", resp.text
        except ContractError as e:
            err1 = str(e)
        messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": REASK.format(error=err1)}]
        try:
            resp2 = self._call("researcher", self._system_blocks("researcher"), messages, "researcher_reask", attempt=2)
        except LLMError as e:
            return None, f"researcher_llm_error: {e}", resp.text
        try:
            return parse_researcher(resp2.text), "", resp2.text
        except ContractError as e:
            return None, f"researcher_malformed: {e} (after one re-ask; first error: {err1})", resp2.text

    # -- engineer ------------------------------------------------------------
    @staticmethod
    def engineer_message(plan: ResearcherPlan, champion_files: Dict[str, str], task: str, contract_note: str = "") -> str:
        return (f"# Change specification (from the Researcher)\nHYPOTHESIS: {plan.hypothesis}\nCATEGORY: {plan.category}\n"
                f"EXPECTED RISK: {plan.expected_risk}\nCHANGE SPEC:\n{plan.change_spec}\n\n"
                f"# Current champion files\n{render_file_blocks(champion_files)}\n\n"
                f"# Pipeline contract\n{contract_note}\n\n# TASK\n{task}")

    def engineer(self, plan: ResearcherPlan, champion_files: Dict[str, str], contract_note: str = "") -> Tuple[Optional[Dict[str, str]], str]:
        _, task = load_prompt(self.prompts_dir, "engineer")
        messages = [{"role": "user", "content": self.engineer_message(plan, champion_files, task, contract_note)}]
        try:
            resp = self._call("engineer", self._system_blocks("engineer"), messages, "engineer")
        except LLMError as e:
            return None, f"engineer_llm_error: {e}"
        files = parse_file_blocks(resp.text)
        if "pipeline.py" in files:
            return files, ""
        why = "no `=== FILE: pipeline.py ===` block with the complete file was found"
        if resp.stop_reason == "max_tokens":
            why += " (your reply was cut off at the output-token limit: keep the file compact, no commentary)"
        messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": REASK.format(error=why)}]
        try:
            resp2 = self._call("engineer", self._system_blocks("engineer"), messages, "engineer_reask", attempt=2)
        except LLMError as e:
            return None, f"engineer_llm_error: {e}"
        files = parse_file_blocks(resp2.text)
        if "pipeline.py" in files:
            return files, ""
        return None, "engineer_malformed: no pipeline.py file block (after one re-ask)"

    # -- debugger ------------------------------------------------------------
    @staticmethod
    def debugger_message(plan: ResearcherPlan, files: Dict[str, str], error_excerpt: str, attempt: int, task: str) -> str:
        return (f"# Experiment intent\nHYPOTHESIS: {plan.hypothesis}\nCHANGE SPEC:\n{plan.change_spec}\n\n"
                f"# Failing files (debug attempt {attempt})\n{render_file_blocks(files)}\n\n"
                f"# Error (last lines of the run)\n```\n{error_excerpt}\n```\n\n# TASK\n{task}")

    def debugger(self, plan: ResearcherPlan, files: Dict[str, str], error_excerpt: str, attempt: int) -> DebugOutcome:
        _, task = load_prompt(self.prompts_dir, "debugger")
        messages = [{"role": "user", "content": self.debugger_message(plan, files, error_excerpt, attempt, task)}]
        try:
            resp = self._call("debugger", self._system_blocks("debugger"), messages, f"debugger_{attempt}")
        except LLMError as e:
            return DebugOutcome(action="abandon", reason=f"debugger_llm_error: {e}")
        try:
            return parse_debugger(resp.text)
        except ContractError as e:
            err1 = str(e)
        messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": REASK.format(error=err1)}]
        try:
            resp2 = self._call("debugger", self._system_blocks("debugger"), messages, f"debugger_{attempt}_reask", attempt=2)
            return parse_debugger(resp2.text)
        except (LLMError, ContractError) as e:
            return DebugOutcome(action="abandon", reason=f"debugger_malformed: {e}")

    # -- scribe --------------------------------------------------------------
    def scribe_lesson(self, plan: ResearcherPlan, result: HarnessResult, decision: str, best_primary: Optional[float]) -> str:
        """≤ 20 words, single line; the harness truncates whatever comes back and never lets it be empty."""
        _, task = load_prompt(self.prompts_dir, "scribe_lesson")
        facts = (f"HYPOTHESIS: {plan.hypothesis}\nCATEGORY: {plan.category}\nRESULT: {json.dumps(result.to_dict())}\n"
                 f"DECISION (harness): {decision}\nBEST PRIMARY AFTER: {best_primary}\n")
        messages = [{"role": "user", "content": f"# Facts (measured by the harness)\n{facts}\n# TASK\n{task}"}]
        try:
            resp = self._call("scribe_lesson", self._system_blocks("scribe_lesson"), messages, "scribe_lesson")
            lesson = truncate_words(resp.text.strip().strip('"'), 20)
        except LLMError as e:
            lesson = ""
            self.last_error = f"scribe_lesson_llm_error: {e}"
        if not lesson:
            lesson = truncate_words(f"{decision}: {result.status} {result.vs_best} for {plan.hypothesis}", 20)
        return lesson

    def scribe_logentry(self, facts: Dict[str, Any]) -> str:
        _, task = load_prompt(self.prompts_dir, "scribe_logentry")
        messages = [{"role": "user", "content": f"# Facts (measured by the harness)\n```json\n{json.dumps(facts, indent=1, default=str)}\n```\n\n# TASK\n{task}"}]
        try:
            resp = self._call("scribe_logentry", self._system_blocks("scribe_logentry"), messages, "scribe_logentry")
            return resp.text.strip()
        except LLMError as e:
            self.last_error = f"scribe_logentry_llm_error: {e}"
            return f"(narrative unavailable: {e})"
