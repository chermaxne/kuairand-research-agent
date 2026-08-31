"""
Research tools for the Researcher role: arXiv search (free, no API key) and a scoped web fetch,
so the knowledge library can be refreshed with real, current papers instead of staying static
for the whole run.

IMPORTANT -- process boundary: these run inside the HARNESS process, when it calls out to the
LLM API on the Researcher's behalf. They are completely separate from sandbox.py's sandboxed
subprocess that runs generated pipeline.py -- that subprocess still has zero network access
(see README's guarantee table). Adding these tools gives the HARNESS PROCESS ITSELF a new
outbound network path it didn't have before; the experiment sandbox's isolation is unaffected.

STATED CONSTRAINT (enforced in code, not just convention): `web_fetch` only accepts arxiv.org
URLs -- i.e. only ever a URL `arxiv_search` itself returned, never an arbitrary address a model
or user supplies. This is what keeps the injection surface bounded to a curated, low-risk corpus;
if `web_fetch` is ever extended to a general search API, this restriction has to be revisited
deliberately, not silently dropped.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
FETCH_TIMEOUT_S = 10          # a hung fetch shouldn't eat the 6h wall-clock ceiling
FETCH_MAX_CHARS = 4000        # keep fetched content from blowing up the next briefing's tokens
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ALLOWED_FETCH_HOSTS = {"arxiv.org", "export.arxiv.org"}


@dataclass
class SearchResult:
    title: str
    authors: List[str]
    summary: str
    url: str
    published: str


def arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Free, keyless, structured -- a better fit for 'find papers relevant to X' than a general
    web search API, and it needs no API key to manage or leak."""
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results,
              "sortBy": "relevance"}
    resp = requests.get(ARXIV_API, params=params, timeout=FETCH_TIMEOUT_S)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    results = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
        url = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
        authors = [(a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "")
                   for a in entry.findall("atom:author", _ATOM_NS)]
        results.append(asdict(SearchResult(title=title, authors=authors,
                                            summary=summary[:FETCH_MAX_CHARS],
                                            url=url, published=published)))
    return results


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def web_fetch(url: str) -> str:
    """Plain GET + crude tag stripping, restricted to arxiv.org (see module docstring's STATED
    CONSTRAINT). Swap in `trafilatura` if you start fetching messier general web pages and the
    crude strip pulls in too much nav/footer junk -- but only after deliberately revisiting the
    host restriction, not as an incidental side effect."""
    host = (urlparse(url).hostname or "").lower()
    if host not in _ALLOWED_FETCH_HOSTS:
        raise ValueError(f"web_fetch only accepts arxiv.org URLs (got host {host!r}) -- "
                         f"only fetch a URL that arxiv_search itself returned")
    resp = requests.get(url, timeout=FETCH_TIMEOUT_S, headers={"User-Agent": "kuairand-agent/0.1"})
    resp.raise_for_status()
    text = _TAG_RE.sub(" ", resp.text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:FETCH_MAX_CHARS]


# ---------------------------------------------------------------------------
# Tool schemas -- OpenAI/OpenRouter function-calling shape. This is what most models served
# through OpenRouter (including glm-5.2, qwen3-coder) expect. If a role is called through a
# NATIVE Anthropic client instead of via OpenRouter, the tool block shape differs -- see the
# note at the bottom of docs/tool_loop_sketch.py.
# ---------------------------------------------------------------------------
RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": ("Search arXiv for papers relevant to a research direction. "
                             "Use sparingly -- at most 3 calls per iteration."),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a specific URL, e.g. an arXiv abstract page found via arxiv_search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

TOOL_EXECUTORS = {
    "arxiv_search": lambda args: arxiv_search(args["query"]),
    "web_fetch": lambda args: web_fetch(args["url"]),
}


# ---------------------------------------------------------------------------
# Knowledge-library append -- the harness's job, not the LLM's. The Researcher proposes a
# NEW_KNOWLEDGE field in its response (paraphrase + source URL, one string); this function is
# what actually writes it, with dedup and a size cap so a 50-iteration run can't make every
# future briefing bigger and bigger.
# ---------------------------------------------------------------------------
# The real knowledge/library.md is already ~23k chars at the time this was wired in -- cap set
# with real headroom above that, not the sketch's original 12000 (which would have made every
# append silently a no-op against the actual file, forever, with no error).
MAX_LIBRARY_CHARS = 40000


def append_knowledge(library_path: str, new_entry: str) -> bool:
    """Returns False (and does nothing) if the entry looks like a near-duplicate or the file is
    full -- a human should prune between runs in that case, not have the harness silently drop or
    silently keep growing the briefing every future iteration pays for."""
    new_entry = new_entry.strip()
    if not new_entry:
        return False
    with open(library_path, "r", encoding="utf-8") as f:
        current = f.read()
    fingerprint = new_entry[:80]
    if fingerprint in current:
        return False
    if len(current) + len(new_entry) > MAX_LIBRARY_CHARS:
        return False
    with open(library_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {new_entry}\n")
    return True


if __name__ == "__main__":
    # Standalone smoke test -- no LLM involved, just confirms the arXiv call itself works before
    # it's wired into anything load-bearing.
    for r in arxiv_search("multi-task learning recommendation click-through rate", max_results=3):
        print(r["title"], "--", r["url"])
