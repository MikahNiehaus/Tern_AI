"""Live DuckDuckGo web search, chat.py's "web" retrieval choice, never used
to build SFT training data (build_sft_dataset.py has no import of this
file or toolstore/, confirmed, that isolation is structural not just
intended: DuckDuckGo rate limits requests, so it cannot be used to bulk
generate a training corpus the way the Wikipedia summaries were).

Result is printed directly, the same shape chat.py's dispatch_call()
already uses for tool output, never routed through model.generate(): the
SFT model (model/sft/build_sft_dataset.py) only ever saw Wikipedia
summary shaped Context: blocks paired with a fixed "What is {title}?"
question, zero exposure to search snippet shaped content or arbitrary
question phrasing paired with real context, so there is no trained basis
for it to safely paraphrase a live web result. It would also be a real
prompt injection surface (untrusted web text reaching the same prompt a
live CALL: tool dispatcher parses) for no benefit over just showing the
real result.

ddgs (github.com/deedy5/ddgs, renamed from duckduckgo_search) confirmed
by reading its own real source: DDGS().text() needs no context manager,
returns list[dict] with title/href/body keys, and its own exceptions.py
shows RatelimitException is a real subclass of the base DDGSException,
but a live, currently open upstream issue (deedy5/ddgs#478) confirms rate
limit errors sometimes surface as the generic base class instead, so the
base class is what gets caught here, not just the specific subclass.
"""
from ddgs import DDGS


def web_search(query):
    # Catches Exception broadly, not just ddgs.exceptions.DDGSException,
    # mirroring chat.py's dispatch_call() right next to this call site.
    # Found by bad-cop, real execution: ddgs 9.14.4's own internals aren't
    # fully covered by its documented exception hierarchy (e.g. a bare
    # ValueError path in its own ThreadPoolExecutor setup), and a real,
    # unmocked test proved a failure that is not a DDGSException propagates
    # uncaught and kills the entire chat session, not just this one turn,
    # including a user's next, unrelated local question they never get a
    # chance to ask. This is untrusted external I/O, same boundary
    # dispatch_call()'s own broad catch already exists to guard.
    try:
        results = DDGS().text(query, max_results=1)
    except Exception as e:
        return f"[web search error: {e}]"
    if not results:
        return "[web search: no results found]"
    top = results[0]
    return f"{top['body']} (source: {top['href']})"
