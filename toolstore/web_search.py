"""Live DuckDuckGo web search, chat.py's "web" retrieval choice, never used
to build SFT training data (build_sft_dataset.py has no import of this
file or toolstore/, confirmed, that isolation is structural not just
intended: DuckDuckGo rate limits requests, so it cannot be used to bulk
generate a training corpus the way the Wikipedia summaries were).

chat.py's run_web_search() now feeds a real result back through
_model.generate() (changed on direct instruction: "talk to ai using rag
should ALWAYS get a response made with AI generated on the spot"), not
printed as-is the way this docstring used to describe: the SFT model only
ever saw Wikipedia summary shaped Context: blocks paired with a fixed
"What is {title}?" question, zero exposure to search snippet shaped
content or arbitrary question phrasing paired with real context, so there
is no trained basis for it to reliably paraphrase a live web result well —
a real, accepted quality tradeoff, not an oversight (this project's own
standing priority: a worse but genuinely self-generated answer over a
better one that leans on unmodified outside text). The one part of the old
reasoning that still holds and is still enforced structurally, not just by
convention: chat.py's live-search-generated answer is never passed to
CALL_RE/dispatch_call() the way its normal answer is, so untrusted web text
reaching the model still cannot make it to a real tool dispatch, closing
the prompt injection surface the old design called out without needing to
keep the whole result unrouted through generate() to do it.

ddgs (github.com/deedy5/ddgs, renamed from duckduckgo_search) confirmed
by reading its own real source: DDGS().text() needs no context manager,
returns list[dict] with title/href/body keys, and its own exceptions.py
shows RatelimitException is a real subclass of the base DDGSException,
but a live, currently open upstream issue (deedy5/ddgs#478) confirms rate
limit errors sometimes surface as the generic base class instead, so the
base class is what gets caught here, not just the specific subclass.
"""
from ddgs import DDGS


class WebSearchError(Exception):
    """Any failure of the live search itself, normalized to one type.

    Every outcome of web_search() below has to be distinguishable by
    chat.py, which shows a real result to the model and a failure to the
    user verbatim. Both used to come back as a plain string, so the caller
    had to recover the difference by sniffing a bracketed prefix off text
    whose body half is page content DuckDuckGo returns, not this project's
    own string. bad-cop proved that collides for real: a result whose body
    starts with "[web search error:" was classified as a diagnostic and
    printed to the user completely untouched by the model (measured: one
    generate() call for a turn that owed two). Raising for a failure and
    returning None for a genuine miss puts that signal out of band, where
    no page body can forge it — the same boundary shape psf/requests uses
    (HTTPAdapter.send catches urllib3's exceptions and re-raises requests'
    own ConnectionError/ReadTimeout rather than returning an error value).
    """


def web_search(query):
    """Returns the top DuckDuckGo hit as "body (source: url)", or None when
    the search genuinely found nothing. Raises WebSearchError if the lookup
    itself failed.

    Catches Exception broadly, not just ddgs.exceptions.DDGSException,
    mirroring chat.py's dispatch_call() at the other untrusted external I/O
    boundary. Found by bad-cop, real execution: ddgs 9.14.4's own internals
    aren't fully covered by its documented exception hierarchy (e.g. a bare
    ValueError path in its own ThreadPoolExecutor setup), and a real,
    unmocked test proved a failure that is not a DDGSException propagates
    uncaught and kills the entire chat session, not just this one turn,
    including a user's next, unrelated local question they never get a
    chance to ask. Normalizing every one of them to WebSearchError here is
    what makes that guarantee a caller can rely on; chat.py's
    run_web_search() still catches Exception, so anything that escapes this
    normalization ends the turn rather than the session.
    """
    try:
        results = DDGS().text(query, max_results=1)
    except Exception as e:
        raise WebSearchError(e) from e
    if not results:
        return None
    top = results[0]
    return f"{top['body']} (source: {top['href']})"
