"""The swappable retrieval interface chat.py depends on, not query.py's
FAISS/sqlite internals directly. Mirrors the real world precedent
(LangChain's BaseRetriever: query: str -> list[Document], Document =
page_content + free form metadata dict, confirmed by reading its actual
source), in plain Python via typing.Protocol rather than LangChain's own
ABC, since there is no shared implementation to inherit here, just a shape
to match, structural typing needs no base class at all. One implementation
exists today (query.py's FAISS + cross encoder stack); a future one (a web
search API, a different vector store, anything else) only needs to match
this same shape, chat.py would not change.
"""
from typing import Protocol, TypedDict


class RetrievedDoc(TypedDict):
    content: str
    metadata: dict
    score: float


class Retriever(Protocol):
    def retrieve(self, query_text: str) -> list[RetrievedDoc]:
        """Return the top matches for query_text, best first, empty list if
        nothing relevant. No db handle, no backend specific tuning params
        (k, thresholds, connection objects) in this signature, those belong
        on the concrete retriever's own module level config or __init__,
        never in the call chat.py makes.
        """
        ...
