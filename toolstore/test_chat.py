"""Regression tests for answer_question()'s branch order and its live
search guard.

Both cases below were real, reproduced failures, not hypotheticals: the
branch order shipped with `source is not None` ahead of the refusal check,
which printed "I don't have information about that." tagged as a sourced
answer and never ran the fallback, and the fallback call site had no
exception guard of its own, so a raise out of web_search() came straight
back out of answer_question() and (via gui.py's worker thread or the CLI
loop) ended the session instead of the turn.

No GPU and no checkpoint needed: load() returns immediately once
chat._model is set, so a tiny fake model, encoder and decoder let the real
answer_question() control flow run against controlled "generated" text.

Run with: python -m unittest discover -s toolstore
"""
import unittest
from contextlib import nullcontext
from unittest import mock

import torch

import chat

# The machine this trains on runs SFT on the GPU for days at a time, near
# its VRAM limit. chat.py picks device='cuda' at import when it can, which
# would put this test's fake tensors on the same card for no reason.
chat.device = "cpu"

REFUSAL = "I don't have information about that."
SEARCH_RESULT = "Paris is the capital of France. (source: example.com)"
# Real retrieve() result shape, one reranked doc, as build_prompt() reads it.
SOURCE = {
    "content": "Paris is the capital of France.",
    "metadata": {"rowid": 1, "type": "fact", "title": "Paris"},
    "score": 9.0,
}


class _FakeModel:
    def generate(self, x, max_new_tokens, temperature, top_k):
        # Length only, the faked _decode ignores the ids entirely.
        return torch.cat([x, torch.zeros((1, 5), dtype=torch.long)], dim=1)


class _ChatFlowTest(unittest.TestCase):
    def install(self, generated_answer):
        chat._model = _FakeModel()
        chat._encode = lambda s: [0] * max(1, len(s) // 4)
        chat._decode = lambda ids: generated_answer
        chat._ctx = nullcontext()
        self.addCleanup(setattr, chat, "_model", None)


class TestRefusalFallsBackToSearch(_ChatFlowTest):
    def test_no_source_and_refusal_searches(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT) as m_web:
            out = chat.answer_question("What is the capital of Freedonia?")
        self.assertEqual(m_web.call_count, 1)
        self.assertIn(SEARCH_RESULT, out)
        self.assertIn("[no local match, fell back to live search]", out)

    def test_source_present_and_refusal_still_searches(self):
        """The one the branch order got wrong. A source clearing the rerank
        threshold does not mean the model used it, and a refusal tagged
        [source: ...] is self contradictory to the user.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT) as m_web:
            out = chat.answer_question("What is the capital of France?")
        self.assertEqual(m_web.call_count, 1)
        self.assertIn(SEARCH_RESULT, out)
        self.assertNotIn(REFUSAL, out)
        self.assertNotIn("[source:", out)
        self.assertIn("[model refused the local match, fell back to live search]", out)

    def test_greeting_reply_does_not_search(self):
        """Greetings train a different, varied reply for the same
        Context: (none) shape. Searching DuckDuckGo for "hi" returns noise.
        """
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("hi")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[no confident source found", out)

    def test_sourced_answer_is_still_cited(self):
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the capital of France?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[source: Paris, rerank score: 9.00]", out)

    def test_tool_call_wins_over_every_other_branch(self):
        self.install("CALL: calculator(2+2)")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is 2 plus 2?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[tool: calculator]", out)


class TestSearchFailureEndsTheTurnNotTheSession(_ChatFlowTest):
    def test_fallback_survives_a_raising_web_search(self):
        """web_search.py catching Exception broadly is the first line of
        defense, not the only one: its own docstring records a real crash
        that killed the session when a failure escaped its except clause.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", side_effect=RuntimeError("ddgs internal crash")):
            out = chat.answer_question("What is the boiling point of mercury?")
        self.assertIn("[web search error: ddgs internal crash]", out)
        self.assertIn("fell back to live search", out)

    def test_web_prefix_survives_a_raising_web_search(self):
        self.install("unused")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search", side_effect=RuntimeError("ddgs internal crash")):
            out = chat.answer_question("web: capital of france")
        self.assertFalse(m_retrieve.called)
        self.assertIn("[web search error: ddgs internal crash]", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
