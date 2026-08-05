"""Regression tests for answer_question()'s branch order, its live search
guard, and its mode toggle.

Both cases below were real, reproduced failures, not hypotheticals: the
branch order shipped with `source is not None` ahead of the refusal check,
which printed "I don't have information about that." tagged as a sourced
answer and never ran the fallback, and the fallback call site had no
exception guard of its own, so a raise out of web_search() came straight
back out of answer_question() and (via gui.py's worker thread or the CLI
loop) ended the session instead of the turn.

TestModeToggle covers the mode= argument the GUI's Chat tab
passes, and it exists because of a measured hole rather than for symmetry:
this file had zero occurrences of `mode=` in it, so reverting
answer_question()'s `use_retrieval=(mode != "web")` back to a bare
build_prompt(question) left every test here passing, 0 failures, while
mode="web" silently went back to consulting the vector store. Two more
mutants measured against the tests below: dropping the `if mode ==
"vector"` guard is caught only by test_vector_mode_refusal_does_not_search,
and moving the "web:" prefix check after the mode logic is caught only by
test_web_prefix_overrides_every_mode.

The last three classes are the same kind of measured hole, one round later,
for answer_turn()'s dict return. Every test above calls answer_question(),
which is now a wrapper that throws "context" and "source_title" away, so
the whole data path the GUI's RAG context viewer reads had zero assertions
on it: setting "context" to None unconditionally on BOTH the sourced branch
and the refusal-with-source fallback branch left all 44 tests above passing,
0 failures. TestBuildPromptContext, TestAnswerTurnContext and
TestAnswerQuestionMatchesAnswerTurn call answer_turn()/build_prompt()
directly for that reason, and they kill that mutant on both lines.

The logging classes at the end are the round after that, and they are
about the per turn log rather than the answer. Two of them assert measured
failures of the first version of it: a real write failure (the FileHandler's
own stream raising OSError, not a mocked chat.logger.info) never reached
_log_turn's except block at all, because logging.Handler.emit() catches it
first and hands it to Handler.handleError(), which dumps a raw traceback to
stderr and returns normally — 1421 bytes of uncaptured stderr and zero
logger.error() calls, against a docstring promising the opposite — and the
module level log directory setup ran unguarded at import, so a logs/ path
occupied by a plain file raised FileExistsError out of `import chat` and took
gui.py's Chat tab and the CLI loop with it. Both are driven through the real
handler here, never a mock of the logger, since mocking chat.logger.info is
exactly what hid the first bug.

TestOneBrokenHandlerDoesNotBreakTheOthers is the third measured failure of
that same handler, one round later again: the first fix for the write failure
above re-raised out of handleError, which aborts Logger.callHandlers()'s
single walk over this logger's and every ancestor logger's handlers, so
gui.py's root handler received nothing at all for a turn whose file write
failed. Every other logging test here drives a logger with exactly one
handler, which is why none of them could see it.

No GPU and no checkpoint needed: load() returns immediately once
chat._model is set, so a tiny fake model, encoder and decoder let the real
answer_question() control flow run against controlled "generated" text.

Run with: python -m unittest discover -s toolstore
"""
import importlib.util
import io
import logging
import os
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext, redirect_stderr
from unittest import mock

import tiktoken
import torch

import chat
import web_search

# The machine this trains on runs SFT on the GPU for days at a time, near
# its VRAM limit. chat.py picks device='cuda' at import when it can, which
# would put this test's fake tensors on the same card for no reason.
chat.device = "cpu"

_real_chat_log_handlers = None
_real_get_paraphrase_titles = None


def setUpModule():
    """Every test in this file drives the real answer_turn()/_log_turn(),
    which would otherwise append hundreds of mocked, fake-fixture turns
    ("Paris is the capital of France" repeated dozens of times) into the
    real logs/chat_turns.log a user reads to review actual conversations.
    unittest's own setUpModule/tearDownModule hooks (run once for the whole
    file, not per test) are the standard place for this, not a per test
    mock: every class here would otherwise need the same patch repeated.

    Also patches chat._get_paraphrase_titles() to always accept SOURCE's one
    hardcoded title, "Paris": that gate now reads model/sft's real, built
    trained_rag_titles.json (the file this project's own dataset build
    writes), not a self-contained fact about the test fixture, so a test
    environment that has never run a real SFT dataset build (no such file
    on disk yet) would otherwise gate every one of the many SOURCE-based
    tests below out to the no-match branch, none of which are testing that
    gating behavior at all. TestParaphraseTitleGating overrides this default
    per test to actually exercise the gate itself.
    """
    global _real_chat_log_handlers, _real_get_paraphrase_titles
    _real_chat_log_handlers = chat.logger.handlers
    chat.logger.handlers = [logging.NullHandler()]
    _real_get_paraphrase_titles = chat._get_paraphrase_titles
    chat._get_paraphrase_titles = lambda: {"Paris"}


def tearDownModule():
    chat.logger.handlers = _real_chat_log_handlers
    chat._get_paraphrase_titles = _real_get_paraphrase_titles

REFUSAL = "I don't have information about that."
SEARCH_RESULT = "Paris is the capital of France. (source: example.com)"
# Real retrieve() result shape, one reranked doc, as build_prompt() reads it.
SOURCE = {
    "content": "Paris is the capital of France.",
    "metadata": {"rowid": 1, "type": "fact", "title": "Paris"},
    "score": 9.0,
}
# Two real corpus/wikipedia_summaries/summaries.tsv rows, copied verbatim.
# rag_examples() trains the sourced answer to continue with exactly this text,
# and the first sentence of each contains a period that is NOT a sentence end.
COUNTY_SUMMARY = (
    "Jackson County is a county located in the U.S. state of Illinois "
    "with a population of 57,230 at the 2020 census."
)
RIFLE_SUMMARY = (
    "The Karabiner 98k is a bolt-action rifle chambered for the "
    "7.92x57mm Mauser cartridge. It was the standard German service rifle."
)
# A third real row, verbatim (title "Tara Murphy"). 638 rows of the real
# summaries.tsv open with a 4+ letter title abbreviation like this one.
PROFESSOR_SUMMARY = (
    "Prof. Tara Murphy is an Australian Astrophysicist and CAASTRO chief "
    "investigator working at the University of Sydney."
)


class _FakeModel:
    def generate(self, x, max_new_tokens, temperature, top_k):
        # Length only, the faked _decode ignores the ids entirely.
        return torch.cat([x, torch.zeros((1, 5), dtype=torch.long)], dim=1)


class _ChatFlowTest(unittest.TestCase):
    def install(self, generated_answer):
        self.install_model()
        chat._encode = lambda s: [0] * max(1, len(s) // 4)
        chat._decode = lambda ids: generated_answer

    def install_model(self):
        """The model half of install(), on its own, for the tests that want
        the real tokenizer rather than the length//4 stand in.
        """
        chat._model = _FakeModel()
        chat._ctx = nullcontext()
        self.addCleanup(setattr, chat, "_model", None)

    def use_real_temp_log(self):
        """Points chat.logger at a REAL ChatTurnFileHandler on a throwaway
        file, for the tests that have to read back what was written or make
        the write itself fail. setUpModule's NullHandler is enough for every
        other test here and is restored afterwards; the production handler
        class, not a plain logging.FileHandler, because its overridden
        handleError() is the whole subject of TestLogWriteFailureIsReported.
        Returns (path, handler).
        """
        tmpdir = tempfile.mkdtemp(prefix="chat_log_test_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        path = os.path.join(tmpdir, "chat_turns.log")
        handler = chat.ChatTurnFileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        saved = chat.logger.handlers
        chat.logger.handlers = [handler]
        self.addCleanup(setattr, chat.logger, "handlers", saved)
        self.addCleanup(handler.close)
        return path, handler


class TestRefusalFallsBackToSearch(_ChatFlowTest):
    def test_no_source_and_refusal_searches(self):
        """The fallback answer is the model generating on the spot from the
        search result (the "_FakeModel"/install() stand-in always decodes to
        the same installed text regardless of input, so it comes back as
        REFUSAL again here — a real model would generate something that
        actually uses the search content, but the property under test is
        that the raw DuckDuckGo text is never shown untouched, which holds
        either way).
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT) as m_web:
            out = chat.answer_question("What is the capital of Freedonia?")
        self.assertEqual(m_web.call_count, 1)
        self.assertNotIn(SEARCH_RESULT, out,
                         "the raw DuckDuckGo text must never be shown untouched, "
                         "only what the model generates from it")
        self.assertIn(REFUSAL, out)
        self.assertIn("[no local match, fell back to live search, AI generated from the search result]", out)

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
        self.assertNotIn(SEARCH_RESULT, out,
                         "the raw DuckDuckGo text must never be shown untouched, "
                         "only what the model generates from it")
        self.assertNotIn("[source:", out)
        self.assertIn("[model refused the local match, fell back to live search, AI generated from the search result]", out)

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

    def test_greeting_reply_under_web_mode_says_vector_store_off(self):
        """Same greeting, mode="web": retrieve() never runs there
        (use_retrieval=False), so "no confident source found" would falsely
        claim a search happened and came up empty. Same distinction the
        refusal-fallback branch already draws for this mode.
        """
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("hi", mode="web")
        self.assertFalse(m_retrieve.called)
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[vector store off", out)
        self.assertNotIn("no confident source found", out)

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


class TestFirstSentenceCut(_ChatFlowTest):
    """The sourced branch's cut down to one sentence. These all assert on
    the answer line only, so the citation tag's title (always SOURCE's
    "Paris") is irrelevant to them.

    The two abbreviation/decimal cases below were real, reproduced
    failures of the first heuristic shipped here (a plain
    r"[^.!?]+[.!?]*"), measured over every 300th row of the real
    summaries.tsv: it cut 5.2% of them mid-abbreviation or mid-number,
    printing "...a county located in the U." and "...chambered for the 7.".

    The title-abbreviation cases after them were the reproduced failure of
    the heuristic that replaced it (a lookbehind stack enumerating
    abbreviations by letter count, 1 through 3). Any longer title fell
    through it and was read as a real sentence end, so 638 real corpus rows
    opening "Prof. <Name> is ..." printed as the single word "Prof." — a
    worse failure than the one it fixed, since a fragment at least still
    carries facts and a bare abbreviation carries none.

    The stray-")" and unlisted-abbreviation cases at the end are the two real
    gaps found in the guards themselves: the paren guard reverted a good cut on
    any bracket mismatch rather than on an unclosed one, and the word floor
    only ever caught a mis-cut that landed inside the first three words.
    """

    def test_sourced_answer_is_cut_to_its_first_sentence(self):
        """rag_examples() trains the full Wikipedia summary as the answer,
        which reads as a paragraph, not a concise reply. The real reported
        case: "As the capital of France, Paris is the seat of France's
        national government. For the executive, the two chief officers..."
        should print only the first sentence.
        """
        self.install(
            "As the capital of France, Paris is the seat of France's national government. "
            "For the executive, the two chief officers each have their own official residences."
        )
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the capital of France?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(
            "As the capital of France, Paris is the seat of France's national government.\n[source:",
            out,
        )
        self.assertNotIn("For the executive", out)

    def test_abbreviation_is_not_a_sentence_end(self):
        """U.S. county/township/city articles are one of the largest
        structurally repeated title categories in enwiki, and one of the
        query classes ("what is X county?") this cut exists to serve, so
        cutting at "the U." both breaks the English and drops the part
        the user asked for (which state).
        """
        self.install(COUNTY_SUMMARY)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Jackson County, Illinois?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{COUNTY_SUMMARY}\n[source:", out)

    def test_decimal_point_is_not_a_sentence_end(self):
        """Same shape, a number instead of an abbreviation: the answer for
        a rifle caliber was cut to "...chambered for the 7.".
        """
        self.install(RIFLE_SUMMARY)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the Karabiner 98k?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(
            "The Karabiner 98k is a bolt-action rifle chambered for the "
            "7.92x57mm Mauser cartridge.\n[source:",
            out,
        )
        self.assertNotIn("standard German service rifle", out)

    def test_four_letter_title_is_not_a_sentence_end(self):
        """The reported case: "Prof. Tara Murphy is an Australian
        Astrophysicist..." printed as "Prof." and nothing else.
        """
        self.install(PROFESSOR_SUMMARY)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("Who is Tara Murphy?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{PROFESSOR_SUMMARY}\n[source:", out)

    def test_five_letter_title_is_not_a_sentence_end(self):
        """Same shape one letter longer, the gap after the one that was
        reported. "Comdr." is in ABBREVIATIONS, so unlike the unknown
        abbreviation case below this one is caught by the lexicon itself, not
        by the fallback: a lookbehind enumerating letter counts would need yet
        another entry for it, and another after that for "Messrs.".
        """
        summary = "Comdr. Harold Baxter commanded the vessel during the Atlantic crossing of 1943."
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("Who was Harold Baxter?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{summary}\n[source:", out)

    def test_unknown_abbreviation_keeps_the_full_paragraph(self):
        """The one that decides whether there is a round four. No lexicon is
        complete, so this asserts on the FAILURE mode rather than on a fix:
        "Gebr." is deliberately not in ABBREVIATIONS, SENTENCE_BOUNDARY_RE
        does split it wrongly, and MIN_SENTENCE_WORDS turns that into the
        pre-cut behavior (whole paragraph, every fact intact) instead of a
        one word non answer.
        """
        summary = ("Gebr. Heinemann is a German retail company operating duty "
                   "free shops in airports. It was founded in 1879.")
        self.assertEqual(chat.SENTENCE_BOUNDARY_RE.split(summary, maxsplit=1)[0], "Gebr.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Gebr. Heinemann?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{summary}\n[source:", out)

    def test_two_abbreviations_in_the_first_sentence(self):
        """A title at the start and another mid sentence, so a rule that
        only guards the sentence initial position still cuts at the second.
        """
        self.install("Prof. Alan Reed of St. Andrews University studies marine "
                     "biology. He has published widely on coral reefs.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("Who is Alan Reed?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Prof. Alan Reed of St. Andrews University studies marine "
                      "biology.\n[source:", out)
        self.assertNotIn("coral reefs", out)

    def test_sentence_really_ending_after_an_abbreviation_is_still_cut(self):
        """The opposite direction: an abbreviation the lexicon knows, in a
        sentence that genuinely ends right after it. Protecting "U.S." must
        not mean the cut never happens at all.
        """
        self.install("The U.S. flag is red, white, and blue. It has fifty stars, "
                     "one for each state.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What does the U.S. flag look like?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("The U.S. flag is red, white, and blue.\n[source:", out)
        self.assertNotIn("fifty stars", out)

    def test_open_parenthesis_keeps_the_full_paragraph(self):
        """The other way a cut lands somewhere that is not a sentence end:
        inside a parenthetical. A real corpus row (title "Nez Perce County,
        Idaho"), and the piece the regex keeps is exactly 4 words, so it clears
        MIN_SENTENCE_WORDS — only the unclosed "(" saves it.
        """
        summary = ("Nez Perce County (pron. Nezz Purse) is a county located in "
                   "the U.S. state of Idaho. As of the 2010 census, the "
                   "population was 39,265.")
        raw = chat.SENTENCE_BOUNDARY_RE.split(summary, maxsplit=1)[0]
        self.assertEqual(raw, "Nez Perce County (pron.")
        self.assertGreaterEqual(len(raw.split()), chat.MIN_SENTENCE_WORDS,
                                "the word floor must not be what saves this row")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Nez Perce County?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{summary}\n[source:", out)

    def test_nested_parentheses_keep_the_full_paragraph(self):
        """The paren guard counts, it does not just look for a "(", so a
        parenthetical that opens, closes an inner one, and is cut before its
        own close is still caught (2 opens against 1 close).
        """
        summary = ("The Act 1811 (51 Vict. (as amended) III, c. 23. Was a piece "
                   "of legislation. It was repealed later.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the Act 1811?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{summary}\n[source:", out)

    def test_regnal_year_abbreviation_is_not_a_sentence_end(self):
        """The row the paren guard used to be tested with. "Geo." (the regnal
        year "51 Geo. III") is in CORPUS_ABBREVIATIONS now, so this is caught by
        the lexicon before the guards see it, and the real first sentence is
        cut rather than the whole paragraph being kept.
        """
        summary = ("The Slave Trade Felony Act 1811 (51 Geo. III, c. 23) was a "
                   "piece of British legislation that made engagement in the "
                   "slave trade a felony. The earlier Slave Trade Act 1807 "
                   "merely imposed fines.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the Slave Trade Felony Act 1811?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("The Slave Trade Felony Act 1811 (51 Geo. III, c. 23) was "
                      "a piece of British legislation that made engagement in "
                      "the slave trade a felony.\n[source:", out)
        self.assertNotIn("merely imposed fines", out)

    def test_stray_close_paren_is_not_a_mid_parenthetical_cut(self):
        """Real corpus row (title "Goalpara"). Upstream wiki markup stripping
        drops an {{IPA}}/pronunciation template and leaves its ")" behind, so
        the first sentence carries a ")" that never had a "(". That is in the
        text before any cut happens and says nothing about where the cut
        landed, so it must not revert a correctly bounded sentence — which is
        why the guard tests for more opens than closes, not for any mismatch.
        14 rows in a 37,223 row sample had this shape.
        """
        summary = ("Goalpara, Pron: ) is the district headquarters of Goalpara "
                   "district, Assam, India. It is situated  to the west of "
                   "Guwahati.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Goalpara?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Goalpara, Pron: ) is the district headquarters of "
                      "Goalpara district, Assam, India.\n[source:", out)
        self.assertNotIn("west of Guwahati", out)

    def test_stray_ipa_markers_are_not_a_mid_parenthetical_cut(self):
        """Same shape, real corpus row (title "Bishopbriggs"): "(; ); )" is
        what survives of a stripped IPA template, three markers with no real
        opening "(" for the last one.
        """
        summary = ("Bishopbriggs (; ); ) is a town in East Dunbartonshire, "
                   "Scotland. It lies on the northern fringe of Greater "
                   "Glasgow.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Bishopbriggs?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Bishopbriggs (; ); ) is a town in East Dunbartonshire, "
                      "Scotland.\n[source:", out)
        self.assertNotIn("Greater Glasgow", out)

    def test_approx_is_not_a_sentence_end(self):
        """MIN_SENTENCE_WORDS is positional: it only catches a mis-cut inside
        the first three words, and "The bridge, built approx." is exactly 4.
        "approx" is one of the common English abbreviations pysbd's list omits,
        so the lexicon is what has to catch this, not the floor.
        """
        summary = ("The bridge, built approx. Twenty years after the war, was "
                   "renovated in 2005 by the city council. It remains in use "
                   "today.")
        self.assertEqual(len("The bridge, built approx.".split()),
                         chat.MIN_SENTENCE_WORDS,
                         "sanity: the bad cut sits exactly at the word floor")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the bridge?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("The bridge, built approx. Twenty years after the war, "
                      "was renovated in 2005 by the city council.\n[source:", out)
        self.assertNotIn("remains in use today", out)

    def test_natl_is_not_a_sentence_end(self):
        """Seven words in, well clear of the floor, so only the lexicon can
        catch it.
        """
        summary = ("The organization changed its name from Natl. Cotton Council "
                   "to the current name in 1985. It remains active today.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the Natl. Cotton Council?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("The organization changed its name from Natl. Cotton "
                      "Council to the current name in 1985.\n[source:", out)
        self.assertNotIn("remains active today", out)

    def test_intl_is_not_a_sentence_end(self):
        summary = ("The society was formerly known as the Intl. Chess Federation "
                   "before the 1950 reorganization. It now has over 100 members.")
        self.install(summary)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the Intl. Chess Federation?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("The society was formerly known as the Intl. Chess "
                      "Federation before the 1950 reorganization.\n[source:", out)
        self.assertNotIn("over 100 members", out)

    def test_units_and_club_names_are_still_cut(self):
        """The other direction on the lexicon: entries were chosen by reading
        the real cuts they would destroy, so units and the two letter club and
        state codes that genuinely END Wikipedia sentences were deliberately
        left out ("km" alone ends 7 correct cuts in the sample). If someone
        adds them, these stop being cut.
        """
        for summary, expected in (
            ("The Vit is a river in Bulgaria with a length of 188 km. It flows "
             "north.", "The Vit is a river in Bulgaria with a length of 188 km."),
            ("Raul plays for Burgos CF. He joined in 2019.",
             "Raul plays for Burgos CF."),
        ):
            with self.subTest(summary=summary):
                self.install(summary)
                with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                        mock.patch.object(chat, "web_search"):
                    out = chat.answer_question("What is it?")
                self.assertIn(f"{expected}\n[source:", out)

    def test_word_like_abbreviations_do_not_block_a_real_sentence_end(self):
        """The round five audit, and the reason the lookbehind no longer
        carries pysbd's whole list. pysbd applies a general ABBREVIATIONS entry
        only when the next character is NOT a capital (scan_for_replacements:
        "if not upper or am.strip().lower() in prepositive"); only its
        PREPOSITIVE_ABBREVIATIONS suppress a period before a capital. This
        lookbehind fires only before a capital, so every entry was getting
        prepositive treatment and ordinary words were swallowing real sentence
        ends: "art" 1520 rows of the corpus, "inc" 1784, "ltd" 828, "may" 431,
        "etc" 549, against 2, 31, 9, 0 and 4 rows each rescued from a mis-cut.
        All five rows below are verbatim corpus rows.
        """
        for summary, expected in (
            ("An artist is a person engaged in an activity related to creating "
             "art, practicing the arts, or demonstrating an art. The common usage "
             "in both everyday speech and academic discourse refers to a "
             "practitioner in the visual arts only.",
             "An artist is a person engaged in an activity related to creating "
             "art, practicing the arts, or demonstrating an art."),
            ("Apple Watch is a line of smartwatches produced by Apple Inc. It "
             "incorporates fitness tracking, health-oriented capabilities, and "
             "wireless telecommunication.",
             "Apple Watch is a line of smartwatches produced by Apple Inc."),
            ("The Phoenix Living Poets was a series of slim books of poetry "
             "published from 1960 until 1983 by Chatto and Windus Ltd. The poets "
             "included in the series offer a cross-section of poets of the era.",
             "The Phoenix Living Poets was a series of slim books of poetry "
             "published from 1960 until 1983 by Chatto and Windus Ltd."),
            ("May Day is a public holiday, in some regions, usually celebrated "
             "on 1 May or the first Monday of May. It is an ancient festival "
             "marking the first day of summer.",
             "May Day is a public holiday, in some regions, usually celebrated "
             "on 1 May or the first Monday of May."),
            ("A museum is distinguished by a collection of often unique objects "
             "that forms the core of its activities for exhibitions, education, "
             "research, etc. This differentiates it from an archive or library.",
             "A museum is distinguished by a collection of often unique objects "
             "that forms the core of its activities for exhibitions, education, "
             "research, etc."),
        ):
            with self.subTest(expected=expected):
                self.install(summary)
                with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                        mock.patch.object(chat, "web_search"):
                    out = chat.answer_question("What is it?")
                self.assertIn(f"{expected}\n[source:", out)

    def test_corpus_entries_dropped_by_the_audit_no_longer_block(self):
        """The same audit run against round four's own additions. Three of the
        31 measured net negative and were removed: "fl" (17 rows rescued
        against 61 swallowed — the US state code ends far more real sentences
        than the Latin floruit begins), "spp" (2/12) and "seq" (0/6, never a
        rescue). Verbatim corpus rows.
        """
        for summary, expected in (
            ("Camillo Ricordi, (born 1957) is a diabetes researcher based in "
             "Miami, FL. He currently serves as Director of the Diabetes "
             "Research Institute, a position he has held since 1996.",
             "Camillo Ricordi, (born 1957) is a diabetes researcher based in "
             "Miami, FL."),
            ("β-Zearalenol is a nonsteroidal estrogen of the resorcylic "
             "acid lactone group related to mycoestrogens found in Fusarium spp. "
             "It is the β epimer of α-zearalenol.",
             "β-Zearalenol is a nonsteroidal estrogen of the resorcylic "
             "acid lactone group related to mycoestrogens found in Fusarium spp."),
            ("The Communications Act of 1934 is a United States federal law "
             "signed by President Franklin D. Roosevelt on June 19, 1934 and "
             "codified as Chapter 5 of Title 47 of the United States Code,  et "
             "seq. The Act replaced the Federal Radio Commission with the "
             "Federal Communications Commission (FCC).",
             "The Communications Act of 1934 is a United States federal law "
             "signed by President Franklin D. Roosevelt on June 19, 1934 and "
             "codified as Chapter 5 of Title 47 of the United States Code,  et "
             "seq."),
        ):
            with self.subTest(expected=expected):
                self.install(summary)
                with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                        mock.patch.object(chat, "web_search"):
                    out = chat.answer_question("What is it?")
                self.assertIn(f"{expected}\n[source:", out)

    def test_word_like_abbreviations_that_earn_their_place_still_block(self):
        """The other half of the same audit, and the reason it was run per
        entry rather than by dropping everything word-like. These three look
        exactly as droppable as the five above and measure the opposite way,
        because what follows them is a name and not a sentence: "bros"
        1803 rows rescued against 127, "jr" 1333/474, "ft" 72/13. Dropping any
        of them truncates the name instead. Verbatim corpus rows.
        """
        for summary in (
            "Cartoon Network Studios is an American animation studio owned by "
            "the Global Kids, Young Adults & Classics division of Warner Bros. "
            "Entertainment, a subsidiary of AT&T's WarnerMedia.",
            "The Otherwise Award, formerly known as the James Tiptree Jr. "
            "Award, is an annual literary prize for works of science fiction.",
            "Mark Thomas Griffin (born 1957), better known as MC 900 Ft. Jesus, "
            "is an American rapper from  Kentucky.",
        ):
            with self.subTest(summary=summary):
                self.assertEqual(chat.first_sentence(summary), summary)

    def test_gloss_abbreviations_still_block_their_own_translation(self):
        """What "lit" and "trans" are in the lexicon FOR: the translation gloss
        that opens a large class of articles. 2214 of "lit"'s 2287 blocked
        boundaries and 407 of "trans"'s 417 are this shape. Verbatim corpus
        rows.
        """
        for summary, expected in (
            ("Hősök tere (), lit. Heroes' Square, is one of the major "
             "squares in Budapest, Hungary, noted for its iconic Millennium "
             "Monument. The square lies at the outbound end of Andrássy "
             "Avenue next to City Park.",
             "Hősök tere (), lit. Heroes' Square, is one of the major "
             "squares in Budapest, Hungary, noted for its iconic Millennium "
             "Monument."),
            ("Dhoom 2 (trans. Blast 2) is a 2006 Indian Hindi-language action "
             "thriller heist film directed by Sanjay Gadhvi. The film stars "
             "Hrithik Roshan and Abhishek Bachchan.",
             "Dhoom 2 (trans. Blast 2) is a 2006 Indian Hindi-language action "
             "thriller heist film directed by Sanjay Gadhvi."),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(chat.first_sentence(summary), expected)

    def test_gloss_abbreviation_as_an_ordinary_word_is_the_measured_residual(self):
        """The reported round five failure, kept as an asserted residual rather
        than "fixed", because the fix it implies measures worse than the bug.
        Both rows below are real and really do come back whole: "Trans" is Neil
        Young's 1982 album and "lit" is the participle of "light", and the
        lexicon cannot tell either from the gloss the entry exists for.

        Dropping the two entries to fix these rows was measured over all
        6,439,528 corpus rows and trades 1634 correct cuts for 49 on "lit", and
        188 for 5 on "trans" — the same failure these rows show, 33 times more
        often. 54 rows in 6.4M is well inside the tolerance this module already
        documents. If a later round makes the lexicon aware of the enclosing
        parenthetical (96.8% of "lit"'s real uses sit inside one, against 0% of
        the two rows here), these become ordinary passing cases; until then the
        behavior is asserted so it cannot change silently.
        """
        album = ('Neil Young in Berlin is a live video by Neil Young, directed '
                 'by Michael Lindsay-Hogg, and recorded in October 1982 during '
                 'the European Tour for his album Trans. It includes the song '
                 '"After Berlin" written especially for that concert and only '
                 'performed once.')
        self.assertEqual(chat.first_sentence(album), album,
                         'the album title "Trans." reads as the gloss '
                         'abbreviation, so no boundary is left to cut at')

        burner = ("Auto reignition is a process used in gas burners to control "
                  "ignition devices based on whether a burner flame is lit. This "
                  "information can be used to stop an ignition device from "
                  "sparking, which is no longer necessary after the flame is "
                  "lit. It can also be used to start the sparking device again "
                  "if the flame goes out while the burner is still supplying "
                  "gas, for example, from a gust of wind or vibration.")
        self.assertEqual(chat.first_sentence(burner), burner,
                         "both real boundaries in this row end on \"lit.\", so "
                         "the whole paragraph is kept")

    def test_word_floor_is_positional_which_is_the_known_residual(self):
        """The measured limitation, asserted rather than left implicit. The
        same unlisted abbreviation is caught when it lands early and is NOT
        caught when it lands past the floor, because the floor counts words
        rather than judging the token. Closing this would need a rule that
        tells an unlisted abbreviation from a rare proper noun, and none
        exists: over 174,042 real rows the two are indistinguishable by length,
        capitalisation and period-attachment alike, so any rule strict enough
        to catch "Gebr." also destroys the 1,625 correct cuts ending "...in
        Iran.". Residual after the lexicon audit: 3 fragments in 114,789 cuts
        (0.003%). If a later round closes this, update the second half.
        """
        early = "Gebr. Heinemann is a German retailer. It was founded in 1879."
        self.assertEqual(chat.SENTENCE_BOUNDARY_RE.split(early, maxsplit=1)[0],
                         "Gebr.")
        self.assertEqual(chat.first_sentence(early), early,
                         "an early mis-cut falls back to the full paragraph")

        late = ("The company was founded as Gebr. Heinemann in Hamburg in 1879. "
                "It operates duty free shops.")
        self.assertEqual(chat.first_sentence(late),
                         "The company was founded as Gebr.",
                         "known residual: past the floor the same abbreviation "
                         "yields a fragment")

    def test_short_real_sentence_is_still_cut(self):
        """MIN_SENTENCE_WORDS is 4, not 5, precisely so the "<Surname> is a
        surname." disambiguation rows (149 of 20,049 cuts in the sample)
        still get cut.
        """
        self.install("Ryzkhov is a surname. Notable people with the surname "
                     "include several athletes.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Ryzkhov?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Ryzkhov is a surname.\n[source:", out)
        self.assertNotIn("Notable people", out)

    def test_answer_with_no_terminator_at_all_keeps_full_text(self):
        """Cut off mid word by the max_new_tokens budget, so there is no
        ".", "!" or "?" anywhere to cut at: the answer must come back
        whole, not empty.
        """
        no_terminator = "The mitochondria is the powerhouse of the cell and produces most of the"
        self.install(no_terminator)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the mitochondria?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(f"{no_terminator}\n[source:", out)

    def test_empty_answer_does_not_crash(self):
        self.install("")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is Jackson County, Illinois?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[source:", out)

    def test_greeting_reply_is_not_cut_to_its_first_sentence(self):
        """The no match shape's greeting replies are trained as real two
        sentence answers ("Hello! What would you like to know?"), not
        excess to trim, unlike the RAG shape. Only the sourced branch gets
        the first sentence cut.
        """
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("hi")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Hello! What would you like to know?", out)


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


class TestNoResultsFoundIsNotFedToTheModel(_ChatFlowTest):
    """bad-cop's real finding: a search that genuinely finds nothing used to
    come back from web_search() as a placeholder string that did not start
    with "[web search error:", so run_web_search()'s diagnostic check missed
    it entirely and it fell straight through to _context_prompt()/
    _generate() as if it were a real retrieved passage — a placeholder
    handed to the model and shown back tagged "AI generated from the search
    result", worse than the honest diagnostic that case deserves.
    web_search() reports it as None now and run_web_search() returns
    (text, is_diagnostic), so no caller has to recognize a magic string.
    """

    def test_run_web_search_flags_no_results_as_diagnostic(self):
        with mock.patch.object(chat, "web_search", return_value=None):
            text, is_diagnostic = chat.run_web_search("a query DuckDuckGo has nothing for")
        self.assertEqual(text, chat.NO_RESULTS_MESSAGE)
        self.assertTrue(is_diagnostic)

    def test_no_results_answer_is_never_generated_from(self):
        """End to end through the real web_search(), with only ddgs itself
        mocked, because the bug lived in the seam between the two modules
        and a mock of web_search() is a mock of exactly the thing that was
        wrong. An empty DuckDuckGo response must leave _generate() called
        once (the initial refusal), never a second time to "paraphrase"
        nothing as if it were real search content.
        """
        self.install(REFUSAL)
        real_generate = chat._generate
        with mock.patch.object(web_search, "DDGS") as m_ddgs, \
                mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "_generate", side_effect=real_generate) as m_generate:
            m_ddgs.return_value.text.return_value = []
            out = chat.answer_question("What is the boiling point of mercury?")
        self.assertEqual(m_generate.call_count, 1,
                         "the model must only generate once here (the initial "
                         "refusal), never a second time from an empty search")
        self.assertIn(chat.NO_RESULTS_MESSAGE, out)
        self.assertNotIn("AI generated from the search result", out,
                         "nothing was actually generated, the tag must not claim it was")


class TestDiagnosticSignalCannotComeFromPageContent(_ChatFlowTest):
    """bad-cop's collision finding, and the seam it lives on. Whether a live
    search produced real content used to be decided by sniffing a bracketed
    prefix off the returned string, but half of that string is the page body
    DuckDuckGo hands back — unconstrained text this project does not
    control. bad-cop reproduced the collision: a genuine hit whose body
    opened with "[web search error:" was classified as a diagnostic and
    printed to the user with the model never seeing it (one generate() call
    for a turn that owed two), breaking the "always AI generated on the
    spot" contract with exactly the text an attacker chooses.

    web_search() signals its three outcomes out of band now (raise / None /
    a result string), so the tests below pin both halves of that seam: the
    contract web_search() really keeps, and chat.py branching only on it.
    """

    def test_a_result_body_that_looks_like_a_diagnostic_is_still_generated_from(self):
        """Driven from a real ddgs response whose body is the colliding
        text, through the real web_search(), for the same reason as above:
        the seam is the thing under test.
        """
        colliding = "[web search error: how to read this log line] (source: example.com)"
        self.install(REFUSAL)
        real_generate = chat._generate
        with mock.patch.object(web_search, "DDGS") as m_ddgs, \
                mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "_generate", side_effect=real_generate) as m_generate:
            m_ddgs.return_value.text.return_value = [{
                "body": "[web search error: how to read this log line]",
                "href": "example.com",
            }]
            result = chat.answer_turn("What is the capital of Freedonia?")
        self.assertEqual(m_generate.call_count, 2,
                         "real search content must be generated from, whatever "
                         "its body text happens to start with")
        self.assertEqual(result["context"], colliding)
        self.assertNotIn(colliding, result["text"],
                         "the raw DuckDuckGo text must never be shown untouched")

    def test_web_search_returns_none_when_duckduckgo_finds_nothing(self):
        """The web_search.py half of the seam, against the real function:
        an empty result list is None, never a string a caller could mistake
        for content.
        """
        with mock.patch.object(web_search, "DDGS") as m_ddgs:
            m_ddgs.return_value.text.return_value = []
            self.assertIsNone(web_search.web_search("a query with no hits"))

    def test_web_search_raises_websearcherror_when_the_lookup_fails(self):
        """The other half: every failure ddgs can raise, documented or not,
        arrives as one type, with the original exception kept as its cause.
        """
        with mock.patch.object(web_search, "DDGS") as m_ddgs:
            m_ddgs.return_value.text.side_effect = ValueError("ddgs internal crash")
            with self.assertRaises(web_search.WebSearchError) as caught:
                web_search.web_search("q")
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertIn("ddgs internal crash", str(caught.exception))

    def test_a_failed_lookup_is_reported_at_error_level(self):
        """The failure is shown to the user as text; the traceback belongs
        in the log too, not swallowed at the boundary.
        """
        with mock.patch.object(chat, "web_search",
                               side_effect=RuntimeError("ddgs internal crash")), \
                mock.patch.object(chat.logger, "error") as m_error:
            text, is_diagnostic = chat.run_web_search("q")
        self.assertTrue(is_diagnostic)
        self.assertIn("[web search error: ddgs internal crash]", text)
        self.assertEqual(m_error.call_count, 1)
        self.assertTrue(m_error.call_args.kwargs.get("exc_info"))


class TestRefusedVectorContextStaysRecoverable(_ChatFlowTest):
    """bad-cop's other real finding: once "context" was repointed at the
    live search text that actually produced the visible answer, the
    original vector passage the model saw and refused had nowhere left to
    go, contradicting _log_turn()'s own "complete record" docstring.
    "refused_context"/"refused_source_title" keep it recoverable.
    """

    def test_refused_context_and_title_are_populated(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of France?")
        self.assertEqual(result["refused_context"], SOURCE["content"])
        self.assertEqual(result["refused_source_title"], "Paris")
        # And "context" still reflects what produced the shown answer, not
        # the refused passage, per the earlier fix in the same area.
        self.assertEqual(result["context"], SEARCH_RESULT)

    def test_refused_context_is_none_when_there_was_no_real_source(self):
        """The other half: a true local miss (nothing retrieved at all)
        has no passage to have refused, so both fields stay None rather
        than inventing a false "refused" record.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of Freedonia?")
        self.assertIsNone(result["refused_context"])
        self.assertIsNone(result["refused_source_title"])

    def test_refused_context_is_logged(self):
        path, _handler = self.use_real_temp_log()
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            chat.answer_turn("What is the capital of France?")
        with open(path, "r", encoding="utf-8") as f:
            logged = f.read()
        self.assertIn("REFUSED LOCAL CONTEXT (source: Paris):", logged)
        self.assertIn(SOURCE["content"], logged)


class TestModeToggle(_ChatFlowTest):
    """The three values gui.py's Chat tab Source toggle passes as mode=.

    "web" is the one with real behavior behind it and the one these tests
    are mostly about. It turns off the vector store, NOT the model: an
    earlier version of it was a bare `return run_web_search(question)`,
    which returned a spam calculator site for "what is 17 times 23" instead
    of dispatching the calculator tool, and an unrelated top ranked article
    for "hi". So every branch answer_question() has must still be reachable
    under it, with retrieve() alone removed from the loop.
    """

    def test_web_mode_never_calls_retrieve(self):
        """The core claim of the redesign, asserted on the mock's call count
        rather than on the output: retrieve() must not run at all, not merely
        have its result ignored.
        """
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]) as m_retrieve, \
                mock.patch.object(chat, "web_search") as m_web:
            chat.answer_question("What is the capital of France?", mode="web")
        self.assertEqual(m_retrieve.call_count, 0,
                         "retrieve() ran under mode='web', vector store not actually skipped")
        self.assertEqual(m_web.call_count, 0)

    def test_auto_and_vector_modes_still_call_retrieve(self):
        """The other half of the same wiring: only "web" turns retrieval off."""
        for mode in ("auto", "vector"):
            with self.subTest(mode=mode):
                self.install("Paris.")
                with mock.patch.object(chat, "retrieve", return_value=[SOURCE]) as m_retrieve, \
                        mock.patch.object(chat, "web_search"):
                    chat.answer_question("What is the capital of France?", mode=mode)
                self.assertEqual(m_retrieve.call_count, 1)

    def test_web_mode_still_dispatches_a_tool_call(self):
        """The reported failure that motivated the redesign, as a test: a
        math question under this mode must reach the calculator, not a search
        engine's idea of a calculator.
        """
        self.install("CALL: calculator(17*23)")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is 17 times 23?", mode="web")
        self.assertEqual(m_retrieve.call_count, 0)
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("[tool: calculator]", out)
        self.assertIn("391", out)

    def test_web_mode_greeting_does_not_search(self):
        """The second reported failure: "hi" is small talk the model has a
        trained reply for, and searching for it returns noise.
        """
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("hi", mode="web")
        self.assertEqual(m_retrieve.call_count, 0)
        self.assertEqual(m_web.call_count, 0)
        self.assertIn("Hello!", out)

    def test_web_mode_refusal_falls_back_to_search(self):
        """The one path that does reach DuckDuckGo here, and it is the same
        refusal triggered fallback "auto" already uses.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT) as m_web:
            out = chat.answer_question("how far is the sun?", mode="web")
        self.assertEqual(m_retrieve.call_count, 0,
                         "retrieve() must never run under mode='web'")
        self.assertEqual(m_web.call_count, 1)
        self.assertNotIn(SEARCH_RESULT, out,
                         "the raw DuckDuckGo text must never be shown untouched")
        self.assertIn("AI generated from the search result", out)

    def test_web_mode_refusal_tag_names_the_disabled_vector_store(self):
        """source is always None under this mode, so the tag must not borrow
        either of the other two reasons: "model refused the local match" is
        unreachable, and "no local match" would claim the vector store was
        searched and came back empty when it was never consulted.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            out = chat.answer_question("how far is the sun?", mode="web")
        self.assertIn("[vector store off, fell back to live search, AI generated from the search result]", out)
        self.assertNotIn("model refused the local match", out)
        self.assertNotIn("no local match", out)

    def test_web_mode_never_tags_an_answer_as_sourced(self):
        """The `elif source is not None` branch (first_sentence() plus the
        [source: ...] tag) is structurally unreachable here. Fed a real two
        sentence Wikipedia shaped answer, the output must fall to the
        unsourced branch and keep both sentences, since only the sourced
        branch cuts.
        """
        two_sentences = ("As the capital of France, Paris is the seat of France's "
                         "national government. For the executive, the two chief "
                         "officers each have their own official residences.")
        self.install(two_sentences)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the capital of France?", mode="web")
        self.assertEqual(m_web.call_count, 0)
        self.assertNotIn("[source:", out)
        # "vector store off", not "no confident source found": retrieve()
        # never ran under mode="web" (use_retrieval=False), so nothing was
        # actually searched, same distinction the refusal-fallback branch
        # already draws for the same mode.
        self.assertIn("[vector store off", out)
        self.assertIn("For the executive", out)

    def test_explicit_auto_matches_the_default_mode(self):
        """mode="auto" is the pre-toggle behavior and the default, so the two
        call shapes must agree exactly, on the case this file's own branch
        order regression was about.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            out_default = chat.answer_question("What is the capital of France?")
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            out_explicit = chat.answer_question("What is the capital of France?", mode="auto")
        self.assertEqual(out_default, out_explicit)

    def test_vector_mode_refusal_does_not_search(self):
        """What "vector" is for: the live search is out of the loop, so the
        model's refusal is the answer rather than a cue to go fetch one
        elsewhere. Asserted for both retrieval outcomes, because neither of
        them may reach DuckDuckGo, only the tag below differs between them.
        """
        for docs in ([SOURCE], []):
            with self.subTest(retrieved=len(docs)):
                self.install(REFUSAL)
                with mock.patch.object(chat, "retrieve", return_value=docs), \
                        mock.patch.object(chat, "web_search") as m_web:
                    out = chat.answer_question("What is the capital of France?", mode="vector")
                self.assertEqual(m_web.call_count, 0)
                self.assertIn("answered from the model's own training only", out)

    def test_vector_mode_refusal_without_a_source_says_none_was_found(self):
        """Nothing cleared the rerank threshold, so "no confident source
        found" is literally what happened and stays the wording here.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the capital of Freedonia?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(
            "[no confident source found — answered from the model's own training only]", out)

    def test_vector_mode_refusal_with_a_source_says_the_model_refused_it(self):
        """The real, reproduced failure this pair of tests grew for.
        retrieve() returned a passage scoring 9.00, above rerank_threshold
        7.0, and build_prompt() really did put it in the Context: block the
        model was given this turn; the model refused it anyway. Tagging that
        "no confident source found" asserts the opposite of what the code
        just did, and described one event two contradictory ways depending
        only on the toggle: "auto" calls the identical input "model refused
        the local match", asserted below so the two cannot drift apart again.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            out = chat.answer_question("What is the capital of France?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertIn(
            "[model refused the local match — answered from the model's own training only]", out)
        self.assertNotIn("no confident source found", out)

        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            auto_out = chat.answer_question("What is the capital of France?", mode="auto")
        self.assertIn("model refused the local match", auto_out)

    def test_web_prefix_overrides_every_mode(self):
        """The "web: <question>" prefix is checked before any mode logic, so
        it stays a real override under all three, including "vector", which
        otherwise never searches at all: same model-generated-from-search-
        result answer either way, never the raw DuckDuckGo text.
        """
        for mode in ("auto", "vector", "web"):
            with self.subTest(mode=mode):
                self.install("unused")
                with mock.patch.object(chat, "retrieve", return_value=[]) as m_retrieve, \
                        mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT) as m_web:
                    out = chat.answer_question("web: capital of france", mode=mode)
                self.assertFalse(m_retrieve.called)
                m_web.assert_called_once_with("capital of france")
                self.assertNotEqual(out, SEARCH_RESULT,
                                    "the raw DuckDuckGo text must never be shown untouched")
                self.assertIn("unused", out)
                self.assertIn("[source: live search result (DuckDuckGo), AI generated]", out)


class _RealTokenizerTest(_ChatFlowTest):
    """The real gpt2 encode/decode pair load() installs, instead of
    _ChatFlowTest's length//4 stand in. What the tests below are about is
    truncate_to_tokens()'s own budget arithmetic, and a stand in that counts
    characters cannot catch a real token boundary error in it.
    """

    def setUp(self):
        enc = tiktoken.get_encoding("gpt2")
        self.addCleanup(setattr, chat, "_encode", chat._encode)
        self.addCleanup(setattr, chat, "_decode", chat._decode)
        chat._encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        chat._decode = lambda ids: enc.decode([t for t in ids if t < enc.n_vocab])


class TestBuildPromptContext(_RealTokenizerTest):
    """build_prompt()'s third return value: the exact text substituted into
    the Context: block, which is what the GUI's RAG context viewer shows.

    It is deliberately not source["content"]. The retrieved passage is
    truncated to whatever the template budget leaves, so showing the raw
    content would claim the model saw text it never got.
    """

    def test_context_is_the_truncated_text_not_the_raw_content(self):
        """The property the whole three-tuple return exists for. A passage
        far longer than block_size must come back cut, and the cut text must
        be exactly what is in the prompt.
        """
        long_content = "Paris is the capital of France. " * 2000
        source = dict(SOURCE, content=long_content)
        with mock.patch.object(chat, "retrieve", return_value=[source]):
            prefix, best, context_text = chat.build_prompt("What is the capital of France?")
        self.assertIs(best, source)
        self.assertNotEqual(context_text, long_content,
                            "context must be the truncated text, not the raw passage")
        self.assertIn(context_text, prefix)
        self.assertLess(len(chat._encode(context_text)), len(chat._encode(long_content)))

    def test_short_content_is_used_untruncated(self):
        """The other direction: nothing is cut when it fits, so the context
        is the passage verbatim and the prefix is the plain RAG template.
        """
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]):
            prefix, best, context_text = chat.build_prompt("What is the capital of France?")
        self.assertEqual(context_text, SOURCE["content"])
        self.assertEqual(
            prefix,
            f"Context: {context_text}\nQuestion: What is the capital of France?\nAnswer:")

    def test_no_match_returns_no_source_and_no_context(self):
        """The Context: (none) shape has no context to show, and says so with
        None rather than with an empty string.
        """
        with mock.patch.object(chat, "retrieve", return_value=[]):
            prefix, best, context_text = chat.build_prompt("What is the capital of Freedonia?")
        self.assertEqual(
            prefix,
            "Context: (none)\nQuestion: What is the capital of Freedonia?\nAnswer:")
        self.assertIsNone(best)
        self.assertIsNone(context_text)

    def test_a_question_filling_the_block_leaves_an_empty_context(self):
        """The budget <= 0 boundary, reachable from ordinary (if extreme)
        typed input rather than from a crash: a question near block_size
        tokens leaves max(budget, 0) == 0, so the Context: block really is
        empty and "context" is "" — falsy but NOT None, since a real source
        was used. gui.py's `if context:` renders no link for it, which is the
        wanted behavior (there is nothing to show), and asserting "" here is
        what keeps that distinguishable from the None branches below.
        """
        huge_question = "what is the capital of France " * 400
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]):
            _, best, context_text = chat.build_prompt(huge_question)
        self.assertIsNotNone(best, "a real source was retrieved and used")
        self.assertEqual(context_text, "")

        self.install_model()
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn(huge_question)
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source_title"], "Paris",
                         "the title still travels with the context, empty or not")


class TestParaphraseTitleGating(_RealTokenizerTest):
    """The "never copy and paste the RAG result" hard rule's inference-time
    half: build_sft_dataset.py's rag_examples() now trains the RAG shape
    exclusively on titles with a real Simple Wikipedia paraphrase (no copy
    fallback), so a retrieval hit for any OTHER title asks the model to
    paraphrase content it was never taught to. build_prompt() must treat
    that hit exactly like a real miss.

    _get_paraphrase_titles() is mocked directly rather than relying on the
    real corpus/simple_wikipedia_summaries/summaries.tsv on disk (SOURCE's
    "Paris" title only happens to pass the other tests in this file because
    Simple Wikipedia really does have a "Paris" article, an implicit and
    fragile dependency this class does not want).
    """

    def test_title_with_a_trained_paraphrase_is_sourced(self):
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "_get_paraphrase_titles", return_value={"Paris"}):
            prefix, best, context_text = chat.build_prompt("What is the capital of France?")
        self.assertIs(best, SOURCE)
        self.assertEqual(context_text, SOURCE["content"])
        self.assertIn("Context: Paris is the capital of France.", prefix)

    def test_title_without_a_trained_paraphrase_falls_through_to_no_match(self):
        """The actual new behavior: a real retrieval hit, just not for a
        title rag_examples() ever trained a paraphrase for, must come back
        identical to the true `if not docs:` miss above, not a Context:
        block built from content the model was never taught to paraphrase.
        """
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "_get_paraphrase_titles", return_value={"Some Other Title"}):
            prefix, best, context_text = chat.build_prompt("What is the capital of France?")
        self.assertEqual(
            prefix,
            "Context: (none)\nQuestion: What is the capital of France?\nAnswer:")
        self.assertIsNone(best)
        self.assertIsNone(context_text)

    def test_gated_title_answer_turn_behaves_like_a_real_no_match(self):
        """End to end through answer_turn(): a gated-out retrieval must take
        the same no-source path a true miss takes, refusal included, never a
        [source: Paris ...] tag claiming the model was trained to answer
        about a title it was gated out of. It still ends up falling back to
        a live search and generating a real answer from that (the "always
        AI generated on the spot" rule applies here too), so it is tagged
        "live search result", never the gated vector title.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "_get_paraphrase_titles", return_value=set()), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of France?")
        self.assertNotEqual(result["source_title"], "Paris")
        self.assertNotIn("[source: Paris", result["text"])


class TestAnswerTurnContext(_ChatFlowTest):
    """answer_turn()'s "context" and "source_title", the pair the GUI's
    expandable "[+ show RAG context]" viewer is built on. They travel
    together: a title without the text it labels is as useless as text with
    no attribution, so every case below asserts both.
    """

    def test_sourced_branch_carries_the_context(self):
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn("What is the capital of France?")
        self.assertEqual(result["context"], SOURCE["content"])
        self.assertEqual(result["source_title"], "Paris")

    def test_refusal_fallback_context_reflects_what_produced_the_shown_answer(self):
        """The branch that is easiest to get wrong: the visible answer now
        comes from the model generating on the spot from the live search
        result (per the explicit "talk to ai using rag should ALWAYS get a
        response made with AI generated on the spot" rule), not from the
        vector passage it originally refused, so "context" must reflect
        THAT — the text that actually produced what's on screen — not the
        refused, discarded first attempt. Same reasoning as the
        "AI generated from the search result" tag TestRefusalFallsBackToSearch
        asserts on the same branch.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of France?")
        self.assertEqual(result["context"], SEARCH_RESULT)
        self.assertEqual(result["source_title"], "live search result")

    def test_vector_mode_refusal_also_carries_the_context_it_refused(self):
        """The same event as the test above with only the toggle changed, so
        it must carry the same pair. mode="vector" turns off the live search,
        not the vector store (gui.py's label for it is "Vector only"), and the
        sourced branch under this very mode already returns both fields, so
        dropping them here alone would leave one mode's context viewer working
        on one branch and dead on the other.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is the capital of France?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertEqual(result["context"], SOURCE["content"])
        self.assertEqual(result["source_title"], "Paris")

    def test_vector_mode_refusal_renders_the_gui_context_link(self):
        """The GUI visible consequence of the test above, not just the dict
        value: gui.py's ChatTab.render_answer() draws "[+ show RAG context]"
        only under `if context:`, so a None there is the difference between
        the user being able to see the passage the model refused and having
        no way to. Simulates that exact condition against real answer_turn()
        output; no Tk needed, the branch under test is plain data logic.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn("What is the capital of France?", mode="vector")
        self.assertTrue(
            bool(result.get("context")),
            "the model was shown a real retrieved passage this turn (score "
            "9.0, above rerank_threshold=7.0) and refused it, but the user "
            "gets no link to see what it saw")

    def test_vector_mode_sourced_branch_carries_the_context(self):
        """The sourced branch under mode="vector", the combination the two
        tests above are measured against: it does not branch on mode at all,
        so both fields travel exactly as they do under "auto". Asserted rather
        than read off the code, since it is what makes withholding them on the
        refusal branch of the same mode inconsistent.
        """
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is the capital of France?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertEqual(result["context"], SOURCE["content"])
        self.assertEqual(result["source_title"], "Paris")

    def test_vector_mode_refusal_without_a_source_carries_no_context(self):
        """The other half of property one: retrieval really came back empty,
        so there is nothing the model saw and nothing to invent, even though
        the branch right above now returns a real pair.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is the capital of Freedonia?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["context"])
        self.assertIsNone(result["source_title"])

    def test_sourced_branch_is_unreachable_under_web_mode(self):
        """mode="web" forces use_retrieval=False, so `elif source is not
        None` can never fire under it: the turn falls to the unsourced branch
        with both fields None, never to the sourced branch's pair. Verified
        directly rather than assumed from reading the code, since the two
        outcomes are indistinguishable from the text alone.
        """
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]) as m_retrieve, \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn("What is the capital of France?", mode="web")
        self.assertEqual(m_retrieve.call_count, 0)
        self.assertIsNone(result["context"])
        self.assertIsNone(result["source_title"])

    def test_refusal_without_a_source_now_carries_the_live_search_context(self):
        """Nothing was retrieved locally, but the fallback still runs the
        model on the live search result (per the "always AI generated on
        the spot" rule), so unlike the vector-store branches, this one now
        DOES have a real context to show: what the search fallback actually
        fed the model, not nothing and not the raw DuckDuckGo text either.
        """
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of Freedonia?")
        self.assertEqual(result["context"], SEARCH_RESULT)
        self.assertEqual(result["source_title"], "live search result")

    def test_tool_call_and_unsourced_greeting_carry_no_context(self):
        """The two branches that never touch a Context: block at all (a
        dispatched tool call, and a trained no-match reply that the model
        did not tag as a refusal, both real answers on their own with
        nothing to fall back from): both fields stay None.
        """
        self.install("CALL: calculator(2+2)")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            tool_call = chat.answer_turn("What is 2 plus 2?")

        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search"):
            unsourced = chat.answer_turn("hi")

        for name, result in (("tool call", tool_call), ("unsourced", unsourced)):
            with self.subTest(branch=name):
                self.assertIsNone(result["context"])
                self.assertIsNone(result["source_title"])

    def test_web_prefix_carries_the_live_search_context(self):
        """The explicit "web:" prefix path, the other place a live search
        result is now handed to the model instead of shown untouched: it
        must carry the same real context/title pair the auto-fallback
        branch above does, not None the way it did back when this path
        printed DuckDuckGo's text directly.
        """
        self.install("unused")
        with mock.patch.object(chat, "retrieve"), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            web_prefix = chat.answer_turn("web: capital of france")
        self.assertEqual(web_prefix["context"], SEARCH_RESULT)
        self.assertEqual(web_prefix["source_title"], "live search result")


class TestAnswerQuestionMatchesAnswerTurn(_ChatFlowTest):
    """answer_question() is now `return answer_turn(question, mode)["text"]`,
    so its output must stay byte identical to answer_turn()'s on every
    branch — the CLI loop and the 44 tests above are all still reading it.
    One case per branch, since a divergence could be introduced on any one
    of them alone.
    """

    def assert_wrapper_matches(self, question, mode="auto"):
        turn = chat.answer_turn(question, mode=mode)
        self.assertEqual(turn["text"], chat.answer_question(question, mode=mode))

    def test_tool_call_branch(self):
        self.install("CALL: calculator(2+2)")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            self.assert_wrapper_matches("What is 2 plus 2?")

    def test_sourced_branch(self):
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            self.assert_wrapper_matches("What is the capital of France?")

    def test_refusal_with_source_branch(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            self.assert_wrapper_matches("What is the capital of France?")

    def test_refusal_without_source_branch(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            self.assert_wrapper_matches("What is the capital of Freedonia?")

    def test_unsourced_branch(self):
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search"):
            self.assert_wrapper_matches("hi")

    def test_web_prefix_branch(self):
        self.install("unused")
        with mock.patch.object(chat, "retrieve"), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            self.assert_wrapper_matches("web: capital of france")


class TestLogWriteFailureIsReported(_ChatFlowTest):
    """_log_turn's two promises when the log file itself breaks: the caller
    still gets its answer, AND the failure is reported at ERROR level rather
    than vanishing.

    The second half was a real, measured failure. Mocking chat.logger.info to
    raise (the first two tests) skips the whole dispatch path and cannot see
    it; against a real handler whose stream.write raised OSError, logger.error
    was called 0 times, because logging.Handler.emit() catches the write
    failure itself and hands it to Handler.handleError(), which prints
    "--- Logging error ---" plus a traceback to stderr and returns normally.
    chat.ChatTurnFileHandler overrides handleError to report the failure
    itself instead, so every test below drives the real handler.

    test_real_handler_write_failure_reaches_logger_error's called_once is
    load bearing beyond "it was reported": the ERROR line it asks for is
    routed straight back into the same handler whose write is failing, so
    without ChatTurnFileHandler's own recursion guard that count would climb
    until the stack ran out.
    """

    def test_logger_info_raising_does_not_break_answer_turn(self):
        self.install("Paris.")
        with mock.patch.object(chat.logger, "info", side_effect=OSError("disk full")), \
                mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn("What is the capital of France?")
        self.assertIn("[source: Paris", result["text"])

    def test_logger_error_is_actually_called_on_failure(self):
        self.install("Paris.")
        with mock.patch.object(chat.logger, "info", side_effect=OSError("disk full")), \
                mock.patch.object(chat.logger, "error") as m_error, \
                mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"):
            chat.answer_turn("What is the capital of France?")
        self.assertTrue(m_error.called,
                        "logger.error was never called on a logging failure")

    def test_real_handler_write_failure_reaches_logger_error(self):
        """The one the mocks above cannot see, and the reason
        ChatTurnFileHandler exists. The failure is injected at the real
        stream.write and travels the genuine Logger.info -> Handler.handle ->
        Handler.emit path.
        """
        _, handler = self.use_real_temp_log()
        self.install("Paris.")
        stderr = io.StringIO()
        with mock.patch.object(handler.stream, "write",
                               side_effect=OSError("simulated disk full")), \
                mock.patch.object(chat.logger, "error", wraps=chat.logger.error) as m_error, \
                mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"), \
                redirect_stderr(stderr):
            result = chat.answer_turn("What is the capital of France?")

        self.assertIn("[source: Paris", result["text"],
                      "the answer must survive a log write failure")
        m_error.assert_called_once()
        self.assertIn("failed to log chat turn", m_error.call_args.args[0])
        self.assertIn("simulated disk full", str(m_error.call_args.args[1]))

    def test_a_write_failure_leaves_one_named_line_on_stderr_not_a_raw_dump(self):
        """Where a maintainer actually looks during a disk full incident.
        chat_turns.log is by definition unwritable in this scenario, so the
        ERROR line cannot land there and stderr is the only sink left — but it
        must carry chat.py's own named message, not stdlib logging's
        "--- Logging error ---" traceback, which says nothing about which
        subsystem lost what.
        """
        _, handler = self.use_real_temp_log()
        self.install("Paris.")
        stderr = io.StringIO()
        with mock.patch.object(handler.stream, "write",
                               side_effect=OSError("simulated disk full")), \
                mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search"), \
                redirect_stderr(stderr):
            chat.answer_turn("What is the capital of France?")
        written = stderr.getvalue()
        self.assertIn("chat: failed to log chat turn", written)
        self.assertIn("simulated disk full", written)
        self.assertNotIn("--- Logging error ---", written)

    def test_a_failure_with_a_writable_file_is_reported_in_the_log_itself(self):
        """The other failure shape, and the one the docstring's "reported
        through this same logger" is literally true of: the file is fine and
        building the block is what failed (here, a result dict missing the
        "text" key). The ERROR line then lands in chat_turns.log, where
        someone reviewing the conversation will actually find it.
        """
        path, handler = self.use_real_temp_log()
        chat._log_turn("What is the capital of France?", "auto",
                       {"context": None, "source_title": None, "web_result": None})
        handler.flush()
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("failed to log chat turn", content)
        self.assertIn("KeyError", content, "exc_info=True must carry the traceback")


class _RaisingHandler(logging.Handler):
    """A handler that raises out of handle(), the thing a handler is not
    supposed to do. Only StreamHandler.emit() contains its own failures, so
    a bare Handler subclass whose emit raises reaches Logger.callHandlers()
    uncontained, which is how a third party handler on an ancestor logger
    would misbehave.
    """

    def emit(self, record):
        raise RuntimeError("ancestor handler is broken")


class TestOneBrokenHandlerDoesNotBreakTheOthers(_ChatFlowTest):
    """The other half of that handler's contract: reporting its own write
    failure must not cost every OTHER handler the record.

    gui.py line 44 calls logging.basicConfig(), installing a StreamHandler on
    the ROOT logger in the same process chat.logger lives in as a child with
    propagate left at its default True (chat.py never sets propagate=False),
    which is exactly the "gui.py may already have called it first when both
    modules load in the same process" case chat.py's own comment names as the
    reason its dedicated handler exists. Logger.callHandlers() walks "this
    logger's handlers, then every ancestor logger's handlers" in ONE loop
    (`while c: for hdlr in c.handlers: ... hdlr.handle(record)`, CPython
    Lib/logging/__init__.py) with no try/except of its own, and
    Handler.handle() only wraps emit() in try/finally to release its lock, so
    a handler that raises aborts that walk where it stands.

    Measured against the first version of ChatTurnFileHandler, whose
    handleError was a bare `raise`: logger.info() raised OSError straight out
    to the caller and the root handler's buffer was empty — one handler's
    broken file silently cost every other handler the record. Nothing else in
    this file exercises propagation or a second handler at all, which is how
    that shipped.
    """

    def setUp(self):
        root = logging.getLogger()
        self.addCleanup(setattr, root, "handlers", root.handlers)
        self.addCleanup(setattr, root, "level", root.level)
        self.addCleanup(setattr, chat.logger, "propagate", chat.logger.propagate)
        self.root_buf = io.StringIO()
        root_handler = logging.StreamHandler(self.root_buf)
        root_handler.setFormatter(logging.Formatter("ROOT: %(message)s"))
        root.handlers = [root_handler]
        root.setLevel(logging.INFO)
        chat.logger.propagate = True

    def test_a_plain_filehandler_write_failure_reaches_the_root_handler(self):
        """Baseline: stdlib's own default handleError has no such side
        effect, so what the next test asks for is a property of the override,
        not an unavoidable consequence of a failed write.
        """
        tmpdir = tempfile.mkdtemp(prefix="chat_log_test_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        plain = logging.FileHandler(os.path.join(tmpdir, "plain.log"),
                                    encoding="utf-8")
        plain.setFormatter(logging.Formatter("%(message)s"))
        self.addCleanup(plain.close)
        saved = chat.logger.handlers
        chat.logger.handlers = [plain]
        self.addCleanup(setattr, chat.logger, "handlers", saved)

        stderr = io.StringIO()
        with mock.patch.object(plain.stream, "write",
                               side_effect=OSError("simulated disk full")), \
                redirect_stderr(stderr):
            chat.logger.info("a real turn record")

        self.assertIn("ROOT: a real turn record", self.root_buf.getvalue())

    def test_a_real_write_failure_reaches_the_root_handler_and_is_reported(self):
        """Both properties at once, on the real shipped handler: gui.py's
        root handler still gets the turn it would have gotten if
        chat_turns.log had been writable, AND the failure is still named
        rather than lost.
        """
        _, handler = self.use_real_temp_log()
        stderr = io.StringIO()
        with mock.patch.object(handler.stream, "write",
                               side_effect=OSError("simulated disk full")), \
                redirect_stderr(stderr):
            chat.logger.info("a real turn record")  # must not raise

        propagated = self.root_buf.getvalue()
        self.assertIn("ROOT: a real turn record", propagated,
                      "a failure of chat.py's own file handler must not cost "
                      "gui.py's root handler the record")
        self.assertIn("failed to log chat turn", propagated,
                      "and the failure itself must still be reported, not "
                      "quietly swallowed to keep the walk going")
        self.assertIn("chat: failed to log chat turn", stderr.getvalue(),
                      "the ERROR line has nowhere to land but stderr when "
                      "the log file is what broke")
        self.assertNotIn("--- Logging error ---", stderr.getvalue())

    def test_a_raising_ancestor_handler_still_cannot_kill_the_turn(self):
        """The exposure the fix above creates, in the other direction: now
        that a record really does reach ancestor handlers, one of them can
        raise back through Logger.callHandlers() into _log_turn. That must
        still cost a log line and not the answer, which is what
        _report_log_failure's own stderr fallback is left for.
        """
        path, _ = self.use_real_temp_log()
        logging.getLogger().handlers = [_RaisingHandler()]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            chat._log_turn("What is the capital of France?", "auto",
                           {"text": "Paris.", "context": None,
                            "source_title": None, "web_result": None})

        self.assertIn("chat: failed to log chat turn", stderr.getvalue())
        self.assertIn("ancestor handler is broken", stderr.getvalue())
        with open(path, encoding="utf-8") as f:
            written = f.read()
        self.assertIn("ANSWER:", written,
                      "and chat.py's own handler, which runs before the "
                      "broken one, still wrote the turn")


class TestLogSetupFailureDoesNotBreakImport(_ChatFlowTest):
    """Module import must survive an unusable logs/ directory.

    os.makedirs(_LOG_DIR, exist_ok=True) and the FileHandler construction run
    at import time, outside every try/except _log_turn has, so before the
    guard a logs/ path occupied by a plain file raised FileExistsError
    straight out of `import chat` — taking gui.py's Chat tab and the CLI loop
    down over a log line. Driven against a real copy of the real chat.py,
    loaded through the real import machinery from a temp tree where that path
    is blocked, since the bug lives in module level code that a normal import
    of the already imported module can never re run.
    """

    def load_chat_copy_with_blocked_log_dir(self):
        tmp = tempfile.mkdtemp(prefix="chat_log_setup_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pkg = os.path.join(tmp, "toolstore")
        os.makedirs(pkg)
        shutil.copy(chat.__file__, os.path.join(pkg, "chat.py"))
        blocked = os.path.join(tmp, "logs")
        with open(blocked, "w") as f:
            f.write("not a directory")
        with self.assertRaises(OSError, msg="sanity: the blocked path must really break makedirs"):
            os.makedirs(blocked, exist_ok=True)

        # The module level block only runs when the "chat" logger has no
        # handlers yet, and logging.getLogger("chat") is process wide, so the
        # already configured handlers have to step aside for this import.
        saved = chat.logger.handlers
        chat.logger.handlers = []
        self.addCleanup(setattr, chat.logger, "handlers", saved)
        # chat.py inserts ../model on sys.path at import; from the temp tree
        # that is a path that does not exist.
        self.addCleanup(setattr, sys, "path", list(sys.path))

        spec = importlib.util.spec_from_file_location(
            "chat_blocked_log_dir", os.path.join(pkg, "chat.py"))
        module = importlib.util.module_from_spec(spec)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            spec.loader.exec_module(module)
        self.assertEqual(os.path.normpath(module._LOG_DIR), os.path.normpath(blocked),
                         "sanity: the copy must target the blocked path")
        return module, stderr.getvalue()

    def test_import_survives_a_logs_path_that_is_not_a_directory(self):
        module, stderr = self.load_chat_copy_with_blocked_log_dir()
        self.assertTrue(hasattr(module, "answer_turn"),
                        "the module must import fully, not partially")
        self.assertIn("chat turn logging to", stderr)
        self.assertIn("disabled", stderr)
        self.assertIn("FileExistsError", stderr, "the real cause must be named")

    def test_the_fallback_handler_is_stderr_at_warning_not_a_file(self):
        """Degrading to stderr keeps failures visible with no file to write
        them to; the WARNING floor keeps every turn's full block (RAG context
        included) out of the console it just fell back to.
        """
        self.load_chat_copy_with_blocked_log_dir()
        self.assertEqual(len(chat.logger.handlers), 1)
        handler = chat.logger.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertNotIsInstance(handler, logging.FileHandler)
        self.assertEqual(handler.level, logging.WARNING)

    def test_a_turn_still_logs_without_crashing_after_the_fallback(self):
        """The point of degrading rather than raising: turns keep working,
        they just have nowhere to be written.
        """
        module, _ = self.load_chat_copy_with_blocked_log_dir()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            module._log_turn("What is the capital of France?", "auto",
                             {"text": "Paris.", "context": None,
                              "source_title": None, "web_result": None})
        self.assertEqual(stderr.getvalue(), "",
                         "an ordinary turn must not spam the console it fell back to")


class TestLoggedTurnContent(_ChatFlowTest):
    """What actually lands in the file, read back off disk through a real
    handler rather than asserted on a mock's call args.
    """

    def test_large_context_round_trips_into_the_log_file(self):
        path, handler = self.use_real_temp_log()
        self.install("Paris.")
        source = dict(SOURCE, content="Paris is the capital of France. " * 2000)
        with mock.patch.object(chat, "retrieve", return_value=[source]), \
                mock.patch.object(chat, "web_search"):
            result = chat.answer_turn("What is the capital of France?")
        handler.flush()
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("RAG CONTEXT", content)
        self.assertIn(result["context"], content,
                      "the log must carry the context the model really saw, whole")
        self.assertIn("ANSWER:", content)

    def test_unicode_context_and_answer_round_trip_in_the_log_file(self):
        """The handler is opened encoding="utf-8" for the same reason
        sys.stdout is reconfigured at the top of chat.py: the corpus is real
        Wikipedia text and this machine's console default is cp1252.
        """
        path, handler = self.use_real_temp_log()
        self.install("東京は日本の首都です。 — emoji test \U0001F600")
        source = dict(SOURCE, content="東京は日本の首都です。北京語ではない。 éèê")
        with mock.patch.object(chat, "retrieve", return_value=[source]), \
                mock.patch.object(chat, "web_search"):
            chat.answer_turn("What is the capital of Japan?")
        handler.flush()
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("東京は日本の首都です", content)
        self.assertIn("\U0001F600", content)

    def test_concurrent_turns_produce_uninterleaved_blocks(self):
        """gui.py calls answer_turn() from a worker thread, so two turns can
        be mid _log_turn at once (two windows, or a second Enter before the
        busy flag catches up). Each block must land whole: the record is one
        emit() call under the handler's own lock, and this asserts that on the
        real file rather than taking the docs' word for it.
        """
        path, handler = self.use_real_temp_log()
        threads_count = 60
        errors = []

        def worker(i):
            try:
                chat.logger.info("\n".join([
                    chat._LOG_SEPARATOR,
                    f"THREADSTART<{i}>",
                    "body " * 50,
                    f"THREADEND<{i}>",
                ]) + "\n")
            except Exception as e:  # noqa: BLE001 - reported below, not swallowed
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        handler.flush()

        self.assertEqual(errors, [])
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for i in range(threads_count):
            start = content.find(f"THREADSTART<{i}>")
            end = content.find(f"THREADEND<{i}>")
            self.assertNotEqual(start, -1, f"thread {i} start marker missing")
            self.assertNotEqual(end, -1, f"thread {i} end marker missing")
            between = content[start:end]
            for j in range(threads_count):
                # Exact bracketed tokens: "<1>" is a substring of "<10>".
                if j != i:
                    self.assertNotIn(f"<{j}>", between,
                                     f"thread {j} interleaved inside thread {i}'s record")


class TestWebResultField(_ChatFlowTest):
    """answer_turn()'s "web_result", the third field the log block reads.
    It is populated on exactly two branches, the "web:" prefix and the
    refusal fallback under mode "auto"/"web", and None everywhere else,
    including mode="vector"'s refusal, which never searches. Asserted per
    branch because the field had zero coverage when it was added: the log's
    LIVE SEARCH RESULT section is the only thing that reads it.
    """

    def test_web_prefix_branch_sets_web_result(self):
        self.install("unused")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("web: capital of france")
        self.assertFalse(m_retrieve.called)
        self.assertEqual(result["web_result"], SEARCH_RESULT)

    def test_auto_refusal_fallback_sets_web_result(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("What is the capital of Freedonia?")
        self.assertEqual(result["web_result"], SEARCH_RESULT)

    def test_web_mode_refusal_fallback_sets_web_result(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            result = chat.answer_turn("how far is the sun?", mode="web")
        self.assertFalse(m_retrieve.called)
        self.assertEqual(result["web_result"], SEARCH_RESULT)

    def test_vector_mode_refusal_web_result_is_none(self):
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is the capital of France?", mode="vector")
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["web_result"])

    def test_sourced_branch_web_result_is_none(self):
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is the capital of France?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["web_result"])

    def test_tool_call_web_result_is_none(self):
        self.install("CALL: calculator(2+2)")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("What is 2 plus 2?")
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["web_result"])

    def test_unsourced_greeting_web_result_is_none(self):
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("hi")
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["web_result"])

    def test_web_mode_unsourced_web_result_is_none(self):
        self.install("Hello! What would you like to know?")
        with mock.patch.object(chat, "retrieve") as m_retrieve, \
                mock.patch.object(chat, "web_search") as m_web:
            result = chat.answer_turn("hi", mode="web")
        self.assertEqual(m_retrieve.call_count, 0)
        self.assertEqual(m_web.call_count, 0)
        self.assertIsNone(result["web_result"])

    def test_live_search_section_is_logged_only_when_web_result_is_present(self):
        """The end to end version: _log_turn's conditional section really
        does appear (and not appear) in the file, on the two branches either
        side of it.
        """
        path, handler = self.use_real_temp_log()
        self.install(REFUSAL)
        with mock.patch.object(chat, "retrieve", return_value=[]), \
                mock.patch.object(chat, "web_search", return_value=SEARCH_RESULT):
            chat.answer_turn("What is the capital of Freedonia?")
        handler.flush()
        with open(path, encoding="utf-8") as f:
            searched = f.read()
        self.assertIn("LIVE SEARCH RESULT (DuckDuckGo):", searched)
        self.assertIn(SEARCH_RESULT, searched)

        path, handler = self.use_real_temp_log()
        self.install("Paris.")
        with mock.patch.object(chat, "retrieve", return_value=[SOURCE]), \
                mock.patch.object(chat, "web_search") as m_web:
            chat.answer_turn("What is the capital of France?")
        handler.flush()
        with open(path, encoding="utf-8") as f:
            sourced = f.read()
        self.assertNotIn("LIVE SEARCH RESULT", sourced)
        self.assertEqual(m_web.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
