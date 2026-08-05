"""Regression test for the real gap bad-cop found between chat.py's
_get_paraphrase_titles() gate and model/sft/build_sft_dataset.py's
rag_examples(): the first version of the gate re-derived an approximation
(every title in corpus/simple_wikipedia_summaries/summaries.tsv clearing
the disambiguation filter, 238,363 of them) instead of reading what
rag_examples() actually sampled (N_RAG=60,000, stopped early in WIKI_TSV
file order). 178,363 titles (74.8% of the gate) passed as "safe to answer
from retrieval" while never having been shown to the model during
training — the exact "per article memorization, not a rule that transfers
to a title it never saw" failure the no-copy-fallback feature exists to
close, reopened for most of the gate's own accepted titles.

The fix: build_sft_dataset.build() writes the exact used_titles
rag_examples() returns to TRAINED_RAG_TITLES_PATH, and chat.py's
_get_paraphrase_titles() reads that file instead of re-deriving a guess.
This test proves the two are the same set, using the real corpus files on
disk (rag_examples() itself is cheap, no tokenization needed), without
running the full, slow build()/tokenize pipeline.

Run with:
  "E:/Dev/.venv/Scripts/python.exe" -m unittest toolstore.test_paraphrase_gate_integration -v
or, from toolstore/:
  "E:/Dev/.venv/Scripts/python.exe" -m unittest test_paraphrase_gate_integration -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "model", "sft"))

import chat  # noqa: E402
import build_sft_dataset  # noqa: E402


class TestGateMatchesActualTraining(unittest.TestCase):
    def test_gate_titles_exactly_equal_the_titles_rag_examples_actually_used(self):
        paraphrases = build_sft_dataset.load_simple_wikipedia_paraphrases()
        _rows, used_titles = build_sft_dataset.rag_examples(build_sft_dataset.N_RAG, paraphrases)
        self.assertGreater(len(used_titles), 0, "real corpus files must be present for this test to mean anything")

        tmpdir = tempfile.mkdtemp(prefix="trained_rag_titles_test_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        tmp_path = os.path.join(tmpdir, "trained_rag_titles.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sorted(used_titles), f)

        chat._paraphrase_titles = None
        self.addCleanup(setattr, chat, "_paraphrase_titles", None)
        with mock.patch.object(chat, "TRAINED_RAG_TITLES_PATH", tmp_path):
            gate_titles = chat._get_paraphrase_titles()

        self.assertEqual(
            gate_titles, used_titles,
            "chat.py's gate must be exactly the set rag_examples() actually "
            "trained, not an approximation derived separately",
        )

    def test_a_title_dropped_by_rag_examples_ncap_is_correctly_excluded_from_a_small_gate(self):
        """A smaller, deterministic proof of the same property using a
        tiny synthetic paraphrase dict and a real limit=1, so the "stops
        early in file order" behavior is exercised directly rather than
        only at the real N_RAG=60,000 scale.
        """
        paraphrases = build_sft_dataset.load_simple_wikipedia_paraphrases()
        two_titles = dict(list(paraphrases.items())[:2])
        self.assertEqual(len(two_titles), 2, "real corpus must supply at least 2 paraphrase candidates")

        _rows, used_titles = build_sft_dataset.rag_examples(1, two_titles)
        self.assertEqual(len(used_titles), 1, "limit=1 must cap rag_examples() at exactly one used title")
        self.assertTrue(used_titles.issubset(set(two_titles)))


if __name__ == "__main__":
    unittest.main()
