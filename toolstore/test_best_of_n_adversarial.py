"""Adversarial tests for _best_of_n_generate()'s selection contract: that
it returns the candidate the reranker actually scored highest, and that
n=1 skips the reranker call entirely.

Neither was covered by test_chat.py, which reaches this function only
through _run_turn() and only with a _FakeReranker returning uniform 0.0
scores, so a correct argmax is indistinguishable there from "always return
candidates[0]". Proven, not assumed: three hand-built mutants of
_best_of_n_generate (argmax result discarded and candidates[0] returned;
max() swapped for min(); `n < 1` instead of `n <= 1`) all pass
test_chat.py's 104 tests unchanged and all three fail here.

Not covered here on purpose: query.py's real CrossEncoder. Loading it is
an ~80MB model load, the same network/disk dependency test_chat.py's
setUpModule() documents avoiding. It was checked out of band instead, and
predict() returns a float32 numpy array (not a list) that max(..., key=)
orders correctly, scoring an on-topic candidate 9.36 against -7.29/-11.21/
-8.11 for off-topic and empty ones.

Run with: "E:/Dev/.venv/Scripts/python.exe" -m unittest discover -s "E:/Dev/toolstore" -p "test_best_of_n_adversarial.py" -v
"""
import unittest
from unittest import mock

import chat


class _ScoringReranker:
    """A reranker that assigns a distinct, known score per candidate text,
    so the winner can be checked by content, not by index coincidence.
    """
    def __init__(self, score_map):
        self.score_map = score_map
        self.calls = []

    def predict(self, pairs, show_progress_bar=False):
        self.calls.append(pairs)
        return [self.score_map[content] for (_q, content) in pairs]


class TestBestOfNHonoursN(unittest.TestCase):
    """How many candidates get generated, and whether the reranker is paid
    for at all: n=1 must not call get_reranker() or reranker.predict() at
    all, not just short-circuit after generating, and the default n must be
    the N_BEST_OF config constant rather than a number of its own.
    """

    def test_n_equals_1_never_touches_the_reranker(self):
        calls = {"count": 0}

        def fake_generate(prefix):
            calls["count"] += 1
            return f"candidate-{calls['count']}"

        with mock.patch.object(chat, "_generate", side_effect=fake_generate), \
                mock.patch.object(chat, "get_reranker",
                                   side_effect=AssertionError(
                                       "get_reranker() must not be called when n=1")) as m_reranker:
            result = chat._best_of_n_generate("Context: x\nQuestion: y\nAnswer:", "y", n=1)

        self.assertEqual(result, "candidate-1")
        self.assertEqual(calls["count"], 1)
        m_reranker.assert_not_called()

    def test_the_default_n_is_the_N_BEST_OF_config_constant(self):
        """Every call site in chat.py omits n, so the n=1 skip above and the
        4-candidate cost are both only ever reached through the default. A
        default decoupled from N_BEST_OF (hardcoded, or left at a number of
        its own) would silently ignore the config constant.

        Note this has to be checked by calling with n omitted, not by
        monkeypatching chat.N_BEST_OF: a default argument is bound once at
        def time, so mock.patch.object(chat, "N_BEST_OF", ...) cannot reach
        it and a test written that way asserts nothing about the default.
        """
        generated = []

        def fake_generate(prefix):
            generated.append(f"candidate-{len(generated) + 1}")
            return generated[-1]

        # Ascending scores, so the winner is the last candidate generated:
        # a default that generated too few would score a different winner,
        # not just a different count.
        reranker = _ScoringReranker({f"candidate-{i}": float(i)
                                     for i in range(1, chat.N_BEST_OF + 1)})
        with mock.patch.object(chat, "_generate", side_effect=fake_generate), \
                mock.patch.object(chat, "get_reranker", return_value=reranker):
            result = chat._best_of_n_generate("prefix", "question")

        self.assertEqual(len(generated), chat.N_BEST_OF)
        self.assertEqual(result, f"candidate-{chat.N_BEST_OF}")


class TestBestOfNActuallySelectsTheHighestScoringCandidate(unittest.TestCase):
    """The functional contract test_chat.py's _FakeReranker cannot exercise:
    it always returns uniform 0.0 scores (by design, to keep the existing
    44+ SOURCE-based tests decode-content-agnostic), so no existing test can
    tell a correct argmax from a bug that e.g. picks the lowest score, the
    first score, or the last generated candidate regardless of score.
    """

    def test_the_highest_scored_candidate_wins_not_the_first_or_last(self):
        candidates = iter(["low-quality answer", "BEST ANSWER", "medium answer", "worst answer"])

        def fake_generate(prefix):
            return next(candidates)

        reranker = _ScoringReranker({
            "low-quality answer": 0.1,
            "BEST ANSWER": 9.9,
            "medium answer": 3.0,
            "worst answer": -5.0,
        })

        with mock.patch.object(chat, "_generate", side_effect=fake_generate), \
                mock.patch.object(chat, "get_reranker", return_value=reranker):
            result = chat._best_of_n_generate("prefix", "the question", n=4)

        self.assertEqual(result, "BEST ANSWER")
        # Pairs must be (question, candidate), the same query_text-first shape
        # query.py's own search() uses for (query_text, content).
        self.assertEqual(reranker.calls[0][0], ("the question", "low-quality answer"))
        self.assertEqual(reranker.calls[0][1], ("the question", "BEST ANSWER"))

    def test_the_lowest_scored_candidate_never_wins(self):
        """The mutant this guards against: swapping max() for min(), or
        reversing the score comparison. Run the same scenario and confirm
        the worst-scored candidate is specifically excluded.
        """
        candidates = iter(["A", "B", "C"])

        def fake_generate(prefix):
            return next(candidates)

        reranker = _ScoringReranker({"A": 5.0, "B": 1.0, "C": 3.0})
        with mock.patch.object(chat, "_generate", side_effect=fake_generate), \
                mock.patch.object(chat, "get_reranker", return_value=reranker):
            result = chat._best_of_n_generate("prefix", "q", n=3)
        self.assertEqual(result, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
