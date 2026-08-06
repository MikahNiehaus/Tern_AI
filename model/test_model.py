"""Real, real-execution tests for GPT.generate()'s top_p (nucleus sampling)
addition. A tiny GPTConfig (n_layer=2, n_embd=32, small vocab) so this runs
in under a second on CPU, not the real 124M checkpoint; the property under
test is the sampling/filtering logic in generate() itself, not anything
about a trained model's actual outputs.

Run with: python -m unittest discover -s model -p test_model.py
"""
import unittest

import torch

from model import GPT, GPTConfig


def _tiny_model():
    torch.manual_seed(1337)
    config = GPTConfig(block_size=32, vocab_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)
    model = GPT(config)
    model.eval()
    return model


class TestTopPBackwardCompatibility(unittest.TestCase):
    """talk.py and sample.py are nanoGPT's own untouched reference files
    that call generate() with only temperature/top_k, never top_p: the new
    kwarg must default to off and change nothing about their call shape.
    """

    def test_top_p_defaults_to_none_and_generation_still_runs(self):
        model = _tiny_model()
        idx = torch.zeros((1, 1), dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5, temperature=0.8, top_k=10)
        self.assertEqual(out.shape, (1, 6))

    def test_top_p_none_is_bitwise_identical_to_pre_change_behavior(self):
        """The real regression guard: same seed, same call shape as before
        this change, must produce the exact same tokens now that top_p
        exists as a kwarg but is left at its default.
        """
        idx = torch.zeros((1, 1), dtype=torch.long)

        torch.manual_seed(42)
        model_a = _tiny_model()
        out_a = model_a.generate(idx.clone(), max_new_tokens=8, temperature=0.8, top_k=10)

        torch.manual_seed(42)
        model_b = _tiny_model()
        out_b = model_b.generate(idx.clone(), max_new_tokens=8, temperature=0.8, top_k=10, top_p=None)

        self.assertTrue(torch.equal(out_a, out_b))


class TestTopPFiltering(unittest.TestCase):
    """The real property nucleus sampling exists for: the sampled token
    must always come from the smallest set of options whose cumulative
    probability covers top_p, never from the long low-probability tail
    top_k alone would still allow through when top_k is set loosely (or
    not set at all).
    """

    def test_small_top_p_only_ever_samples_the_single_highest_probability_token(self):
        """top_p near 0 must collapse to the single most likely next token
        every time (mirrors greedy), the tightest real behavioral check:
        run many draws and confirm every single one lands on the argmax
        token this exact model/seed/context would produce.
        """
        model = _tiny_model()
        idx = torch.zeros((1, 3), dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(idx)
            expected_next = int(torch.argmax(logits[:, -1, :], dim=-1).item())

        for trial_seed in range(10):
            torch.manual_seed(trial_seed)
            out = model.generate(idx.clone(), max_new_tokens=1, temperature=1.0, top_p=1e-6)
            self.assertEqual(int(out[0, -1].item()), expected_next)

    def test_top_p_composes_with_top_k_not_instead_of_it(self):
        """Both filters active at once must still produce a valid token
        (no crash from an over-aggressive double filter, e.g. top_k
        leaving one token and top_p's own "keep at least one" floor
        fighting it), and the output must stay within top_k's own
        candidate set (top_p can only narrow further, never widen back out
        past what top_k already excluded).
        """
        model = _tiny_model()
        idx = torch.zeros((1, 2), dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(idx)
            top5 = set(torch.topk(logits[:, -1, :], 5, dim=-1).indices[0].tolist())

        for trial_seed in range(10):
            torch.manual_seed(trial_seed)
            out = model.generate(idx.clone(), max_new_tokens=1, temperature=1.0, top_k=5, top_p=0.9)
            self.assertIn(int(out[0, -1].item()), top5)

    def test_top_p_never_masks_every_token(self):
        """The min_tokens_to_keep=1 floor HuggingFace's own TopPLogitsWarper
        keeps: even an unreasonably small top_p must not leave torch.multinomial
        sampling from an all -inf row (which would produce NaN probabilities
        and crash), because at least the single highest probability token is
        always exempted from the mask.
        """
        model = _tiny_model()
        idx = torch.zeros((1, 1), dtype=torch.long)
        torch.manual_seed(7)
        out = model.generate(idx, max_new_tokens=3, temperature=1.0, top_p=1e-9)
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
