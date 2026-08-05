"""Regression test for the real gap bad-cop found in _input_fingerprint():
load_simple_wikipedia_paraphrases() calls first_sentence(text), imported
from toolstore/sentence_boundary.py (a separate file, reached via
sys.path.insert, not this script's own source), to build the exact text
every RAG training row is trained on. The first version of
_input_fingerprint() hashed only this script's own __file__ plus the three
corpus files, so a real edit to sentence_boundary.py (its own file history
documents five real rounds of lexicon/regex changes) changed what the built
dataset actually contains while being completely invisible to
dataset_needs_rebuild() — orchestrate.py would have kept training on / serving
the OLD .npy files forever after such a change.

TestRebuildOnMissingTrainedTitles covers the other half of the same fix, and
it exists because of a measured hole rather than for symmetry: deleting
dataset_needs_rebuild()'s `if not os.path.exists(TRAINED_RAG_TITLES_PATH)`
check entirely left all three tests above passing, 0 failures. That check is
what keeps a pre-fix dataset (the four .npy files and a matching manifest on
disk, but no trained_rag_titles.json, exactly the state this repo was in
before the fix) from being reported current forever, which would leave
chat.py's gate permanently empty — every retrieval hit silently downgraded
to the no-match branch, with nothing ever triggering the rebuild that would
write the file.

Run with:
  "E:/Dev/.venv/Scripts/python.exe" -m unittest model.sft.test_dataset_staleness -v
or, from model/sft/:
  "E:/Dev/.venv/Scripts/python.exe" -m unittest test_dataset_staleness -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import build_sft_dataset


class TestFingerprintCoversSentenceBoundaryModule(unittest.TestCase):
    def test_fingerprint_has_a_signature_for_sentence_boundary_py(self):
        fp = build_sft_dataset._input_fingerprint()
        self.assertIn("sentence_boundary_sig", fp)
        self.assertIsNotNone(
            fp["sentence_boundary_sig"],
            "toolstore/sentence_boundary.py must really exist for this to be meaningful",
        )

    def test_the_signature_is_really_computed_from_sentence_boundary_py_on_disk(self):
        """Not a placeholder value: it must be _file_signature() of the
        real sentence_boundary.py path, the same function every other
        entry in the fingerprint dict uses, so a real edit to that file
        (size or mtime changing) changes this value the same way editing
        WIKI_TSV changes "wiki_tsv".
        """
        sentence_boundary_path = os.path.join(
            build_sft_dataset.TOOLSTORE, "sentence_boundary.py")
        expected = build_sft_dataset._file_signature(sentence_boundary_path)
        fp = build_sft_dataset._input_fingerprint()
        self.assertEqual(fp["sentence_boundary_sig"], expected)

    def test_touching_the_real_file_changes_the_fingerprint(self):
        """The actual real-world property: an edit to sentence_boundary.py
        (mtime changes, as every real edit does) must move
        _input_fingerprint()'s output, so dataset_needs_rebuild() reports
        stale. Touches the real file's mtime and restores it in a finally
        block, never its content.
        """
        sentence_boundary_path = os.path.join(
            build_sft_dataset.TOOLSTORE, "sentence_boundary.py")
        original_stat = os.stat(sentence_boundary_path)
        fp_before = build_sft_dataset._input_fingerprint()
        try:
            new_mtime = original_stat.st_mtime + 1
            os.utime(sentence_boundary_path, (new_mtime, new_mtime))
            fp_after = build_sft_dataset._input_fingerprint()
        finally:
            os.utime(sentence_boundary_path, (original_stat.st_atime, original_stat.st_mtime))
        self.assertNotEqual(
            fp_before["sentence_boundary_sig"], fp_after["sentence_boundary_sig"],
            "editing sentence_boundary.py must move the fingerprint so "
            "dataset_needs_rebuild() reports the dataset is stale",
        )


class TestRebuildOnMissingTrainedTitles(unittest.TestCase):
    """Every path here is redirected into a temp directory, so this never
    reads or writes the real .npy files, manifest or title list: the
    property under test is dataset_needs_rebuild()'s own logic, not the
    state of this machine's built dataset.
    """

    def setUp(self):
        tmpdir = tempfile.mkdtemp(prefix="dataset_staleness_test_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        self.titles_path = os.path.join(tmpdir, "trained_rag_titles.json")
        manifest_path = os.path.join(tmpdir, "dataset_manifest.json")
        for name in ("sft_train_x.npy", "sft_train_y.npy", "sft_val_x.npy", "sft_val_y.npy"):
            open(os.path.join(tmpdir, name), "wb").close()
        # A manifest that matches the current real inputs exactly, so a
        # fingerprint mismatch can never be what makes the assertions below
        # pass: the title list is the only thing left that differs.
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(build_sft_dataset._input_fingerprint(), f)
        for attr, value in (("OUT_DIR", tmpdir),
                            ("MANIFEST_PATH", manifest_path),
                            ("TRAINED_RAG_TITLES_PATH", self.titles_path)):
            patcher = mock.patch.object(build_sft_dataset, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_trained_titles_forces_a_rebuild(self):
        self.assertFalse(os.path.exists(self.titles_path))
        self.assertTrue(
            build_sft_dataset.dataset_needs_rebuild(),
            "a dataset with no trained_rag_titles.json cannot be current: "
            "chat.py's gate would read no titles at all and downgrade every "
            "retrieval hit to no match",
        )

    def test_present_trained_titles_with_a_matching_manifest_is_current(self):
        """The other side of the same check, so it is a real condition and
        not just an unconditional True: everything present and matching
        must NOT rebuild, otherwise orchestrate.py would loop rebuilding
        forever.
        """
        with open(self.titles_path, "w", encoding="utf-8") as f:
            json.dump(["Paris"], f)
        self.assertFalse(build_sft_dataset.dataset_needs_rebuild())


if __name__ == "__main__":
    unittest.main()
