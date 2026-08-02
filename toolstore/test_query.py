"""Regression tests for query.py's disambiguation filter.

This predicate decides which retrieval candidates are thrown away, so a
wrong answer in either direction is silent: too loose and a real article
becomes permanently unreachable, too tight and a contentless stub gets fed
to the model as Context. It has been wrong in both directions already, so
every case below is a real row copied verbatim out of the real 6.4 million
row vectorstore.db, quoted here as a fixture so the test runs without the
12GB database attached.

Run with: python -m unittest discover -s toolstore
"""
import unittest

from query import _is_disambiguation

# (rowid, title, first paragraph as stored), real disambiguation stubs.
DISAMBIGUATION_ROWS = [
    (1374, "BLM", "BLM most commonly refers to:"),
    (2096, "Buffalo", "Buffalo most commonly refers to:"),
    (2244, "Capital", "Capital most commonly refers to:"),
    (4243, "English", "English usually refers to:"),
    (688, "Auriga", "Auriga or AURIGA can refer to:"),
    (24, "Alien", "Alien primarily refers to:  Alien (law), a person in a country who is "
                  "not a national of that country  Enemy alien, a national of a country at war"),
    (4182, "Discharge", "Discharge may refer to"),
    (22235, "Hackney", "Hackney may refer to"),
    (23091, "Treaty of Paris",
     "Treaty of Paris may refer to one of many treaties signed in Paris, France:"),
]

# Real substantive articles. Every one of these was silently deleted from
# retrieval by an earlier version of the filter.
ARTICLE_ROWS = [
    (1194, "Ackermann function",
     "In computability theory, the Ackermann function, named after Wilhelm Ackermann, is one "
     "of the simplest and earliest-discovered examples of a total computable function that is "
     "not primitive recursive. The term may refer to any of a number of variants."),
    (3162, "Cranberry",
     "Cranberries are a group of evergreen dwarf shrubs or trailing vines in the subgenus "
     "Oxycoccus of the genus Vaccinium. In Britain, cranberry may refer to the native species "
     "Vaccinium oxycoccos."),
    (264, "Adelaide",
     "Adelaide ( ) is the capital city of South Australia, the state's largest city and the "
     'fifth-most populous city of Australia. "Adelaide" may refer to the Adelaide city centre.'),
    (18735, "Inflation",
     "In economics, inflation refers to a general progressive increase in prices of goods and "
     "services in an economy."),
    (23979, "Software architecture",
     "Software architecture refers to the fundamental structures of a software system and the "
     "discipline of creating such structures and systems."),
    (2520, "Cryptanalysis",
     'Cryptanalysis (from the Greek kryptos, "hidden", and analyein, "to analyze") refers to '
     "the process of analyzing information systems in order to understand hidden aspects."),
    (9259, "Middle East",
     "The Middle East (, ISO 233: ) is a geopolitical term that commonly refers to the region "
     "spanning the Levant, Arabian Peninsula, Anatolia and Egypt."),
    (24109, "Furniture",
     "Furniture refers to movable objects intended to support various human activities such as "
     "seating (e.g., Stools, Chairs, and sofas), eating (tables), storing items."),
]


class TestIsDisambiguation(unittest.TestCase):
    def test_flags_real_disambiguation_stubs(self):
        for rowid, title, content in DISAMBIGUATION_ROWS:
            with self.subTest(rowid=rowid, title=title):
                self.assertTrue(
                    _is_disambiguation(title, content),
                    f"rowid {rowid} {title!r} is a disambiguation stub but was kept as a "
                    f"retrieval candidate, so the model can be handed it as Context",
                )

    def test_keeps_real_articles_that_use_refers_to_as_prose(self):
        for rowid, title, content in ARTICLE_ROWS:
            with self.subTest(rowid=rowid, title=title):
                self.assertFalse(
                    _is_disambiguation(title, content),
                    f"rowid {rowid} {title!r} is a real article but was filtered out, which "
                    f"removes it from every future query with no way to retrieve it again",
                )

    def test_flags_disambiguation_title_suffix(self):
        self.assertTrue(_is_disambiguation("Mercury (disambiguation)", "Mercury is a planet."))

    def test_handles_missing_title_and_content(self):
        self.assertFalse(_is_disambiguation(None, None))
        self.assertFalse(_is_disambiguation("", ""))


if __name__ == "__main__":
    unittest.main()
