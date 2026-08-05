"""First sentence extraction, real regex plus a real abbreviation lexicon,
not a full NLP sentence tokenizer. Split out of chat.py into its own module
so model/sft/build_sft_dataset.py (a data prep script with no reason to
import torch, tiktoken, faiss, or sentence-transformers, chat.py's own
transitive dependencies) can reuse this same, already tested logic to build
concise training targets instead of duplicating it or hand rolling a weaker
version. chat.py imports first_sentence() from here now instead of defining
it, so there is exactly one real implementation.
"""
import regex  # NOT stdlib re: variable width lookbehind, see SENTENCE_BOUNDARY_RE

# Verbatim copy of pysbd 0.3.4's English abbreviation list
# (pysbd/lang/common/standard.py, class Standard.Abbreviation.ABBREVIATIONS,
# MIT licensed), itself a direct port of the Ruby gem pragmatic_segmenter's
# own list. Swiped rather than hand written on purpose: which tokens are
# abbreviations is a lexicon, not a pattern, which is exactly why every real
# segmenter carries one (NLTK's punkt *learns* one; pragmatic_segmenter and
# pysbd hardcode this one) and why two previous attempts here to infer it from
# the token's shape both failed. Matched case insensitively, so the sentence
# initial "Prof." and the mid sentence "prof." are one entry, not two.
# Kept verbatim and separate from the corpus additions below so this block
# stays a straight copy of upstream, diffable against a future pysbd release.
PYSBD_ABBREVIATIONS = [
    'adj', 'adm', 'adv', 'al', 'ala', 'alta', 'apr', 'arc', 'ariz', 'ark',
    'art', 'assn', 'asst', 'attys', 'aug', 'ave', 'bart', 'bld', 'bldg',
    'blvd', 'brig', 'bros', 'btw', 'cal', 'calif', 'capt', 'cl', 'cmdr', 'co',
    'col', 'colo', 'comdr', 'con', 'conn', 'corp', 'cpl', 'cres', 'ct',
    'd.phil', 'dak', 'dec', 'del', 'dept', 'det', 'dist', 'dr', 'dr.phil',
    'dr.philos', 'drs', 'e.g', 'ens', 'esp', 'esq', 'etc', 'exp', 'expy',
    'ext', 'feb', 'fed', 'fla', 'ft', 'fwy', 'fy', 'ga', 'gen', 'gov', 'hon',
    'hosp', 'hr', 'hway', 'hwy', 'i.e', 'ia', 'id', 'ida', 'ill', 'inc',
    'ind', 'ing', 'insp', 'is', 'jan', 'jr', 'jul', 'jun', 'kan', 'kans',
    'ken', 'ky', 'la', 'lt', 'ltd', 'maj', 'man', 'mar', 'mass', 'may', 'md',
    'me', 'med', 'messrs', 'mex', 'mfg', 'mich', 'min', 'minn', 'miss',
    'mlle', 'mm', 'mme', 'mo', 'mont', 'mr', 'mrs', 'ms', 'msgr', 'mssrs',
    'mt', 'mtn', 'neb', 'nebr', 'nev', 'no', 'nos', 'nov', 'nr', 'oct', 'ok',
    'okla', 'ont', 'op', 'ord', 'ore', 'p', 'pa', 'pd', 'pde', 'penn',
    'penna', 'pfc', 'ph', 'ph.d', 'pl', 'plz', 'pp', 'prof', 'pvt', 'que',
    'rd', 'rs', 'ref', 'rep', 'reps', 'res', 'rev', 'rt', 'sask', 'sec',
    'sen', 'sens', 'sep', 'sept', 'sfc', 'sgt', 'sr', 'st', 'supt', 'surg',
    'tce', 'tenn', 'tex', 'univ', 'usafa', 'u.s', 'ut', 'va', 'v', 'ver',
    'viz', 'vs', 'vt', 'wash', 'wis', 'wisc', 'wy', 'wyo', 'yuk', 'fig',
]

# Round five audited both lists in the direction round four never checked: for
# every entry, how often does that token, period attached and followed by a
# capital, end a REAL sentence instead of abbreviating? Method, over all
# 6,439,528 rows of corpus/wikipedia_summaries/summaries.tsv: take every
# boundary the lexicon blocks, run first_sentence() twice on that row (with the
# entry and without it), and read the rows whose answer changes.
#
# What that turned up first is a structural mistake, not a bad entry. pysbd does
# not use the list above the way this file was using it. Its own
# AbbreviationReplacer (pysbd/abbreviation_replacer.py, scan_for_replacements)
# decides:
#
#     upper = str(char).isupper()
#     if not upper or am.strip().lower() in prepositive:
#
# so a general ABBREVIATIONS entry only swallows its period when the next
# character is NOT a capital. Only PREPOSITIVE_ABBREVIATIONS ('dr', 'mr', 'st',
# 'prof', 'v', the titles that always precede a capitalised name) swallow it
# before a capital, and NUMBER_ABBREVIATIONS ('no', 'p', 'pp') only before a
# digit. SENTENCE_BOUNDARY_RE fires ONLY before a capital, so putting the whole
# general list in its lookbehind gives every entry prepositive treatment, which
# is precisely what lets an ordinary English word block a real sentence end.
#
# The entries below are the ones that measurably cost more than they buy, as
# rescues/swallows: rows cut correctly only WITH the entry, against rows cut
# correctly only WITHOUT it. Both buckets were sampled and read, not just
# counted.
#
#   art 2/1520   inc 31/1784  ltd 9/828   man 0/798   etc 4/549   may 0/431
#   mass 4/202   me 5/169     corp 4/113  pa 4/100    ave 3/89    arc 0/73
#   is 1/64      ore 0/42     ct 10/37    ph 9/37     la 2/37     mm 0/35
#   mo 0/34      ok 0/32      p 0/32      min 0/26    penn 0/26   ill 4/26
#   rd 2/24      id 6/19      nos 0/10
#
# The swallowed side reads as ordinary prose: "...in the Celtic style of Insular
# art. These knots are...", "...released in November 1979 by Atari, Inc. The
# player controls...", "...distributed by Toho Co., Ltd. The film stars...",
# "...or the first Monday of May. It is an ancient festival...".
#
# Entries that look equally word-like were measured the same way and KEPT,
# because their blocks really are abbreviations and dropping them truncates a
# name: 'bros' 80/1803 ("Warner Bros. Pictures"), 'jr' 109/1333 ("James Tiptree
# Jr. Award"), 'co' 149/1111 ("Sligo Town, Co. Sligo"), 'v' 2385/1747 ("MGM
# Studios, Inc. v. Grokster"), 'ft' 13/72 ("MC 900 Ft. Jesus"), 'sr' 54/284
# ("John Raymond Dyer Sr. OAM"), 'no' 83/121 ("Lightvessel No. XVII").
SENTENCE_ENDING_ABBREVIATIONS = [
    'arc', 'art', 'ave', 'corp', 'ct', 'etc', 'id', 'ill', 'inc', 'is', 'la',
    'ltd', 'man', 'mass', 'may', 'me', 'min', 'mm', 'mo', 'nos', 'ok', 'ore',
    'p', 'pa', 'penn', 'ph', 'rd',
]

# What pysbd's general English list does not carry, found by auditing the real
# corpus instead of by guessing one entry at a time (round three added the
# lexicon; round four asked what else it is missing and how the answer was
# arrived at).
#
# Method, over every 37th row of corpus/wikipedia_summaries/summaries.tsv
# (174,042 rows, 7.3M tokens): run the real first_sentence() over each row,
# take the token immediately before the terminator of every cut it ACCEPTED,
# and read those tokens ranked by how often they appear period-attached versus
# bare. That is outcome driven, it asks which tokens actually produced a bad
# cut, rather than asking which tokens look like abbreviations.
#
# Two things that experiment settled, both of which argue against continuing to
# extend the list by hand:
#
# 1. A learned list is not better here. NLTK punkt's own abbreviation criterion
#    (PunktTrainer._reclassify_abbrev_types, Kiss & Strunk 2006: Dunning log
#    likelihood scaled by f_length = exp(-len) and f_penalty = len ** -bare_count)
#    scored against this corpus's own counts yields 27 types, 5 of which pysbd
#    already had and most of the rest tokenizer artifacts, and it MISSES "ste",
#    the single largest real mis-cut cause here (13 of 114,663 cuts, "Sault
#    Ste. Marie"), because f_penalty collapses for any type with even a few bare
#    uses. Swapping to a bigger general English list (Moses, CoreNLP) does not
#    help either: the residual here is domain specific, botanical author
#    abbreviations, regnal years, "Sault Ste.", not general English.
# 2. There is no orthographic tell to fall back on. Ranked by every statistic
#    available (length, capitalisation, period attachment ratio), the real
#    abbreviations "ste", "govt", "geo" sit inside a 1,339 type tail of rare
#    proper nouns, "Auriga.", "Bunuel.", "Borken.", "Empoli.", that are
#    CORRECT sentence ends. Nothing separates them, so the lexicon is the only
#    mechanism available, exactly as this file's pysbd comment already says.
#
# Deliberately NOT added, because reading the real cuts showed they end real
# sentences far more often than they truncate one: units ("a length of 188 km.",
# "weighted 94 kg.", "a mantle length of 20 cm.", "27 million lb.") and two
# letter codes that are also club and state names ("Burgos CF.", "Lakeland,
# FL.", "tape delayed MT and PT.").
#
# Round five's audit removed three of round four's own additions on the same
# rescues/swallows measurement used above:
#
#   'fl'  17/61  — the Latin floruit "(fl. AD 408" is real, but the US state
#                  code is far commoner and ends real sentences: "...a record
#                  label operating in Micanopy, FL. It is a subsidiary of...".
#                  Round four's note that "fl" cost "one legitimate cut" was the
#                  one direction figure; over the whole corpus it costs 61.
#   'spp' 2/12   — "...that threatens Quercus spp. The disease is limited to...".
#   'seq' 0/6    — "...found at 15 C.F.R. § 730 et seq. They are administered by
#                  the Bureau of Industry and Security.", never a rescue.
#
# "lit" and "trans" were reported as the same failure and measure the other way,
# so they STAY: 'lit' 1634/49 and 'trans' 190/3, because their real use is the
# gloss that opens an article ("Hősök tere (), lit. Heroes' Square, is one of
# the major squares in Budapest"), which is 2214 of 'lit's 2287 blocks. Dropping
# either trades ~33 good cuts for one. Their residual is real and stays measured
# rather than patched: the band Lit, the album Trans and the participle "lit"
# genuinely end sentences (test_chat.py records both rows), 54 rows in 6,439,528.
CORPUS_ABBREVIATIONS = [
    # Measured mis-cuts, each one read individually in the sample above.
    'auct', 'geo', 'govt', 'ste', 'subgen',
    # Common English abbreviations pysbd omits. Zero measured cost: no accepted
    # cut in the 174,042 row sample ends on any of them.
    'approx', 'assoc', 'intl', 'natl',
    # Reference and bibliographic abbreviations, same zero measured cost.
    # "lit." alone appears period attached 182 times in the sample (the
    # "Chinese: ..., lit. '...'" gloss that opens many articles).
    'abbr', 'bapt', 'ed', 'eds', 'ibid', 'incl', 'lit', 'orig', 'prov',
    'publ', 'resp', 'retd', 'sq', 'subd', 'supp', 'trans', 'transl',
    'vol', 'vols',
]

ABBREVIATIONS = [a for a in PYSBD_ABBREVIATIONS
                 if a not in SENTENCE_ENDING_ABBREVIATIONS] + CORPUS_ABBREVIATIONS

# Where a full Wikipedia summary paragraph gets cut down to its first
# sentence. A split point is a terminator that is NOT preceded by:
#   (?<!\w\.\w.)      an acronym ("U.S. state", "e.g. the", "i.e. it")
#   (?<!\b[A-Z]\.)    a single initial ("J. K. Rowling", "F. W. Murnau"); the
#                     \b keeps it from also blocking a sentence that ends in an
#                     unpunctuated acronym ("...in the USA. It was...")
#   (?<!(?i:\b(?:ABBREVIATIONS)\.))  a known abbreviation, any length
# and IS followed by real whitespace and a capital, i.e. something that can
# actually start the next sentence. Requiring that whitespace is also what
# keeps a decimal off the list: the period in "7.92x57mm" or "2.5 million" is
# followed by a digit, never a space.
#
# The abbreviation lookbehind is one variable width alternation over the whole
# lexicon, which stdlib re cannot compile ("look-behind requires fixed-width
# pattern"), hence the regex module, whose only use here is this. That
# fixed width limit is why the two previous versions of this rule enumerated
# abbreviations BY LETTER COUNT ([A-Z][a-z]\., [A-Z][a-z][a-z]\.), and each
# enumeration was open ended at the top: 4+ letter titles ("Prof.", "Capt.",
# "Corp.") fell straight through, and 638 real corpus rows opening "Prof.
# <Name> is..." were cut down to the bare "Prof.".
#
# Measured over the real corpus (every 211th row of
# corpus/wikipedia_summaries/summaries.tsv, 29,506 rows), lexicon versus
# letter count is better on both axes, not a trade: it cuts 68.0% of rows
# against 67.3%, because the letter count rules also blocked 503 legitimate
# cuts after short capitalized words ("...during World War II.", "...in the
# USA.", "...the World Cup."), and it stops cutting after "vs." and "v." which
# the letter count rules cut straight through.
#
# A lexicon entry cuts both ways here, which is what the round five audit above
# measures per entry: this lookbehind fires only before a capital, so an entry
# blocks a boundary whether the capital opens a name ("Sault Ste. Marie", the
# abbreviation the entry is for) or a new sentence ("...in Micanopy, FL. It is a
# subsidiary of...", a real sentence end the entry destroys). Neither direction
# is free, and neither can be inferred from the token's shape.
SENTENCE_BOUNDARY_RE = regex.compile(
    r'(?<!\w\.\w.)'
    r'(?<!\b[A-Z]\.)'
    r'(?<!(?i:\b(?:' + '|'.join(regex.escape(a) for a in ABBREVIATIONS) + r')\.))'
    r'(?<=[.!?])\s+(?=[A-Z])'
)

# Shortest cut first_sentence() will accept. No lexicon can be complete, so
# the rule above will always miss some abbreviation eventually; this is what
# decides how it fails when it does. Below this, the "sentence" cannot be
# carrying the fact the user asked for and is far likelier to be a mis-cut
# abbreviation, so the full paragraph is kept instead.
#
# 4 is measured, not taste: over the same 29,506 row sample it reverts 26 of
# 20,049 cuts (0.13%), and all 26 were read individually, every one is a real
# mis-cut ("The 2018-19 1." for a Bundesliga season, "Muhammad b.", "Weingut
# Joh.", "Push!" for an album title, "Nehusha (, lit."), none is a real short
# answer. Raising it to 5 would revert 175 instead, and the extra 149 are the
# legitimate, very common "<Surname> is a surname." disambiguation rows, which
# are correct cuts.
#
# What this floor does NOT do, and cannot: it is positional, so it only catches
# a mis-cut that lands inside the first three words. An abbreviation the lexicon
# does not carry, sitting further into the sentence and outside any bracket,
# still yields a fragment ("The company was founded as Gebr.", six words). The
# lexicon is what keeps that rare, and there is no second rule available to
# catch it: ranked by length, capitalisation or period attachment, an unlisted
# abbreviation is indistinguishable from the rare proper nouns that legitimately
# end 1,339 distinct cuts in the sample ("...of Auriga.", "...Burgos CF.",
# "...188 km."), so any orthographic test strict enough to catch "Gebr." also
# destroys "...in Iran." (1,625 correct cuts on its own). Measured, accepted and
# documented rather than patched further.
MIN_SENTENCE_WORDS = 4


def first_sentence(text):
    """The first sentence of text, or the whole of text when cutting it would
    not leave a usable one.

    split, not match: with no real boundary in it (one sentence, or an answer
    the token budget cut off mid word) the whole text comes back as the only
    piece, unchanged, so there is no no match case to guard.

    The two checks are the fallback for what SENTENCE_BOUNDARY_RE's lexicon
    misses, and they are the whole reason a miss is survivable. An unknown
    abbreviation splits the answer at the wrong place, and the wrong place is
    almost always very early in it, so the piece kept is a fragment with no
    content ("Prof.", "Stefan Inglot (b.") rather than a short sentence.
    Returning the full paragraph there is the behavior this cut had before it
    existed: too verbose, but every fact still in it, which is strictly better
    than a confident one word non answer.

    Measured over every 37th row of the real corpus (174,042 rows, disjoint
    from the samples the two rules above were tuned on): 114,663 rows are cut
    (65.9%) and 427 fall back to the full paragraph (0.37% of the rows that
    have a boundary at all).

    Round five's lexicon audit moves the answer on 7,375 rows of the full
    6,439,528, and it is not free in either direction. 60 of those rows read at
    random: 58 are a corrected cut ("...a payments system developed by Block,
    Inc." rather than that plus the next sentence), 2 are a new fragment ("L.
    Greif and Bro., Inc." for the manufactory of that name, "...at 3712 Central
    Ave." before "Ave. SE."). ~3% of the rows it touches trade one failure for
    the other, which is the accepted residual at the same tolerance query.py's
    disambiguation filter and rag_examples() document for their own edge cases.
    """
    first = SENTENCE_BOUNDARY_RE.split(text, maxsplit=1)[0]
    if len(first.split()) < MIN_SENTENCE_WORDS:
        return text
    # More "(" than ")" means the cut landed inside a parenthetical, never at a
    # real sentence end: "Stefan Inglot (b. 1961) is a Polish footballer" cut
    # to "Stefan Inglot (b.". Same class of protection pysbd and
    # pragmatic_segmenter both give punctuation between brackets, and it
    # catches the birth date parentheticals that open a lot of biographies.
    #
    # Strictly greater, not unequal. A ")" with no "(" before it is an upstream
    # artifact of wiki markup stripping, a dropped {{IPA}} or pronunciation
    # template leaving its closing bracket behind, and it is already in the
    # text before any cut happens, so it says nothing about where the cut
    # landed. Unequal counts reverted those rows too: 66 of the 377 rows this
    # guard fired on across the sample were that shape ("Goalpara, Pron: ) is
    # the district headquarters of Goalpara district, Assam, India.",
    # "Bishopbriggs (; ); ) is a town in East Dunbartonshire, Scotland."),
    # complete and correctly bounded first sentences thrown away in favour of
    # the full paragraph, which still contains the same stray bracket anyway.
    if first.count("(") > first.count(")"):
        return text
    return first
