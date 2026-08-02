"""Generate real tool call traces for the SFT tool corpus (SPEC.md Part 3's
CALL: shape), by actually running toolstore/tools.py rather than fabricating
results, matching the Toolformer pattern of training on real executed calls,
both successful and failed.

Writes toolstore/corpus/tool_examples/calls.jsonl, one JSON object per line:
{"question": ..., "call": "CALL: tool_name(args)", "result": ..., "ok": bool}
JSONL rather than TSV (unlike the sibling wikipedia_summaries corpus)
because this record has mixed typed fields (numeric or string result, plus
a boolean), not two flat strings.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from tools import calculator, current_datetime

OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus", "tool_examples")
OUT_FILE = os.path.join(OUT_DIR, "calls.jsonl")

CALCULATOR_CASES = [
    ("What is 2 plus 2?", "2+2"),
    ("What is 17 times 23?", "17*23"),
    ("What is 100 divided by 4?", "100/4"),
    ("What is 2 to the power of 10?", "2**10"),
    ("What is 9 minus 15?", "9-15"),
    ("What is 3 times 3 plus 4?", "3*3+4"),
    ("What is 144 divided by 12?", "144/12"),
    ("What is 7 to the power of 3?", "7**3"),
    ("What is negative 5 plus 12?", "-5+12"),
    ("What is 50 minus 8 times 2?", "50-8*2"),
    # deliberately failing cases too, so the model also sees what a failed call looks like,
    # each tripping a different real guard in tools.py rather than repeating the same failure mode
    ("What is 5 divided by 0?", "5/0"),
    ("What is the square root of banana?", "sqrt(banana)"),
    ("What is 9 to the power of 9 to the power of 9?", "9**9**9"),
]


def run_calculator_cases():
    rows = []
    for question, expr in CALCULATOR_CASES:
        call = f"CALL: calculator({expr})"
        try:
            result = calculator(expr)
            rows.append({"question": question, "call": call, "result": result, "ok": True})
        except ValueError as e:
            rows.append({"question": question, "call": call, "result": str(e), "ok": False})
    return rows


def run_datetime_cases():
    rows = []
    questions = [
        "What is the current date and time?",
        "What time is it right now?",
        "Can you tell me today's date?",
    ]
    for question in questions:
        call = "CALL: current_datetime()"
        result = current_datetime()
        rows.append({"question": question, "call": call, "result": result, "ok": True})
    return rows


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run_calculator_cases() + run_datetime_cases()
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ok_count = sum(1 for r in rows if r["ok"])
    print(f"wrote {len(rows)} tool call traces ({ok_count} ok, {len(rows) - ok_count} failed) to {OUT_FILE}")
