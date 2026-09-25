"""
pipeline.py — run every step in order.

Run it:  python -m volee_ml.pipeline

It's just the five steps one after another. Each step also runs on its own
(`python -m volee_ml.data`, `...features`, `...train`, `...evaluate`,
`...volee`), which is the better way to learn: run one, open its output
file, read it, then run the next.
"""

import runpy

STEPS = [
    ("1. Download and clean matches", "volee_ml.data"),
    ("2. Build features by replaying history", "volee_ml.features"),
    ("3. Train the models", "volee_ml.train"),
    ("4. Evaluate and write reports/results.md", "volee_ml.evaluate"),
    ("5. Score Volee's demo matches", "volee_ml.volee"),
]

if __name__ == "__main__":
    for title, module in STEPS:
        print(f"\n=== {title} ===")
        runpy.run_module(module, run_name="__main__")
