# sensitivity.py
"""Optional: how typical is the tutorial's result? Retrain the incumbent and
candidate with train.py's exact recipe on training shuffle seeds 0-19 and apply
the same gate to each pair.

This is a check run after the result, not part of the gate. It measures how
much the verdict varies between training runs of this recipe, not how a model
behaves on real traffic. Takes under a minute.
"""
import json, tomllib
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from run_eval import decide, need

REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"  # same pinned commit as train.py
gate = tomllib.loads(Path("evals/gate.toml").read_text())["gate"]
cases = [json.loads(l) for l in Path("evals/cases.jsonl").read_text().splitlines()]
held = [json.loads(l) for l in Path("evals/heldout.jsonl").read_text().splitlines()]
base = load_dataset("stanfordnlp/sst2", split="train", revision=REVISION)

def fit(train, n, ngrams):  # train.py's fit(), without the shuffled-label option
    rows = train.select(range(n))
    model = make_pipeline(TfidfVectorizer(ngram_range=ngrams, min_df=2),
                          LogisticRegression(max_iter=1000))
    return model.fit(rows["sentence"], list(rows["label"]))

def right(model, rows):
    return [int(p) == r["label"] for p, r in zip(model.predict([r["text"] for r in rows]), rows)]

seeds = range(20)
tally, regressed = Counter(), Counter()
for seed in seeds:
    train = base.shuffle(seed=seed)
    inc, cand = fit(train, 2_000, (1, 1)), fit(train, 20_000, (1, 2))
    inc_held, inc_cases, new_held, new_cases = right(inc, held), right(inc, cases), right(cand, held), right(cand, cases)
    checks, regressions = decide(gate, inc_held, new_held, inc_cases, new_cases)
    gain_ok, _, _ = checks.values()
    tally["gain cleared"] += gain_ok
    tally["at least one regression"] += bool(regressions)
    tally["whole gate passed"] += all(checks.values())
    tally["both 14/20"] += sum(inc_cases) == 14 and sum(new_cases) == 14
    tally["incumbent under the floor"] += sum(inc_cases) < need(gate["min_case_accuracy"], len(cases))
    regressed.update(cases[i]["id"] for i in regressions)
    print(f"seed {seed:2}: incumbent {sum(inc_held)}/{len(held)} {sum(inc_cases)}/{len(cases)}, "
          f"candidate {sum(new_held)}/{len(held)} {sum(new_cases)}/{len(cases)}, "
          f"regressed {[cases[i]['id'] for i in regressions]}, gate {'PASS' if all(checks.values()) else 'fail'}")

print(f"\nOut of {len(seeds)} training shuffles:")
for label, n in tally.items():
    print(f"  {label:26} {n}")
print("  regressions by case      ", ", ".join(f"{c} {n}" for c, n in regressed.most_common()))
