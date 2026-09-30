# run_eval.py
"""Score each model on the frozen eval and apply the pre-declared gate.

Usage: python run_eval.py candidate      (or: broken)
Exit code 0 = eligible for the next release step, 1 = rejected,
2 = eval inputs were altered, 3 = the eval could not run (e.g. a model file
is missing). A crash must never look like a verdict.
"""
import json, math, sys, tomllib
from fractions import Fraction
from pathlib import Path


def need(share, total):
    """Smallest whole count that meets `share` of `total`, in exact arithmetic.

    Floats are not safe here: 210/300 - 204/300 == 0.019999999999999907, which
    would reject an exactly-2-point gain. Fraction("0.02") * 300 is exactly 6.
    """
    return math.ceil(Fraction(str(share)) * total)


def decide(gate, inc_held, new_held, inc_cases, new_cases):
    """The whole gate as one pure function over per-row right/wrong lists."""
    # An empty or mismatched input must be an error, never a verdict: zip() would
    # silently truncate, and zero rows would pass every condition.
    if not (len(inc_held) == len(new_held) > 0 and len(inc_cases) == len(new_cases) > 0):
        raise ValueError("the gate needs non-empty, equal-length results for both models")
    n_held, n_cases = len(new_held), len(new_cases)
    gain = sum(new_held) - sum(inc_held)
    regressions = [i for i, (a, b) in enumerate(zip(inc_cases, new_cases)) if a and not b]
    return {
        f"held-out gain >= {gate['min_heldout_gain']} (need +{need(gate['min_heldout_gain'], n_held)} of {n_held}, got {gain:+d})":
            gain >= need(gate["min_heldout_gain"], n_held),
        f"authored-case regressions <= {gate['max_case_regressions']}":
            len(regressions) <= gate["max_case_regressions"],
        f"authored accuracy >= {gate['min_case_accuracy']} (need {need(gate['min_case_accuracy'], n_cases)} of {n_cases}, got {sum(new_cases)})":
            sum(new_cases) >= need(gate["min_case_accuracy"], n_cases),
    }, regressions


def main():
    # Everything that can fail lives inside main(), so any crash becomes exit 3.
    import joblib  # third-party: a missing install is a crash, not a verdict
    from freeze import frozen_digest  # step 3's file; missing, it is a crash (3), not a verdict
    digest = frozen_digest()
    if digest != Path("evals/FROZEN.sha256").read_text().strip():
        print("REFUSED: eval inputs changed after freezing. Re-freeze deliberately, and say why.", file=sys.stderr)
        return 2

    gate = tomllib.loads(Path("evals/gate.toml").read_text())["gate"]
    cases = [json.loads(l) for l in Path("evals/cases.jsonl").read_text().splitlines()]
    held = [json.loads(l) for l in Path("evals/heldout.jsonl").read_text().splitlines()]  # the frozen bytes

    def right(m, rows):
        preds = m.predict([r["text"] for r in rows])
        if len(preds) != len(rows):
            raise ValueError(f"got {len(preds)} predictions for {len(rows)} rows")
        return [int(p) == r["label"] for p, r in zip(preds, rows)]

    def score(name):
        m = joblib.load(f"{name}.joblib")
        return right(m, held), right(m, cases)

    inc_held, inc_cases = score("incumbent")
    name = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    new_held, new_cases = score(name)

    n_held, n_cases = len(held), len(cases)
    print(f"{'':12}{f'held-out ({n_held})':>16}{f'authored ({n_cases})':>16}")
    print(f"{'incumbent':12}{sum(inc_held) / n_held:>16.3f}{sum(inc_cases):>13}/{n_cases}")
    print(f"{name:12}{sum(new_held) / n_held:>16.3f}{sum(new_cases):>13}/{n_cases}")

    print(f"\nMisses for {name} on authored cases (read these, don't just count them):")
    for c, ok in zip(cases, new_cases):
        if not ok:
            print(f"  {c['id']:7} [{c['why']}] {c['text']!r} (expected {'pos' if c['label'] else 'neg'})")

    checks, regressions = decide(gate, inc_held, new_held, inc_cases, new_cases)
    print(f"\nGate (frozen inputs {digest[:16]}):")
    for label, passed in checks.items():
        extra = f" {[cases[i]['id'] for i in regressions]}" if "regressions" in label and regressions else ""
        print(f"  {'PASS' if passed else 'FAIL'}  {label}{extra}")
    verdict = all(checks.values())
    print(f"\nVERDICT: {name} {'is eligible for the next release step' if verdict else 'must NOT replace the incumbent'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # a crash is not a verdict: missing model, bad file, anything
        print(f"ERROR: the eval could not run ({type(e).__name__}: {e}). No verdict.", file=sys.stderr)
        code = 3
    sys.exit(code)
