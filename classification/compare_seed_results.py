import argparse
import json
from pathlib import Path


def resolve_summary(path_str):
    path = Path(path_str).resolve()
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return path, json.load(file)


def compare_summaries(baseline_path, candidate_path, min_acc_gain=0.0, min_f1_gain=0.0):
    baseline_summary_path, baseline = resolve_summary(baseline_path)
    candidate_summary_path, candidate = resolve_summary(candidate_path)

    acc_gain = candidate["mean_test_acc"] - baseline["mean_test_acc"]
    f1_gain = candidate["mean_test_f1"] - baseline["mean_test_f1"]
    keep = acc_gain > min_acc_gain and f1_gain >= min_f1_gain

    return {
        "baseline_path": str(baseline_summary_path),
        "candidate_path": str(candidate_summary_path),
        "baseline_acc": float(baseline["mean_test_acc"]),
        "baseline_f1": float(baseline["mean_test_f1"]),
        "candidate_acc": float(candidate["mean_test_acc"]),
        "candidate_f1": float(candidate["mean_test_f1"]),
        "acc_gain": float(acc_gain),
        "f1_gain": float(f1_gain),
        "min_acc_gain": float(min_acc_gain),
        "min_f1_gain": float(min_f1_gain),
        "keep": bool(keep),
    }


def print_decision(decision):
    print(f"baseline:       {decision['baseline_path']}")
    print(f"candidate:      {decision['candidate_path']}")
    print(f"baseline acc:   {decision['baseline_acc']:.6f}")
    print(f"baseline f1:    {decision['baseline_f1']:.6f}")
    print(f"candidate acc:  {decision['candidate_acc']:.6f}")
    print(f"candidate f1:   {decision['candidate_f1']:.6f}")
    print(f"acc gain:       {decision['acc_gain']:.6f}")
    print(f"f1 gain:        {decision['f1_gain']:.6f}")
    print("decision:       KEEP" if decision["keep"] else "decision:       DROP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline summary.json or experiment directory")
    parser.add_argument("--candidate", required=True, help="Candidate summary.json or experiment directory")
    parser.add_argument("--min-acc-gain", type=float, default=0.0)
    parser.add_argument("--min-f1-gain", type=float, default=0.0)
    parser.add_argument("--decision-file", type=str, default=None)
    args = parser.parse_args()

    decision = compare_summaries(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        min_acc_gain=args.min_acc_gain,
        min_f1_gain=args.min_f1_gain,
    )
    print_decision(decision)

    if args.decision_file:
        decision_file = Path(args.decision_file).resolve()
        decision_file.parent.mkdir(parents=True, exist_ok=True)
        with decision_file.open("w", encoding="utf-8") as file:
            json.dump(decision, file, indent=2, ensure_ascii=False)

    raise SystemExit(0 if decision["keep"] else 1)
