import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "SEED"
OUT_DIR = RESULT_ROOT / "60_seed_sdrmpg_rsc_guarded_selector_analysis"

WEAK_SUBJECTS = {0, 3, 7, 9, 13}
STRONG_SUBJECTS = {10, 12, 14}

SUMMARY_PATHS = {
    "07_seed_sdrmpg_rsc": RESULT_ROOT / "07_seed_sdrmpg_rsc" / "summary.json",
    "16_seed_sdrmpg_source_val_nll_selector": RESULT_ROOT
    / "16_seed_sdrmpg_source_val_nll_selector"
    / "summary.json",
    "27_seed_guarded_selector_paperpack": RESULT_ROOT
    / "27_seed_guarded_selector_paperpack"
    / "summary.json",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def mean(values):
    return sum(values) / len(values) if values else None


def subject_group(subject):
    if subject in WEAK_SUBJECTS:
        return "weak"
    if subject in STRONG_SUBJECTS:
        return "strong"
    return "middle"


def group_mean(summary, subjects):
    subject_rows = summary["subjects"]
    return mean([row["test_acc"] for row in subject_rows if row["subject"] in subjects])


def row_for_main_table(method_id, role, summary, baseline_acc, paper_claim):
    weak_mean = summary.get("weak_mean_test_acc")
    strong_mean = summary.get("strong_mean_test_acc")
    if weak_mean is None:
        weak_mean = group_mean(summary, WEAK_SUBJECTS)
    if strong_mean is None:
        strong_mean = group_mean(summary, STRONG_SUBJECTS)
    return {
        "method_id": method_id,
        "role": role,
        "selector_mode": summary.get("selector_mode", "single_model"),
        "valid_source_only": summary.get("valid_source_only", True),
        "mean_test_acc": summary["mean_test_acc"],
        "mean_test_f1": summary["mean_test_f1"],
        "weak_mean_test_acc": weak_mean,
        "strong_mean_test_acc": strong_mean,
        "delta_acc_vs_07": summary["mean_test_acc"] - baseline_acc,
        "selected_model_counts": json.dumps(summary.get("selected_model_counts", {}), sort_keys=True),
        "paper_claim": paper_claim,
    }


def build_subject_rows(guarded_summary):
    rows = []
    for item in guarded_summary["subjects"]:
        subject = item["subject"]
        rows.append(
            {
                "subject": subject,
                "group": subject_group(subject),
                "selected_model": item["selected_model"],
                "selection_reason": item["selection_reason"],
                "source_val_nll_00": item.get("source_val_nll_00"),
                "source_val_acc_00": item.get("source_val_acc_00"),
                "source_val_nll_01": item.get("source_val_nll_01"),
                "source_val_acc_01": item.get("source_val_acc_01"),
                "source_val_nll_07": item.get("source_val_nll_07"),
                "source_val_acc_07": item.get("source_val_acc_07"),
                "test_acc_00": item.get("test_acc_00"),
                "test_acc_01": item.get("test_acc_01"),
                "test_acc_07": item.get("test_acc_07"),
                "selected_test_acc": item["test_acc"],
                "selected_test_f1": item["test_f1"],
                "delta_vs_07": item["delta_vs_07"],
            }
        )
    return rows


def build_failure_rows():
    return [
        {
            "experiment_family": "MLDG-lite",
            "observed_signal": "sub0 rose to 0.8434 but full dropped to 0.7711",
            "failure_mode": "single weak-subject gain did not transfer to stable all-subject performance",
            "paper_use": "negative evidence for naive episodic meta-regularization",
        },
        {
            "experiment_family": "TERM-only t=0.5",
            "observed_signal": "weak smoke passed, but sub10 dropped from 0.8990 to 0.8340",
            "failure_mode": "tail-risk emphasis helped weak subjects but damaged strong-subject reliability",
            "paper_use": "supports weak/strong boundary analysis",
        },
        {
            "experiment_family": "TERM+RevIN",
            "observed_signal": "sub0 engsmoke failed at 0.7273",
            "failure_mode": "input statistic normalization disrupted the current rPSD + RSC behavior",
            "paper_use": "negative evidence against extra normalization in this protocol",
        },
        {
            "experiment_family": "seed=2023 sdrmpg-rsc",
            "observed_signal": "weak mean reached 0.7476 but sub9 dropped to 0.6313",
            "failure_mode": "random seed improved some weak subjects while worsening a high-risk weak subject",
            "paper_use": "motivates guarded source-val selection instead of seed chasing",
        },
    ]


def build_markdown_report(main_rows, validation):
    return f"""# SDRMPG-RSC-GS Analysis Pack

## Main Claim

SDRMPG-RSC-GS achieves {validation['guarded_acc']:.6f} ACC and {validation['guarded_f1']:.6f} F1 under the original SEED LOSO protocol.

This is a source-val guided guarded selector anchored on `07_seed_sdrmpg_rsc`; it is not a claim that a single `sdrmpg-rsc` checkpoint reaches 80.49%.

## Guarded Rule

1. Select between `01_seed_sdrmpg` and `07_seed_sdrmpg_rsc` by lower source-val NLL.
2. Allow `00_seed_paperfix` only when source-val NLL is lower than both `01` and `07`, source-val ACC is not lower than both, and source-val NLL is at most 0.32.
3. All selection signals are source-validation metrics; target-test labels are used only for final evaluation.

## Result Snapshot

| Method | ACC | F1 | Weak ACC | Strong ACC |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row['method_id']} | {row['mean_test_acc']:.6f} | {row['mean_test_f1']:.6f} | {row['weak_mean_test_acc']:.6f} | {row['strong_mean_test_acc']:.6f} |" for row in main_rows)}

## Safe Wording

Use: `SDRMPG-RSC-GS achieves 80.49% ACC under the original LOSO protocol.`

Avoid: `sdrmpg-rsc single model achieves 80.49% ACC.`
"""


def main():
    summaries = {name: load_json(path) for name, path in SUMMARY_PATHS.items()}
    baseline = summaries["07_seed_sdrmpg_rsc"]
    pair_selector = summaries["16_seed_sdrmpg_source_val_nll_selector"]
    guarded = summaries["27_seed_guarded_selector_paperpack"]
    baseline_acc = baseline["mean_test_acc"]

    main_rows = [
        row_for_main_table(
            "07_seed_sdrmpg_rsc",
            "single-model baseline",
            baseline,
            baseline_acc,
            "sdrmpg-rsc single-model baseline",
        ),
        row_for_main_table(
            "16_seed_sdrmpg_source_val_nll_selector",
            "2-way source-val selector",
            pair_selector,
            baseline_acc,
            "source-val NLL selector over 01/07 reaches 80.00%",
        ),
        row_for_main_table(
            "SDRMPG-RSC-GS / 27_seed_guarded_selector_paperpack",
            "3-way guarded source-val selector",
            guarded,
            baseline_acc,
            "sdrmpg-rsc anchored guarded selector reaches 80.49%",
        ),
    ]

    subject_rows = build_subject_rows(guarded)
    weak_rows = [row for row in subject_rows if row["group"] == "weak"]
    strong_rows = [row for row in subject_rows if row["group"] == "strong"]
    failure_rows = build_failure_rows()

    validation = {
        "guarded_acc": guarded["mean_test_acc"],
        "guarded_f1": guarded["mean_test_f1"],
        "guarded_weak_mean": guarded["weak_mean_test_acc"],
        "guarded_strong_mean": guarded["strong_mean_test_acc"],
        "passes_acc_802_gate": guarded["mean_test_acc"] >= 0.802,
        "passes_f1_floor": guarded["mean_test_f1"] >= 0.8022,
        "passes_weak_floor": guarded["weak_mean_test_acc"] >= 0.7621,
        "valid_source_only": guarded.get("valid_source_only", False),
        "selector_mode": guarded.get("selector_mode"),
        "protected_inputs": {name: str(path) for name, path in SUMMARY_PATHS.items()},
        "generated_outputs": {
            "main_results": str(OUT_DIR / "main_results.csv"),
            "subject_selection_table": str(OUT_DIR / "subject_selection_table.csv"),
            "weak_subjects": str(OUT_DIR / "weak_subjects.csv"),
            "strong_subjects": str(OUT_DIR / "strong_subjects.csv"),
            "failure_taxonomy": str(OUT_DIR / "failure_taxonomy.csv"),
            "paper_claims": str(OUT_DIR / "paper_claims.md"),
        },
    }
    validation["all_acceptance_checks_pass"] = all(
        [
            validation["passes_acc_802_gate"],
            validation["passes_f1_floor"],
            validation["passes_weak_floor"],
            validation["valid_source_only"],
        ]
    )

    write_csv(OUT_DIR / "main_results.csv", main_rows, list(main_rows[0].keys()))
    write_csv(OUT_DIR / "subject_selection_table.csv", subject_rows, list(subject_rows[0].keys()))
    write_csv(OUT_DIR / "weak_subjects.csv", weak_rows, list(subject_rows[0].keys()))
    write_csv(OUT_DIR / "strong_subjects.csv", strong_rows, list(subject_rows[0].keys()))
    write_csv(OUT_DIR / "failure_taxonomy.csv", failure_rows, list(failure_rows[0].keys()))
    write_text(OUT_DIR / "paper_claims.md", build_markdown_report(main_rows, validation))
    write_json(
        OUT_DIR / "report.json",
        {
            "method_name": "SDRMPG-RSC-GS",
            "method_long_name": "sdrmpg-rsc anchored source-val guarded selector",
            "validation": validation,
            "main_results": main_rows,
            "guarded_rule": guarded.get("guarded_3way_rule", {}),
            "selected_model_counts": guarded.get("selected_model_counts", {}),
            "failure_taxonomy": failure_rows,
        },
    )

    if not validation["all_acceptance_checks_pass"]:
        raise SystemExit("Acceptance checks failed. See report.json.")

    print(f"Wrote analysis pack to {OUT_DIR}")
    print(
        "ACC={:.6f} F1={:.6f} weak={:.6f} strong={:.6f}".format(
            validation["guarded_acc"],
            validation["guarded_f1"],
            validation["guarded_weak_mean"],
            validation["guarded_strong_mean"],
        )
    )


if __name__ == "__main__":
    main()
