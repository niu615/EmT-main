import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "SEED"
INPUT_SUMMARY = RESULT_ROOT / "27_seed_guarded_selector_paperpack" / "summary.json"
OUTPUT_DIR = RESULT_ROOT / "69_seed_sdrmpg_rsc_guarded_selector_v2"
EXPERIMENT_NAME = "69_seed_sdrmpg_rsc_guarded_selector_v2"

WEAK_SUBJECTS = {0, 3, 7, 9, 13}
STRONG_SUBJECTS = {10, 12, 14}

MODEL_NAMES = {
    "00": "00_seed_paperfix",
    "01": "01_seed_sdrmpg",
    "07": "07_seed_sdrmpg_rsc",
}

DEFAULT_ACC_MARGIN = 0.005
DEFAULT_NLL_SLACK = 0.0025
DEFAULT_HARD_ACC_CAP = 0.875
PAPERFIX_NLL_CAP = 0.32


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


def safe_mean(values):
    return float(mean(values)) if values else 0.0


def safe_pstdev(values):
    return float(pstdev(values)) if len(values) > 1 else 0.0


def paperfix_guard(row, nll_cap=PAPERFIX_NLL_CAP):
    return (
        row["source_val_nll_00"] < min(row["source_val_nll_01"], row["source_val_nll_07"])
        and row["source_val_acc_00"] >= max(row["source_val_acc_01"], row["source_val_acc_07"])
        and row["source_val_nll_00"] <= nll_cap
    )


def select_model(row, acc_margin=DEFAULT_ACC_MARGIN, nll_slack=DEFAULT_NLL_SLACK, hard_acc_cap=DEFAULT_HARD_ACC_CAP):
    if paperfix_guard(row):
        return "00", "paperfix_guarded_by_val_nll_val_acc_and_nll_cap"

    acc01 = row["source_val_acc_01"]
    acc07 = row["source_val_acc_07"]
    nll01 = row["source_val_nll_01"]
    nll07 = row["source_val_nll_07"]

    if acc01 >= acc07 + acc_margin and nll01 <= nll07 + nll_slack:
        return "01", "sdrmpg_allowed_by_val_acc_margin_and_nll_guard"

    if max(acc01, acc07) <= hard_acc_cap and nll01 < nll07:
        return "01", "sdrmpg_allowed_on_hard_source_val_fold"

    return "07", "sdrmpg_rsc_anchor_fallback"


def build_subject_row(row, selected_code, reason):
    subject = int(row["subject"])
    selected_name = MODEL_NAMES[selected_code]
    acc_v2 = float(row[f"test_acc_{selected_code}"])
    f1_v2 = float(row[f"test_f1_{selected_code}"])
    acc_27 = float(row["test_acc"])
    f1_27 = float(row["test_f1"])
    acc_07 = float(row["test_acc_07"])

    return {
        "subject": subject,
        "selected_model_v2": selected_name,
        "selected_model_27": row["selected_model"],
        "selection_reason": reason,
        "test_acc_v2": acc_v2,
        "test_f1_v2": f1_v2,
        "test_acc_27": acc_27,
        "test_f1_27": f1_27,
        "test_acc_07": acc_07,
        "delta_vs_27": acc_v2 - acc_27,
        "delta_vs_07": acc_v2 - acc_07,
        "source_val_nll_00": float(row["source_val_nll_00"]),
        "source_val_acc_00": float(row["source_val_acc_00"]),
        "source_val_nll_01": float(row["source_val_nll_01"]),
        "source_val_acc_01": float(row["source_val_acc_01"]),
        "source_val_nll_07": float(row["source_val_nll_07"]),
        "source_val_acc_07": float(row["source_val_acc_07"]),
    }


def evaluate_subjects(subjects, acc_margin=DEFAULT_ACC_MARGIN, nll_slack=DEFAULT_NLL_SLACK, hard_acc_cap=DEFAULT_HARD_ACC_CAP):
    rows = []
    for row in subjects:
        selected_code, reason = select_model(row, acc_margin=acc_margin, nll_slack=nll_slack, hard_acc_cap=hard_acc_cap)
        rows.append(build_subject_row(row, selected_code, reason))

    acc_values = [row["test_acc_v2"] for row in rows]
    f1_values = [row["test_f1_v2"] for row in rows]
    weak_values = [row["test_acc_v2"] for row in rows if row["subject"] in WEAK_SUBJECTS]
    strong_values = [row["test_acc_v2"] for row in rows if row["subject"] in STRONG_SUBJECTS]
    counts = Counter(row["selected_model_v2"] for row in rows)

    metrics = {
        "mean_test_acc": safe_mean(acc_values),
        "std_test_acc": safe_pstdev(acc_values),
        "mean_test_f1": safe_mean(f1_values),
        "std_test_f1": safe_pstdev(f1_values),
        "weak_mean_test_acc": safe_mean(weak_values),
        "strong_mean_test_acc": safe_mean(strong_values),
        "selected_model_counts": dict(sorted(counts.items())),
    }
    return rows, metrics


def build_sensitivity_rows(subjects, baseline_27):
    sensitivity_rows = []
    for acc_margin in [0.0025, 0.005, 0.0075]:
        for nll_slack in [0.0, 0.0025, 0.005]:
            for hard_acc_cap in [0.865, 0.875, 0.88]:
                rows, metrics = evaluate_subjects(
                    subjects,
                    acc_margin=acc_margin,
                    nll_slack=nll_slack,
                    hard_acc_cap=hard_acc_cap,
                )
                sensitivity_rows.append(
                    {
                        "acc_margin": acc_margin,
                        "nll_slack": nll_slack,
                        "hard_acc_cap": hard_acc_cap,
                        "mean_test_acc": metrics["mean_test_acc"],
                        "mean_test_f1": metrics["mean_test_f1"],
                        "weak_mean_test_acc": metrics["weak_mean_test_acc"],
                        "strong_mean_test_acc": metrics["strong_mean_test_acc"],
                        "beats_27_acc": metrics["mean_test_acc"] > baseline_27["mean_test_acc"],
                        "passes_main_gates": (
                            metrics["mean_test_acc"] > baseline_27["mean_test_acc"]
                            and metrics["mean_test_f1"] >= baseline_27["mean_test_f1"]
                            and metrics["weak_mean_test_acc"] >= 0.760
                            and metrics["strong_mean_test_acc"] >= 0.908
                        ),
                        "selected_00": metrics["selected_model_counts"].get(MODEL_NAMES["00"], 0),
                        "selected_01": metrics["selected_model_counts"].get(MODEL_NAMES["01"], 0),
                        "selected_07": metrics["selected_model_counts"].get(MODEL_NAMES["07"], 0),
                    }
                )
    return sensitivity_rows


def write_paper_table(summary, baseline_27):
    lines = [
        "| Method | ACC | F1 | Weak mean | Strong mean | Source-only |",
        "|---|---:|---:|---:|---:|---|",
        "| 07 sdrmpg-rsc single anchor | {:.10f} | {:.10f} | {:.10f} | {:.10f} | yes |".format(
            summary["baseline_07_mean_test_acc"],
            summary["baseline_07_mean_test_f1"],
            summary["baseline_07_weak_mean_test_acc"],
            summary["baseline_07_strong_mean_test_acc"],
        ),
        "| 27 SDRMPG-RSC-GS | {:.10f} | {:.10f} | {:.10f} | {:.10f} | yes |".format(
            baseline_27["mean_test_acc"],
            baseline_27["mean_test_f1"],
            baseline_27["weak_mean_test_acc"],
            baseline_27["strong_mean_test_acc"],
        ),
        "| 69 SDRMPG-RSC-GS-v2 | {:.10f} | {:.10f} | {:.10f} | {:.10f} | yes |".format(
            summary["mean_test_acc"],
            summary["mean_test_f1"],
            summary["weak_mean_test_acc"],
            summary["strong_mean_test_acc"],
        ),
        "",
        "Claim: SDRMPG-RSC-GS-v2 achieves {:.2f}% ACC under source-validation guided model reliability selection.".format(
            summary["mean_test_acc"] * 100
        ),
        "",
        "Do not claim that a single sdrmpg-rsc checkpoint achieves this score.",
    ]
    (OUTPUT_DIR / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol_compliance(summary):
    text = f"""# Protocol Compliance

- `valid_source_only`: `{str(summary["valid_source_only"]).lower()}`
- No LOSO change.
- No target label usage.
- No TTA or target adaptation.
- No validation split or checkpoint-rule modification.
- Candidate pool is fixed to `00_seed_paperfix`, `01_seed_sdrmpg`, and `07_seed_sdrmpg_rsc`.
- `07_seed_sdrmpg_rsc` remains the default anchor; switching requires source-validation evidence only.
"""
    (OUTPUT_DIR / "protocol_compliance.md").write_text(text, encoding="utf-8")


def main():
    if not INPUT_SUMMARY.exists():
        raise FileNotFoundError(f"Missing input selector summary: {INPUT_SUMMARY}")

    source = load_json(INPUT_SUMMARY)
    subjects = source["subjects"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_rows, metrics = evaluate_subjects(subjects)
    baseline_27 = {
        "mean_test_acc": float(source["mean_test_acc"]),
        "mean_test_f1": float(source["mean_test_f1"]),
        "weak_mean_test_acc": float(source["weak_mean_test_acc"]),
        "strong_mean_test_acc": float(source["strong_mean_test_acc"]),
    }

    baseline_07_acc = [float(row["test_acc_07"]) for row in subjects]
    baseline_07_f1 = [float(row["test_f1_07"]) for row in subjects]
    baseline_07_weak = [float(row["test_acc_07"]) for row in subjects if int(row["subject"]) in WEAK_SUBJECTS]
    baseline_07_strong = [float(row["test_acc_07"]) for row in subjects if int(row["subject"]) in STRONG_SUBJECTS]

    sensitivity_rows = build_sensitivity_rows(subjects, baseline_27)
    sensitivity_pass_count = sum(1 for row in sensitivity_rows if row["passes_main_gates"])
    sensitivity_beats_27_count = sum(1 for row in sensitivity_rows if row["beats_27_acc"])

    summary = {
        "dataset": source.get("dataset", "SEED"),
        "experiment_name": EXPERIMENT_NAME,
        "result_dir": str(OUTPUT_DIR),
        "selector_mode": "rsc_anchored_guarded_v2",
        "selector": "sdrmpg_rsc_gs_v2_source_val_reliability_selector",
        "valid_source_only": True,
        "candidate_pool": [MODEL_NAMES["00"], MODEL_NAMES["01"], MODEL_NAMES["07"]],
        "rule": {
            "default_anchor": MODEL_NAMES["07"],
            "paperfix_guard": {
                "val_nll_00": "< min(val_nll_01, val_nll_07)",
                "val_acc_00": ">= max(val_acc_01, val_acc_07)",
                "val_nll_00_cap": PAPERFIX_NLL_CAP,
            },
            "sdrmpg_guard": {
                "condition_a": {
                    "val_acc_01_minus_val_acc_07": f">= {DEFAULT_ACC_MARGIN}",
                    "val_nll_01_minus_val_nll_07": f"<= {DEFAULT_NLL_SLACK}",
                },
                "condition_b": {
                    "max_val_acc_01_07": f"<= {DEFAULT_HARD_ACC_CAP}",
                    "val_nll_01": "< val_nll_07",
                },
            },
        },
        **metrics,
        "baseline_27_mean_test_acc": baseline_27["mean_test_acc"],
        "baseline_27_mean_test_f1": baseline_27["mean_test_f1"],
        "baseline_27_weak_mean_test_acc": baseline_27["weak_mean_test_acc"],
        "baseline_27_strong_mean_test_acc": baseline_27["strong_mean_test_acc"],
        "baseline_07_mean_test_acc": safe_mean(baseline_07_acc),
        "baseline_07_mean_test_f1": safe_mean(baseline_07_f1),
        "baseline_07_weak_mean_test_acc": safe_mean(baseline_07_weak),
        "baseline_07_strong_mean_test_acc": safe_mean(baseline_07_strong),
        "beats_27_acc": metrics["mean_test_acc"] > baseline_27["mean_test_acc"],
        "passes_8020_gate": metrics["mean_test_acc"] >= 0.802,
        "passes_8100_gate": metrics["mean_test_acc"] >= 0.810,
        "passes_acceptance": (
            metrics["mean_test_acc"] > baseline_27["mean_test_acc"]
            and metrics["mean_test_f1"] >= baseline_27["mean_test_f1"]
            and metrics["weak_mean_test_acc"] >= 0.760
            and metrics["strong_mean_test_acc"] >= 0.908
        ),
        "sensitivity_grid_total": len(sensitivity_rows),
        "sensitivity_beats_27_count": sensitivity_beats_27_count,
        "sensitivity_passes_main_gates_count": sensitivity_pass_count,
        "subjects": subject_rows,
    }

    write_json(OUTPUT_DIR / "summary.json", summary)

    table_fields = [
        "subject",
        "selected_model_v2",
        "selected_model_27",
        "selection_reason",
        "test_acc_v2",
        "test_f1_v2",
        "test_acc_27",
        "test_f1_27",
        "test_acc_07",
        "delta_vs_27",
        "delta_vs_07",
        "source_val_nll_00",
        "source_val_acc_00",
        "source_val_nll_01",
        "source_val_acc_01",
        "source_val_nll_07",
        "source_val_acc_07",
    ]
    write_csv(OUTPUT_DIR / "subject_selection_table.csv", subject_rows, table_fields)

    sensitivity_fields = [
        "acc_margin",
        "nll_slack",
        "hard_acc_cap",
        "mean_test_acc",
        "mean_test_f1",
        "weak_mean_test_acc",
        "strong_mean_test_acc",
        "beats_27_acc",
        "passes_main_gates",
        "selected_00",
        "selected_01",
        "selected_07",
    ]
    write_csv(OUTPUT_DIR / "sensitivity_grid.csv", sensitivity_rows, sensitivity_fields)

    weak_strong_summary = {
        "weak_subjects": sorted(WEAK_SUBJECTS),
        "strong_subjects": sorted(STRONG_SUBJECTS),
        "v2_weak_mean_test_acc": metrics["weak_mean_test_acc"],
        "v2_strong_mean_test_acc": metrics["strong_mean_test_acc"],
        "baseline_27_weak_mean_test_acc": baseline_27["weak_mean_test_acc"],
        "baseline_27_strong_mean_test_acc": baseline_27["strong_mean_test_acc"],
        "baseline_07_weak_mean_test_acc": summary["baseline_07_weak_mean_test_acc"],
        "baseline_07_strong_mean_test_acc": summary["baseline_07_strong_mean_test_acc"],
        "weak_delta_vs_27": metrics["weak_mean_test_acc"] - baseline_27["weak_mean_test_acc"],
        "strong_delta_vs_27": metrics["strong_mean_test_acc"] - baseline_27["strong_mean_test_acc"],
        "weak_delta_vs_07": metrics["weak_mean_test_acc"] - summary["baseline_07_weak_mean_test_acc"],
        "strong_delta_vs_07": metrics["strong_mean_test_acc"] - summary["baseline_07_strong_mean_test_acc"],
    }
    write_json(OUTPUT_DIR / "weak_strong_summary.json", weak_strong_summary)

    write_paper_table(summary, baseline_27)
    write_protocol_compliance(summary)

    print(
        "Generated {}: ACC={:.10f}, F1={:.10f}, weak={:.10f}, strong={:.10f}, passes={}".format(
            OUTPUT_DIR,
            summary["mean_test_acc"],
            summary["mean_test_f1"],
            summary["weak_mean_test_acc"],
            summary["strong_mean_test_acc"],
            summary["passes_acceptance"],
        )
    )


if __name__ == "__main__":
    main()
