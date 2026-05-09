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


def group_val_mean(summary, subjects):
    subject_rows = summary["subjects"]
    return mean([row["val_acc"] for row in subject_rows if row["subject"] in subjects])


def selected_source_metrics(item):
    selected_model = item["selected_model"]
    if selected_model == "00_seed_paperfix":
        return item.get("source_val_acc_00"), item.get("source_val_nll_00")
    if selected_model == "01_seed_sdrmpg":
        return item.get("source_val_acc_01"), item.get("source_val_nll_01")
    if selected_model == "07_seed_sdrmpg_rsc":
        return item.get("source_val_acc_07"), item.get("source_val_nll_07")
    return None, None


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


def build_subject_boundary_rows(baseline_summary, guarded_summary):
    baseline_by_subject = {row["subject"]: row for row in baseline_summary["subjects"]}
    rows = []
    for item in guarded_summary["subjects"]:
        subject = item["subject"]
        baseline = baseline_by_subject[subject]
        selected_val_acc, selected_val_nll = selected_source_metrics(item)
        baseline_gap = baseline["val_acc"] - baseline["test_acc"]
        selected_gap = selected_val_acc - item["test_acc"] if selected_val_acc is not None else None
        rows.append(
            {
                "subject": subject,
                "group": subject_group(subject),
                "baseline_07_val_acc": baseline["val_acc"],
                "baseline_07_test_acc": baseline["test_acc"],
                "baseline_07_val_test_gap": baseline_gap,
                "selected_model": item["selected_model"],
                "selected_source_val_acc": selected_val_acc,
                "selected_source_val_nll": selected_val_nll,
                "selected_test_acc": item["test_acc"],
                "selected_val_test_gap": selected_gap,
                "delta_test_acc_vs_07": item["delta_vs_07"],
                "risk_note": risk_note_for_subject(subject, item["delta_vs_07"]),
            }
        )
    return rows


def risk_note_for_subject(subject, delta_vs_07):
    if subject in WEAK_SUBJECTS and delta_vs_07 > 0.03:
        return "weak_gain_supports_reliability_selection"
    if subject in STRONG_SUBJECTS and delta_vs_07 < -0.03:
        return "strong_subject_regression_risk"
    if delta_vs_07 < -0.03:
        return "selector_guardrail_needed"
    return "stable_or_small_delta"


def build_group_boundary_rows(baseline_summary, pair_selector, guarded_summary):
    rows = []
    for group_name, subjects in [
        ("weak", WEAK_SUBJECTS),
        ("strong", STRONG_SUBJECTS),
        ("all", {row["subject"] for row in baseline_summary["subjects"]}),
    ]:
        baseline_test = group_mean(baseline_summary, subjects)
        baseline_val = group_val_mean(baseline_summary, subjects)
        pair_test = group_mean(pair_selector, subjects)
        guarded_test = group_mean(guarded_summary, subjects)
        rows.append(
            {
                "group": group_name,
                "num_subjects": len(subjects),
                "baseline_07_test_acc": baseline_test,
                "baseline_07_source_val_acc": baseline_val,
                "baseline_07_val_test_gap": baseline_val - baseline_test,
                "pair_16_test_acc": pair_test,
                "pair_16_delta_vs_07": pair_test - baseline_test,
                "guarded_27_test_acc": guarded_test,
                "guarded_27_delta_vs_07": guarded_test - baseline_test,
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
            "experiment_family": "NLL guarded checkpointing",
            "observed_signal": "weak smoke passed at 0.7465, but strong smoke failed with sub12=0.8340",
            "failure_mode": "source-val NLL checkpoint selection helped weak-side behavior but damaged a strong subject",
            "paper_use": "supports guarded model reliability over single-checkpoint optimization",
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


def build_manuscript_outline(validation):
    return f"""# Manuscript Outline: SDRMPG-RSC-GS

## Core Thesis

The main limitation of `sdrmpg-rsc` on SEED is not insufficient representation capacity. The model already achieves high strong-subject accuracy, but weak subjects exhibit a large source-val to target-test gap. SDRMPG-RSC-GS addresses this by modeling subject-dependent candidate reliability using source-validation signals only.

## Suggested Contributions

1. A strong `sdrmpg-rsc` anchor candidate built on TGC, RMPG, TCT, and RSC.
2. A source-val guided guarded selector that defaults to `sdrmpg-rsc` and only switches when validation evidence is strong.
3. A weak/strong subject boundary analysis showing why average-risk single-model tuning repeatedly fails.
4. A negative-result taxonomy that explains why common DG, normalization, loss, checkpoint, and seed-chasing variants are unreliable under this protocol.

## Main Result

`SDRMPG-RSC-GS` achieves {validation['guarded_acc']:.6f} ACC and {validation['guarded_f1']:.6f} F1 under the original SEED LOSO protocol, with weak mean {validation['guarded_weak_mean']:.6f}.

## Required Wording

Use: `SDRMPG-RSC-GS achieves 80.49% ACC under the original LOSO protocol.`

Avoid: `sdrmpg-rsc single model achieves 80.49% ACC.`
"""


def build_protocol_compliance():
    return """# Protocol Compliance Statement

- LOSO protocol is unchanged.
- Target labels are never used for model selection.
- No target adaptation, no TTA, and no target-test feedback are used.
- Source-validation metrics are the only selector inputs.
- The protected `07_seed_sdrmpg_rsc`, `17_seed_guarded_3way_source_val_selector`, and `27_seed_guarded_selector_paperpack` result directories must not be overwritten.
- Test labels are used only once for final evaluation and reporting.
"""


def build_experiment_governance():
    return """# Future Experiment Governance

Single-model exploration is frozen unless a proposed method is mechanistically different from the failed families and has explicit smoke-stop criteria.

Do not repeat:

- Backbone, attention, token, head, or capacity expansion.
- Explicit DG/alignment/source weighting/sampling methods.
- KD, consistency, R-Drop, or EMA-style regularization.
- Mixup, perturbation, pooled-feature repair, or feature-front-end replacement.
- Checkpoint soup, calibration, SWAD, SAM, or seed chasing.
- RSC variants that only retune the same weak/strong trade-off.

If a new idea is unavoidable, it must start from `feat/seed_sdrmpg_rsc`, use a new branch and result directory, pass engsmoke, weak smoke, and strong smoke before full, and be deleted immediately if it fails.
"""


def build_reference_notes():
    return """# Reference Notes

- RSC: Representation Self-Challenging for domain generalization, ECCV 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf
- MLDG: Learning to Generalize, AAAI 2018. https://ojs.aaai.org/index.php/AAAI/article/view/11596
- TERM: Tilted Empirical Risk Minimization, JMLR 2023. https://jmlr.org/beta/papers/v24/21-1095.html
- DORO: Distributionally Robust Optimization with Outliers, ICML 2021. https://icml.cc/virtual/2021/spotlight/8920
- Fishr: Invariant Gradient Variances Across Domains, ICML 2022. https://proceedings.mlr.press/v162/rame22a.html
- SWAD: Stochastic Weight Averaging Densely for Domain Generalization, NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html
- RevIN: Reversible Instance Normalization, ICLR 2022. https://mlanthology.org/iclr/2022/kim2022iclr-reversible/
- DomainBed: In Search of Lost Domain Generalization, ICLR 2021. https://iclr.cc/virtual/2021/poster/2998
- EEG ensemble evidence: IEEE JBHI 2024. https://pubmed.ncbi.nlm.nih.gov/38954558/
- DMMR: Domain Mixup and Multidomain Disentangled Representation for EEG emotion recognition, AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/27819
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
    subject_boundary_rows = build_subject_boundary_rows(baseline, guarded)
    group_boundary_rows = build_group_boundary_rows(baseline, pair_selector, guarded)
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
            "subject_boundary_analysis": str(OUT_DIR / "subject_boundary_analysis.csv"),
            "weak_strong_boundary_summary": str(OUT_DIR / "weak_strong_boundary_summary.csv"),
            "failure_taxonomy": str(OUT_DIR / "failure_taxonomy.csv"),
            "paper_claims": str(OUT_DIR / "paper_claims.md"),
            "manuscript_outline": str(OUT_DIR / "manuscript_outline.md"),
            "protocol_compliance": str(OUT_DIR / "protocol_compliance.md"),
            "experiment_governance": str(OUT_DIR / "experiment_governance.md"),
            "reference_notes": str(OUT_DIR / "reference_notes.md"),
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
    write_csv(
        OUT_DIR / "subject_boundary_analysis.csv",
        subject_boundary_rows,
        list(subject_boundary_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "weak_strong_boundary_summary.csv",
        group_boundary_rows,
        list(group_boundary_rows[0].keys()),
    )
    write_csv(OUT_DIR / "failure_taxonomy.csv", failure_rows, list(failure_rows[0].keys()))
    write_text(OUT_DIR / "paper_claims.md", build_markdown_report(main_rows, validation))
    write_text(OUT_DIR / "manuscript_outline.md", build_manuscript_outline(validation))
    write_text(OUT_DIR / "protocol_compliance.md", build_protocol_compliance())
    write_text(OUT_DIR / "experiment_governance.md", build_experiment_governance())
    write_text(OUT_DIR / "reference_notes.md", build_reference_notes())
    write_json(
        OUT_DIR / "report.json",
        {
            "method_name": "SDRMPG-RSC-GS",
            "method_long_name": "sdrmpg-rsc anchored source-val guarded selector",
            "validation": validation,
            "main_results": main_rows,
            "guarded_rule": guarded.get("guarded_3way_rule", {}),
            "selected_model_counts": guarded.get("selected_model_counts", {}),
            "weak_strong_boundary_summary": group_boundary_rows,
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
