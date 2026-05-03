import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.cross_validation import CrossValidation
from base.utils import eegDataset, get_metrics, get_model, seed_all, set_gpu
from run_seed_experiment import PRESETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "SEED"
DEFAULT_OUTPUT = RESULT_ROOT / "16_seed_sdrmpg_source_val_nll_selector"
DEFAULT_GUARDED_OUTPUT = RESULT_ROOT / "27_seed_guarded_selector_paperpack"
EXPERIMENTS = {
    "00_seed_paperfix": "paperfix",
    "01_seed_sdrmpg": "sdrmpg",
    "07_seed_sdrmpg_rsc": "sdrmpg_rsc",
}
PAIR_EXPERIMENTS = ("01_seed_sdrmpg", "07_seed_sdrmpg_rsc")
GUARDED_EXPERIMENTS = ("00_seed_paperfix", "01_seed_sdrmpg", "07_seed_sdrmpg_rsc")
EXPECTED_RESULTS = {
    "pair_nll": (0.8000420875420875, 0.7952559053268958),
    "guarded_3way": (0.8049242424242424, 0.8022636506611381),
}


def parse_int_list(raw_value):
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_eval_args(preset_name, output_dir):
    preset = PRESETS[preset_name]
    args = SimpleNamespace(
        ROOT=str(PROJECT_ROOT),
        dataset="SEED",
        data_format="rPSD",
        label_type="NA",
        num_class=2,
        num_channel=62,
        num_feature=7,
        model="EmT",
        graph_type="BL",
        layers_graph=parse_int_list(preset.get("layers_graph", "1,2")),
        layers_transformer=8,
        num_adj=2,
        hidden_graph=32,
        num_head=16,
        dim_head=32,
        graph2token="Linear",
        encoder_type="Cheby",
        K=4,
        dropout=0.25,
        alpha=0.25,
        use_simam=bool(preset.get("use_simam", 0)),
        fusion_mode=preset.get("fusion_mode", "mean"),
        pooling_mode=preset.get("pooling_mode", "mean"),
        sta_kernels=parse_int_list(preset.get("sta_kernels", "3")),
        adj_sparsity_weight=float(preset.get("adj_sparsity_weight", 0.0)),
        adj_diversity_weight=float(preset.get("adj_diversity_weight", 0.0)),
        result_dir=str(output_dir),
        results_file=str(output_dir / "results_SEED.txt"),
    )
    return args


def prepare_fold_data(cv, target_subject, subjects):
    data_train, label_train = [], []
    data_test, label_test = cv.load_per_subject(target_subject)
    for source_subject in subjects:
        if source_subject == target_subject:
            continue
        data_temp, label_temp = cv.load_per_subject(source_subject)
        data_train.extend(data_temp)
        label_train.extend(label_temp)

    data_train, label_train, data_test, label_test = cv.prepare_data(
        data_train=data_train,
        label_train=label_train,
        data_test=data_test,
        label_test=label_test,
    )
    _, _, data_val, label_val = cv.split_balance_class(
        data=data_train,
        label=label_train,
        train_rate=0.8,
        random=False,
    )
    return data_val, label_val, data_test, label_test


def checkpoint_path(experiment_name, subject, data_format="rPSD", label_type="NA"):
    return (
        RESULT_ROOT
        / experiment_name
        / "checkpoints"
        / f"model_{data_format}_{label_type}"
        / f"sub{subject}_trial0.pth"
    )


def load_model(args, experiment_name, subject, device):
    model = get_model(args).to(device)
    path = checkpoint_path(experiment_name, subject, args.data_format, args.label_type)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_logits(model, data, label, device, batch_size):
    loader = DataLoader(eegDataset(data, label), batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits_all.append(model(x_batch).cpu())
            labels_all.append(y_batch.cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def nll_from_logits(logits, labels):
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def metrics_from_logits(logits, labels):
    pred = torch.argmax(logits, dim=1).tolist()
    act = labels.tolist()
    acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
    return float(acc), float(f1)


def short_name(experiment_name):
    if experiment_name.startswith("00_"):
        return "00"
    if experiment_name.startswith("01_"):
        return "01"
    if experiment_name.startswith("07_"):
        return "07"
    raise ValueError(f"Unknown experiment name: {experiment_name}")


def select_pair_nll(row):
    if row["source_val_nll_01"] < row["source_val_nll_07"]:
        return "01_seed_sdrmpg", "lower_source_val_nll_01_vs_07"
    return "07_seed_sdrmpg_rsc", "lower_source_val_nll_01_vs_07"


def select_guarded_3way(row):
    pair_model, pair_reason = select_pair_nll(row)
    paperfix_allowed = (
        row["source_val_nll_00"] < min(row["source_val_nll_01"], row["source_val_nll_07"])
        and row["source_val_acc_00"] >= max(row["source_val_acc_01"], row["source_val_acc_07"])
        and row["source_val_nll_00"] <= 0.32
    )
    if paperfix_allowed:
        return "00_seed_paperfix", "paperfix_guarded_by_val_nll_val_acc_and_nll_cap"
    return pair_model, f"fallback_{pair_reason}"


def select_model(row, selector_mode):
    if selector_mode == "pair_nll":
        return select_pair_nll(row)
    if selector_mode == "guarded_3way":
        return select_guarded_3way(row)
    raise ValueError(f"Unsupported selector mode: {selector_mode}")


def mean(values):
    return float(np.mean(values)) if values else 0.0


def std(values):
    return float(np.std(values)) if values else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def model_key(model_name):
    return short_name(model_name)


def summarize_delta(row):
    return {
        "subject": row["subject"],
        "selected_model": row["selected_model"],
        "test_acc": row["test_acc"],
        "test_acc_07": row["test_acc_07"],
        "delta_vs_07": row["delta_vs_07"],
        "test_f1": row["test_f1"],
        "test_f1_07": row["test_f1_07"],
    }


def write_paperpack_reports(output_dir, rows, summary):
    per_subject_fields = [
        "subject",
        "selected_model",
        "selection_reason",
        "source_val_nll_00",
        "source_val_acc_00",
        "source_val_nll_01",
        "source_val_acc_01",
        "source_val_nll_07",
        "source_val_acc_07",
        "test_acc_00",
        "test_acc_01",
        "test_acc_07",
        "test_acc",
        "test_f1",
        "delta_vs_07",
    ]
    per_subject_rows = [
        {field: row.get(field, "") for field in per_subject_fields}
        for row in rows
    ]
    write_csv(output_dir / "per_subject_selection.csv", per_subject_rows)
    write_csv(output_dir / "delta_vs_07.csv", [summarize_delta(row) for row in rows])

    short_counts = {}
    for model_name, count in summary["selected_model_counts"].items():
        short_counts[model_key(model_name)] = count
    write_json(
        output_dir / "model_count.json",
        {
            "selected_model_counts": summary["selected_model_counts"],
            "short_selected_model_counts": short_counts,
        },
    )
    write_json(
        output_dir / "weak_strong_summary.json",
        {
            "weak_subjects": [0, 3, 7, 9, 13],
            "strong_subjects": [10, 12, 14],
            "weak_mean_test_acc": summary["weak_mean_test_acc"],
            "strong_mean_test_acc": summary["strong_mean_test_acc"],
            "baseline_07_mean_test_acc": summary["baseline_07_mean_test_acc"],
            "selector_mean_test_acc": summary["mean_test_acc"],
            "selector_mean_test_f1": summary["mean_test_f1"],
        },
    )

    paper_table = [
        "# Guarded Source-Val Selector Paper Table",
        "",
        "| Method | ACC | F1 | Weak ACC | Strong ACC |",
        "|---|---:|---:|---:|---:|",
        "| sdrmpg-rsc (07) | {:.10f} | {:.10f} | - | - |".format(
            summary["baseline_07_mean_test_acc"],
            summary["baseline_07_mean_test_f1"],
        ),
        "| Guarded 3-way selector | {:.10f} | {:.10f} | {:.10f} | {:.10f} |".format(
            summary["mean_test_acc"],
            summary["mean_test_f1"],
            summary["weak_mean_test_acc"],
            summary["strong_mean_test_acc"],
        ),
        "",
        "Selected model counts: `00={}`, `01={}`, `07={}`.".format(
            short_counts.get("00", 0),
            short_counts.get("01", 0),
            short_counts.get("07", 0),
        ),
        "",
        "Guard rule: paperfix is selected only if `val_nll_00 < min(val_nll_01, val_nll_07)`, "
        "`val_acc_00 >= max(val_acc_01, val_acc_07)`, and `val_nll_00 <= 0.32`; otherwise the "
        "selector falls back to the lower source-val NLL between `sdrmpg` and `sdrmpg-rsc`.",
    ]
    (output_dir / "paper_table.md").write_text("\n".join(paper_table) + "\n", encoding="utf-8")


def run_selector(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.gpu:
        set_gpu(args.gpu)
    seed_all(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = parse_int_list(args.subjects)

    cv_args = build_eval_args("sdrmpg_rsc", output_dir)
    cv = CrossValidation(cv_args)
    experiment_names = PAIR_EXPERIMENTS if args.selector_mode == "pair_nll" else GUARDED_EXPERIMENTS
    model_args = {
        experiment_name: build_eval_args(preset_name, output_dir)
        for experiment_name, preset_name in EXPERIMENTS.items()
        if experiment_name in experiment_names
    }

    rows = []
    for subject in subjects:
        data_val, label_val, data_test, label_test = prepare_fold_data(cv, subject, subjects)
        row = {"subject": int(subject)}
        test_metrics = {}

        for experiment_name in experiment_names:
            model = load_model(model_args[experiment_name], experiment_name, subject, device)
            val_logits, val_labels = collect_logits(model, data_val, label_val, device, args.batch_size)
            test_logits, test_labels = collect_logits(model, data_test, label_test, device, args.batch_size)
            val_nll = nll_from_logits(val_logits, val_labels)
            val_acc, val_f1 = metrics_from_logits(val_logits, val_labels)
            test_acc, test_f1 = metrics_from_logits(test_logits, test_labels)

            name = short_name(experiment_name)
            row[f"source_val_nll_{name}"] = val_nll
            row[f"source_val_acc_{name}"] = val_acc
            row[f"source_val_f1_{name}"] = val_f1
            row[f"test_acc_{name}"] = test_acc
            row[f"test_f1_{name}"] = test_f1
            test_metrics[experiment_name] = {"test_acc": test_acc, "test_f1": test_f1}

        selected_model, selection_reason = select_model(row, args.selector_mode)
        selected = test_metrics[selected_model]
        baseline = test_metrics["07_seed_sdrmpg_rsc"]
        row["selected_model"] = selected_model
        row["selection_reason"] = selection_reason
        row["test_acc"] = selected["test_acc"]
        row["test_f1"] = selected["test_f1"]
        row["delta_vs_07"] = selected["test_acc"] - baseline["test_acc"]
        rows.append(row)
        print(
            "subject {subject}: select {selected_model}, acc={test_acc:.6f}, "
            "f1={test_f1:.6f}, delta_vs_07={delta_vs_07:+.6f}".format(**row),
            flush=True,
        )

    weak_subjects = {0, 3, 7, 9, 13}
    strong_subjects = {10, 12, 14}
    weak_rows = [row for row in rows if row["subject"] in weak_subjects]
    strong_rows = [row for row in rows if row["subject"] in strong_subjects]

    selected_model_counts = {}
    for row in rows:
        selected_model_counts[row["selected_model"]] = selected_model_counts.get(row["selected_model"], 0) + 1

    summary = {
        "dataset": "SEED",
        "experiment_name": output_dir.name,
        "result_dir": str(output_dir),
        "selector_mode": args.selector_mode,
        "selector": (
            "source_val_lower_nll_01_sdrmpg_vs_07_sdrmpg_rsc"
            if args.selector_mode == "pair_nll"
            else "guarded_3way_source_val_selector"
        ),
        "valid_source_only": True,
        "guarded_3way_rule": {
            "enabled": args.selector_mode == "guarded_3way",
            "paperfix_conditions": [
                "val_nll_00 < min(val_nll_01, val_nll_07)",
                "val_acc_00 >= max(val_acc_01, val_acc_07)",
                "val_nll_00 <= 0.32",
            ],
        },
        "mean_test_acc": mean([row["test_acc"] for row in rows]),
        "std_test_acc": std([row["test_acc"] for row in rows]),
        "mean_test_f1": mean([row["test_f1"] for row in rows]),
        "std_test_f1": std([row["test_f1"] for row in rows]),
        "weak_mean_test_acc": mean([row["test_acc"] for row in weak_rows]),
        "strong_mean_test_acc": mean([row["test_acc"] for row in strong_rows]),
        "selected_model_counts": selected_model_counts,
        "baseline_07_mean_test_acc": mean([row["test_acc_07"] for row in rows]),
        "baseline_07_mean_test_f1": mean([row["test_f1_07"] for row in rows]),
        "subjects": rows,
    }
    summary["beats_07_baseline"] = summary["mean_test_acc"] > summary["baseline_07_mean_test_acc"]
    summary["passes_8020_gate"] = summary["mean_test_acc"] >= 0.8020

    write_csv(output_dir / "subject_table.csv", rows)
    write_json(output_dir / "summary.json", summary)
    write_paperpack_reports(output_dir, rows, summary)

    print(
        "Selector mean ACC={:.12f}, F1={:.12f}".format(
            summary["mean_test_acc"],
            summary["mean_test_f1"],
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-mode", choices=["pair_nll", "guarded_3way"], default="pair_nll")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subjects", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--random-seed", type=int, default=2022)
    parser.add_argument("--expected-acc", type=float, default=None)
    parser.add_argument("--expected-f1", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_GUARDED_OUTPUT if args.selector_mode == "guarded_3way" else DEFAULT_OUTPUT)
    if args.expected_acc is None or args.expected_f1 is None:
        expected_acc, expected_f1 = EXPECTED_RESULTS[args.selector_mode]
        args.expected_acc = expected_acc if args.expected_acc is None else args.expected_acc
        args.expected_f1 = expected_f1 if args.expected_f1 is None else args.expected_f1

    summary = run_selector(args)
    acc_ok = abs(summary["mean_test_acc"] - args.expected_acc) <= args.tolerance
    f1_ok = abs(summary["mean_test_f1"] - args.expected_f1) <= args.tolerance
    if args.strict and not (acc_ok and f1_ok):
        raise SystemExit(
            "Expected ACC/F1 mismatch: got {:.12f}/{:.12f}, expected {:.12f}/{:.12f}".format(
                summary["mean_test_acc"],
                summary["mean_test_f1"],
                args.expected_acc,
                args.expected_f1,
            )
        )


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
