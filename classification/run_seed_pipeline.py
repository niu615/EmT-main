import argparse
import json
from datetime import datetime
from pathlib import Path

from manage_seed_experiment import (
    EXPERIMENTS,
    PROJECT_ROOT,
    RESULT_ROOT,
    current_branch,
    ensure_clean_tree,
    finalize_experiment,
    prepare_experiment,
    run_experiment,
)


PIPELINE_STATE = RESULT_ROOT / "seed_pipeline_state.json"


def write_state(payload):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with PIPELINE_STATE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def update_state(stage, preset=None, extra=None):
    payload = {
        "stage": stage,
        "preset": preset,
        "current_branch": current_branch(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    write_state(payload)


def run_single_experiment(preset, gpu, data_exist):
    prepare_experiment(preset)
    update_state("running", preset=preset, extra={"result_dir": str(EXPERIMENTS[preset]["result_dir"])})
    run_experiment(
        preset_name=preset,
        gpu=gpu,
        data_exist=data_exist,
        clean=True,
        detach=False,
        extra=None,
    )
    update_state("finalizing", preset=preset)
    kept = finalize_experiment(preset, delete_on_drop=True)
    update_state(
        "completed",
        preset=preset,
        extra={
            "kept": kept,
            "result_dir": str(EXPERIMENTS[preset]["result_dir"]),
        },
    )
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data-exist", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    if current_branch() != "tmp/seed-paperfix":
        raise SystemExit("Pipeline must be started from branch 'tmp/seed-paperfix'.")
    ensure_clean_tree("start seed pipeline")

    update_state("starting", extra={"project_root": str(PROJECT_ROOT)})
    sdrmpg_kept = run_single_experiment("sdrmpg", gpu=args.gpu, data_exist=args.data_exist)

    if sdrmpg_kept:
        next_preset = "sdrmpg_mssta"
    else:
        next_preset = "mssta_fallback"

    update_state("branching", extra={"next_preset": next_preset, "sdrmpg_kept": sdrmpg_kept})
    next_kept = run_single_experiment(next_preset, gpu=args.gpu, data_exist=args.data_exist)
    update_state(
        "done",
        preset=next_preset,
        extra={
            "sdrmpg_kept": sdrmpg_kept,
            "final_kept": next_kept,
            "final_result_dir": str(EXPERIMENTS[next_preset]["result_dir"]),
        },
    )
