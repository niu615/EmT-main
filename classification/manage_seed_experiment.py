import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from compare_seed_results import compare_summaries, print_decision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "results" / "SEED"
RUNNER = CLASSIFICATION_DIR / "run_seed_experiment.py"
WINDOWS_GIT = Path(r"C:\Program Files\Git\cmd\git.exe")

EXPERIMENTS = {
    "paperfix": {
        "branch": "tmp/seed-paperfix",
        "parent_branch": None,
        "fallback_branch": None,
        "result_dir": RESULT_ROOT / "00_seed_paperfix",
        "compare_against": None,
        "preserve": True,
    },
    "sdrmpg": {
        "branch": "feat/seed_sdrmpg",
        "parent_branch": "tmp/seed-paperfix",
        "fallback_branch": "tmp/seed-paperfix",
        "result_dir": RESULT_ROOT / "01_seed_sdrmpg",
        "compare_against": "paperfix",
        "preserve": False,
    },
    "sdrmpg_mssta": {
        "branch": "feat/seed_sdrmpg_mssta",
        "parent_branch": "feat/seed_sdrmpg",
        "fallback_branch": "feat/seed_sdrmpg",
        "result_dir": RESULT_ROOT / "02_seed_sdrmpg_mssta",
        "compare_against": "sdrmpg",
        "preserve": False,
    },
    "mssta_fallback": {
        "branch": "feat/seed_mssta_fallback",
        "parent_branch": "tmp/seed-paperfix",
        "fallback_branch": "tmp/seed-paperfix",
        "result_dir": RESULT_ROOT / "02_seed_mssta_fallback",
        "compare_against": "paperfix",
        "preserve": False,
    },
    "sdrmpg_rsc": {
        "branch": "feat/seed_sdrmpg_rsc",
        "parent_branch": "feat/seed_sdrmpg",
        "fallback_branch": "feat/seed_sdrmpg",
        "result_dir": RESULT_ROOT / "07_seed_sdrmpg_rsc",
        "compare_against": "sdrmpg",
        "preserve": False,
    },
    "sdrmpg_rsc_rdrop": {
        "branch": "feat/seed_sdrmpg_rsc_rdrop",
        "parent_branch": "feat/seed_sdrmpg_rsc",
        "fallback_branch": "feat/seed_sdrmpg_rsc",
        "result_dir": RESULT_ROOT / "08_seed_sdrmpg_rsc_rdrop",
        "compare_against": "sdrmpg_rsc",
        "preserve": False,
    },
    "sdrmpg_rdrop": {
        "branch": "feat/seed_sdrmpg_rdrop",
        "parent_branch": "feat/seed_sdrmpg",
        "fallback_branch": "feat/seed_sdrmpg",
        "result_dir": RESULT_ROOT / "08_seed_sdrmpg_rdrop",
        "compare_against": "sdrmpg",
        "preserve": False,
    },
}


def resolve_git():
    git_from_path = shutil.which("git")
    if git_from_path:
        return git_from_path
    if WINDOWS_GIT.exists():
        return str(WINDOWS_GIT)
    raise FileNotFoundError("Git executable not found. Install Git or add it to PATH.")


def run_git(args, check=True):
    command = [resolve_git(), "-C", str(PROJECT_ROOT), *args]
    return subprocess.run(command, check=check, capture_output=True, text=True)


def current_branch():
    return run_git(["branch", "--show-current"]).stdout.strip()


def working_tree_dirty():
    return bool(run_git(["status", "--short"]).stdout.strip())


def ensure_clean_tree(action_name):
    if working_tree_dirty():
        raise SystemExit(
            f"Refusing to {action_name}: git working tree is dirty. "
            "Create a checkpoint commit on tmp/seed-paperfix first."
        )


def branch_exists(branch_name):
    completed = run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], check=False)
    return completed.returncode == 0


def switch_branch(branch_name):
    run_git(["switch", branch_name])


def create_branch(branch_name, parent_branch):
    run_git(["switch", parent_branch])
    run_git(["switch", "-c", branch_name])


def delete_branch(branch_name):
    if branch_exists(branch_name):
        run_git(["branch", "-D", branch_name])


def ensure_safe_result_dir(result_dir):
    resolved = result_dir.resolve()
    allowed_root = RESULT_ROOT.resolve()
    if allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to operate outside result root: {resolved}")


def remove_result_dir(result_dir):
    ensure_safe_result_dir(result_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)


def experiment_result_exists(preset_name):
    return EXPERIMENTS[preset_name]["result_dir"].joinpath("summary.json").exists()


def write_decision_file(preset_name, decision):
    result_dir = EXPERIMENTS[preset_name]["result_dir"]
    ensure_safe_result_dir(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "decision.json").open("w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2, ensure_ascii=False)


def show_status():
    print(f"repo:           {PROJECT_ROOT}")
    print(f"current branch: {current_branch()}")
    print(f"worktree:       {'dirty' if working_tree_dirty() else 'clean'}")
    print("")
    for name, config in EXPERIMENTS.items():
        compare_against = config["compare_against"] or "-"
        result_state = "ready" if experiment_result_exists(name) else "missing"
        print(f"[{name}]")
        print(f"  branch:       {config['branch']}")
        print(f"  parent:       {config['parent_branch'] or '-'}")
        print(f"  compare:      {compare_against}")
        print(f"  result dir:   {config['result_dir']}")
        print(f"  result state: {result_state}")
        print("")


def prepare_experiment(preset_name):
    config = EXPERIMENTS[preset_name]
    if config["parent_branch"] is None:
        raise SystemExit(f"{preset_name} is the preserved baseline and does not need branch preparation.")
    ensure_clean_tree(f"prepare {preset_name}")
    if branch_exists(config["branch"]):
        switch_branch(config["branch"])
    else:
        create_branch(config["branch"], config["parent_branch"])
    print(f"Prepared branch: {config['branch']}")
    print(f"Expected result dir: {config['result_dir']}")


def run_experiment(preset_name, gpu=None, data_exist=None, clean=False, detach=False, extra=None):
    expected_branch = EXPERIMENTS[preset_name]["branch"]
    actual_branch = current_branch()
    if actual_branch != expected_branch:
        raise SystemExit(
            f"Current branch is {actual_branch!r}, but preset {preset_name!r} must run on {expected_branch!r}."
        )

    command = [sys.executable, str(RUNNER), "--preset", preset_name]
    if gpu is not None:
        command.extend(["--gpu", gpu])
    if data_exist is not None:
        command.extend(["--data-exist", str(data_exist)])
    if clean:
        command.append("--clean")
    if detach:
        command.append("--detach")
    if extra:
        command.extend(extra)

    subprocess.run(command, cwd=str(CLASSIFICATION_DIR), check=True)


def finalize_experiment(preset_name, delete_on_drop=True):
    config = EXPERIMENTS[preset_name]
    if config["compare_against"] is None:
        raise SystemExit(f"{preset_name} is the baseline and is always preserved.")

    baseline_dir = EXPERIMENTS[config["compare_against"]]["result_dir"]
    candidate_dir = config["result_dir"]
    if not candidate_dir.joinpath("summary.json").exists():
        raise SystemExit(f"Candidate summary not found under {candidate_dir}")

    decision = compare_summaries(baseline_dir, candidate_dir)
    print_decision(decision)

    if decision["keep"]:
        write_decision_file(preset_name, decision)
        print(f"Kept branch {config['branch']} and result dir {candidate_dir}")
        return True

    if not delete_on_drop:
        print("Candidate is worse than baseline, but deletion is disabled.")
        return False

    active_branch = current_branch()
    if active_branch == config["branch"]:
        ensure_clean_tree(f"switch away from dropped experiment {preset_name}")
        switch_branch(config["fallback_branch"])
    remove_result_dir(candidate_dir)
    delete_branch(config["branch"])
    print(f"Dropped branch {config['branch']} and removed result dir {candidate_dir}")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--preset", choices=sorted(EXPERIMENTS.keys()), required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--preset", choices=sorted(EXPERIMENTS.keys()), required=True)
    run_parser.add_argument("--gpu", type=str, default=None)
    run_parser.add_argument("--data-exist", type=int, choices=[0, 1], default=None)
    run_parser.add_argument("--clean", action="store_true")
    run_parser.add_argument("--detach", action="store_true")
    run_parser.add_argument("extra", nargs=argparse.REMAINDER)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--preset", choices=sorted(EXPERIMENTS.keys()), required=True)
    finalize_parser.add_argument("--keep-drop-artifacts", action="store_true")

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "prepare":
        prepare_experiment(args.preset)
    elif args.command == "run":
        run_experiment(
            preset_name=args.preset,
            gpu=args.gpu,
            data_exist=args.data_exist,
            clean=args.clean,
            detach=args.detach,
            extra=args.extra,
        )
    elif args.command == "finalize":
        finalize_experiment(args.preset, delete_on_drop=not args.keep_drop_artifacts)
