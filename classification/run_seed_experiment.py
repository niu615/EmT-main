import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


CLASSIFICATION_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = CLASSIFICATION_DIR / "main-SEED.py"
RESULT_ROOT = CLASSIFICATION_DIR.parent / "results" / "SEED"
WINDOWS_GIT = Path(r"C:\Program Files\Git\cmd\git.exe")

PRESETS = {
    "paperfix": {
        "experiment_name": "00_seed_paperfix",
        "use_simam": 0,
        "fusion_mode": "mean",
        "pooling_mode": "mean",
        "sta_kernels": "3",
        "adj_sparsity_weight": 0.0,
        "adj_diversity_weight": 0.0,
    },
    "sdrmpg": {
        "experiment_name": "01_seed_sdrmpg",
        "use_simam": 0,
        "fusion_mode": "adaptive",
        "pooling_mode": "mean",
        "sta_kernels": "3",
        "adj_sparsity_weight": 1e-4,
        "adj_diversity_weight": 1e-3,
    },
    "sdrmpg_mssta": {
        "experiment_name": "02_seed_sdrmpg_mssta",
        "use_simam": 0,
        "fusion_mode": "adaptive",
        "pooling_mode": "attention",
        "sta_kernels": "3,5",
        "adj_sparsity_weight": 1e-4,
        "adj_diversity_weight": 1e-3,
    },
    "mssta_fallback": {
        "experiment_name": "02_seed_mssta_fallback",
        "use_simam": 0,
        "fusion_mode": "mean",
        "pooling_mode": "attention",
        "sta_kernels": "3,5",
        "adj_sparsity_weight": 0.0,
        "adj_diversity_weight": 0.0,
    },
    "sdrmpg_rsc": {
        "experiment_name": "07_seed_sdrmpg_rsc",
        "use_simam": 0,
        "fusion_mode": "adaptive",
        "pooling_mode": "mean",
        "sta_kernels": "3",
        "adj_sparsity_weight": 1e-4,
        "adj_diversity_weight": 1e-3,
        "train_mode": "rsc",
        "rsc_start_epoch": 2,
        "rsc_drop_ratio": 0.3,
        "rdrop_weight": 0.0,
        "grad_accum_steps": 1,
        "batch_size": 64,
    },
    "sdrmpg_rsc_rdrop": {
        "experiment_name": "08_seed_sdrmpg_rsc_rdrop",
        "use_simam": 0,
        "fusion_mode": "adaptive",
        "pooling_mode": "mean",
        "sta_kernels": "3",
        "adj_sparsity_weight": 1e-4,
        "adj_diversity_weight": 1e-3,
        "train_mode": "rsc",
        "rsc_start_epoch": 2,
        "rsc_drop_ratio": 0.3,
        "rdrop_weight": 0.5,
        "grad_accum_steps": 2,
        "batch_size": 32,
    },
    "sdrmpg_rdrop": {
        "experiment_name": "08_seed_sdrmpg_rdrop",
        "use_simam": 0,
        "fusion_mode": "adaptive",
        "pooling_mode": "mean",
        "sta_kernels": "3",
        "adj_sparsity_weight": 1e-4,
        "adj_diversity_weight": 1e-3,
        "train_mode": "rdrop",
        "rsc_start_epoch": 2,
        "rsc_drop_ratio": 0.3,
        "rdrop_weight": 0.5,
        "grad_accum_steps": 2,
        "batch_size": 32,
    },
}


def get_result_dir(preset_name):
    return RESULT_ROOT / PRESETS[preset_name]["experiment_name"]


def build_env():
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def resolve_git():
    git_from_path = shutil.which("git")
    if git_from_path:
        return git_from_path
    if WINDOWS_GIT.exists():
        return str(WINDOWS_GIT)
    return None


def detect_current_branch():
    git_executable = resolve_git()
    if git_executable is None:
        return ""
    completed = subprocess.run(
        [git_executable, "-C", str(CLASSIFICATION_DIR.parent), "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def build_main_command(args):
    preset = PRESETS[args.preset]
    command = [
        sys.executable,
        "-u",
        str(MAIN_SCRIPT),
        "--experiment-name",
        preset["experiment_name"],
        "--use-simam",
        str(preset["use_simam"]),
        "--fusion-mode",
        preset["fusion_mode"],
        "--pooling-mode",
        preset["pooling_mode"],
        "--sta-kernels",
        preset["sta_kernels"],
        "--adj-sparsity-weight",
        str(preset["adj_sparsity_weight"]),
        "--adj-diversity-weight",
        str(preset["adj_diversity_weight"]),
        "--train-mode",
        preset.get("train_mode", "erm"),
        "--rsc-start-epoch",
        str(preset.get("rsc_start_epoch", 2)),
        "--rsc-drop-ratio",
        str(preset.get("rsc_drop_ratio", 0.3)),
        "--rdrop-weight",
        str(preset.get("rdrop_weight", 0.0)),
        "--grad-accum-steps",
        str(preset.get("grad_accum_steps", 1)),
        "--batch-size",
        str(preset.get("batch_size", 64)),
    ]
    if args.gpu is not None:
        command.extend(["--gpu", args.gpu])
    if args.data_exist is not None:
        command.extend(["--data-exist", str(args.data_exist)])
    if args.extra:
        command.extend(args.extra)
    return command


def build_worker_command(args):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--preset",
        args.preset,
        "--worker",
    ]
    if args.gpu is not None:
        command.extend(["--gpu", args.gpu])
    if args.data_exist is not None:
        command.extend(["--data-exist", str(args.data_exist)])
    if args.extra:
        command.extend(args.extra)
    return command


def write_process_metadata(result_dir, payload):
    result_dir.mkdir(parents=True, exist_ok=True)
    process_file = result_dir / "process.json"
    with process_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def clean_result_dir(result_dir):
    resolved = result_dir.resolve()
    allowed_root = RESULT_ROOT.resolve()
    if allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to delete path outside result root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


@contextmanager
def keep_system_awake():
    if os.name != "nt":
        yield
        return

    kernel32 = ctypes.windll.kernel32
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def execute_worker(args):
    result_dir = get_result_dir(args.preset)
    env = build_env()
    cmd = build_main_command(args)
    metadata = {
        "preset": args.preset,
        "mode": "worker",
        "launcher_pid": os.getppid(),
        "worker_pid": os.getpid(),
        "child_pid": None,
        "branch": detect_current_branch(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "starting",
        "stdout": str(result_dir / "console.log"),
        "stderr": str(result_dir / "console.err.log"),
        "command": cmd,
    }
    write_process_metadata(result_dir, metadata)
    print("Running:", " ".join(cmd), flush=True)

    with keep_system_awake():
        process = subprocess.Popen(cmd, cwd=str(CLASSIFICATION_DIR), env=env)
        metadata["child_pid"] = process.pid
        metadata["status"] = "running"
        write_process_metadata(result_dir, metadata)
        return_code = process.wait()

    metadata["status"] = "finished" if return_code == 0 else "failed"
    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["return_code"] = return_code
    write_process_metadata(result_dir, metadata)
    raise SystemExit(return_code)


def launch_detached_worker(args):
    result_dir = get_result_dir(args.preset)
    result_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = result_dir / "console.log"
    stderr_path = result_dir / "console.err.log"
    cmd = build_worker_command(args)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(CLASSIFICATION_DIR),
            env=build_env(),
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creationflags,
        )

    write_process_metadata(
        result_dir,
        {
            "preset": args.preset,
            "mode": "detached-launcher",
            "launcher_pid": os.getpid(),
            "worker_pid": process.pid,
            "child_pid": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "launching",
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "command": cmd,
        },
    )
    print(f"Detached worker started with PID {process.pid}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--data-exist", type=int, choices=[0, 1], default=None)
    parser.add_argument("--clean", action="store_true", help="Delete the preset result directory before launch")
    parser.add_argument("--detach", action="store_true", help="Run in a detached background worker")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    result_dir = get_result_dir(parsed.preset)
    if parsed.clean:
        clean_result_dir(result_dir)

    if parsed.worker:
        execute_worker(parsed)
    elif parsed.detach:
        launch_detached_worker(parsed)
    else:
        execute_worker(parsed)
