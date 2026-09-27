from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPS_MODULE_PATH = REPO_ROOT / "run_exps.py"


def load_run_exps_module():
    spec = importlib.util.spec_from_file_location("run_exps_module", RUN_EXPS_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_commands_matches_fixed_fiqa_matrix() -> None:
    run_exps = load_run_exps_module()
    args = run_exps.build_argument_parser().parse_args([])

    commands = run_exps.build_commands(args)

    assert len(commands) == 6
    assert all("--device" in command and "cpu" in command for command in commands)
    assert all("--dataset" in command and "fiqa" in command for command in commands)
    assert all("--batch-size" in command and "256" in command for command in commands)
    assert all("--no-download" in command for command in commands)
    assert not any("--allow-download" in command for command in commands)
    assert not any("--dataset-dir" in command for command in commands)

    rerank_pairs = [
        (command[command.index("--max-docs") + 1], command[command.index("--rerank") + 1])
        for command in commands
    ]
    assert rerank_pairs == [
        ("1000", "none"),
        ("1000", "ppr"),
        ("1000", "ppr"),
        ("5000", "none"),
        ("5000", "ppr"),
        ("5000", "ppr"),
    ]


def test_build_commands_includes_dataset_dir_when_provided(tmp_path: Path) -> None:
    run_exps = load_run_exps_module()
    args = run_exps.build_argument_parser().parse_args(
        ["--dataset-dir", str(tmp_path), "--allow-download", "--batch-size", "64"]
    )

    commands = run_exps.build_commands(args)

    assert all("--dataset-dir" in command for command in commands)
    assert all(str(tmp_path) in command for command in commands)
    assert all("--batch-size" in command and "64" in command for command in commands)
    assert all("--allow-download" in command for command in commands)
    assert not any("--no-download" in command for command in commands)
