"""Independent group launches retain ordering, recovery, and output exclusion."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from test_parallel_runner import (
    HARNESS,
    fake_worker,
    read_events,
    scenario,
    stop_harness,
    wait_for_file,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("group,gpus,other", [
    ("qwen", "0,1", "llama"),
    ("llama", "2,3", "qwen"),
])
def test_single_group_dry_run_selects_pair_and_orders_baselines(tmp_path, group, gpus, other):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_parallel.py"), "--group", group,
         "--dry-run", "--output-root", str(tmp_path / "outputs")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    student, teacher = gpus.split(",")
    assert f"{group}: student GPU {student}, teacher GPU {teacher}" in result.stdout
    assert f"{other}:" not in result.stdout
    assert [line.strip().split(",")[0] for line in result.stdout.splitlines()
            if line.startswith("  ")] == [
                "train: kd", "train: abkd", "train: skd",
                "pairs: distillm2", "train: distillm2",
            ]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("group,expected", [("qwen", 0), ("llama", 0), ("all", 2)])
def test_only_selected_groups_must_have_disjoint_gpu_pairs(tmp_path, group, expected):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_parallel.py"), "--group", group,
         "--qwen-gpus", "0,1", "--llama-gpus", "0,1", "--dry-run",
         "--output-root", str(tmp_path / "outputs")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == expected, result.stderr
    if group == "all":
        assert "overlap" in result.stderr


def launch_selected(scenario, groups, *, gpu_groups=None, hold=True):
    spec = dict(
        groups={name: [asdict(stage) for stage in scenario.groups[name]] for name in groups},
        gpu_groups=gpu_groups or {name: scenario.gpu_groups[name] for name in groups},
        state_dir=str(scenario.state_dir), lock_dir=str(scenario.lock_dir),
        worker=str(scenario.worker), events=str(scenario.events), hold=hold,
    )
    harness = scenario.tmp_path / "single_group_harness.py"
    harness.write_text(textwrap.dedent(HARNESS))
    spec_path = scenario.tmp_path / f"selected_{time.time_ns()}.json"
    spec_path.write_text(json.dumps(spec, default=str))
    return subprocess.Popen(
        [sys.executable, str(harness), str(ROOT), str(spec_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )


def test_independent_single_group_processes_share_state_and_resume_in_order(scenario):
    qwen = launch_selected(scenario, ["qwen"])
    llama = None
    try:
        wait_for_file(scenario.events / (scenario.groups["qwen"][0].key + ".holding"))
        llama = launch_selected(scenario, ["llama"], hold=False)
        output = llama.communicate(timeout=30)[0]
        assert llama.returncode == 0, output
        assert qwen.poll() is None
        actual = [row for row in read_events(scenario.events)
                  if row["event"] == "start" and row["group"] == "llama"]
        assert [row["stage"] for row in actual] == [stage.key for stage in scenario.groups["llama"]]
        assert all(row["cuda"] == "2,3" for row in actual)
    finally:
        if llama is not None:
            stop_harness(llama)
        stop_harness(qwen)
    resumed = launch_selected(scenario, ["qwen"], hold=False)
    try:
        output = resumed.communicate(timeout=30)[0]
        assert resumed.returncode == 0, output
    finally:
        stop_harness(resumed)
    starts = [row for row in read_events(scenario.events)
              if row["event"] == "start" and row["group"] == "qwen"]
    assert [row["stage"] for row in starts] == [
        scenario.groups["qwen"][0].key,
        *[stage.key for stage in scenario.groups["qwen"]],
    ]
    assert Path(starts[1]["resume"]).name == "step_000001"
    assert all(row["cuda"] == "0,1" for row in starts)


@pytest.mark.parametrize("first_groups,second_groups,second_gpus", [
    (["qwen"], ["qwen"], {"qwen": ("4", "5")}),
    (["qwen"], ["qwen", "llama"], {"qwen": ("4", "5"), "llama": ("6", "7")}),
    (["qwen", "llama"], ["llama"], {"llama": ("4", "5")}),
])
def test_same_output_group_excluded_even_with_different_gpus(
    scenario, first_groups, second_groups, second_gpus,
):
    first = launch_selected(scenario, first_groups)
    second = None
    try:
        for group in first_groups:
            wait_for_file(scenario.events / (scenario.groups[group][0].key + ".holding"))
        before = read_events(scenario.events)
        second = launch_selected(scenario, second_groups, gpu_groups=second_gpus, hold=False)
        output = second.communicate(timeout=30)[0]
        assert second.returncode == 1, output
        assert "Another live launcher/worker holds" in output
        assert len(read_events(scenario.events)) == len(before)
        assert first.poll() is None
    finally:
        if second is not None:
            stop_harness(second)
        stop_harness(first)
