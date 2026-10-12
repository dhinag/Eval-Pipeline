from pathlib import Path

from goldenset.cli import main


def test_generate_command_writes_the_dataset(tmp_path: Path) -> None:
    out = tmp_path / "data"
    code = main(["generate", "--n", "60", "--seed", "1", "--out", str(out)])
    assert code == 0
    assert (out / "traces.jsonl").exists()
    assert (out / "ground_truth.jsonl").exists()
    assert (out / "manifest.json").exists()
