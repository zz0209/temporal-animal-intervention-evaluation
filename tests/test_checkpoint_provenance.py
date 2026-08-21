from pathlib import Path

from animal_intervention.experiments.historical_set_planning import _checkpoint_key


def test_checkpoint_key_changes_with_config_or_source(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    source = tmp_path / "experiment.py"
    config.write_text("value: 1\n", encoding="utf-8")
    source.write_text("VERSION = 1\n", encoding="utf-8")
    original = _checkpoint_key(["task"], config, source)

    config.write_text("value: 2\n", encoding="utf-8")
    changed_config = _checkpoint_key(["task"], config, source)
    source.write_text("VERSION = 2\n", encoding="utf-8")
    changed_source = _checkpoint_key(["task"], config, source)

    assert original != changed_config
    assert changed_config != changed_source
