from __future__ import annotations

from pathlib import Path

import pytest

from firstmove_eval.config import OpenAIChatSettings, RunConfig, load_config
from firstmove_eval.errors import ConfigurationError


def test_load_config_resolves_relative_paths_and_forbids_unknown_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.jsonl").write_text("", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 1
run:
  output_dir: output
dataset:
  path: data.jsonl
model:
  kind: mock
  unknown: rejected
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unknown"):
        load_config(config_path)

    config_path.write_text(
        """
schema_version: 1
run:
  output_dir: output
dataset:
  path: data.jsonl
model:
  kind: mock
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.dataset.path == (tmp_path / "data.jsonl").resolve()
    assert config.run.output_dir == (tmp_path / "output").resolve()


def test_openai_base_url_must_end_in_v1() -> None:
    with pytest.raises(ValueError, match="end with /v1"):
        OpenAIChatSettings(base_url="https://example.test", model="model")


def test_openai_base_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        OpenAIChatSettings(
            base_url="https://user:secret@example.test/v1",
            model="model",
        )


def test_safe_manifest_does_not_resolve_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SECRET_API_KEY", "super-secret-value")
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "dataset": {"path": str(tmp_path / "data.jsonl")},
            "model": {
                "kind": "openai_chat",
                "base_url": "https://example.test/v1",
                "api_key_env": "SECRET_API_KEY",
                "model": "model",
            },
        }
    )
    manifest = str(config.safe_manifest_dict())
    assert "SECRET_API_KEY" in manifest
    assert "super-secret-value" not in manifest
