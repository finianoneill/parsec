from __future__ import annotations

import json

import pytest

import parsec.cli as cli
from parsec.config import RunConfig
from parsec.gateway.bedrock_adapter import bedrock_model_id


def test_bedrock_model_id_prefixing():
    assert bedrock_model_id("claude-opus-5") == "anthropic.claude-opus-5"
    assert bedrock_model_id("anthropic.claude-opus-5") == "anthropic.claude-opus-5"


def _config(**over) -> RunConfig:
    return RunConfig(session_id="s", query="q", adapter="bedrock", **over)


def test_make_adapter_bedrock_requires_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(SystemExit, match="region"):
        cli.make_adapter(_config())


def test_make_adapter_bedrock_builds_client(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("AWS_REGION", raising=False)
    adapter = cli.make_adapter(_config(aws_region="us-east-1", aws_profile="okta"))
    assert type(adapter).__name__ == "BedrockAdapter"


def test_bedrock_settings_flow_from_config_file(tmp_path):
    from parsec.user_config import apply_config, load_user_config

    cfg_file = tmp_path / "proj.json"
    cfg_file.write_text(
        json.dumps({"adapter": "bedrock", "aws_region": "us-west-2", "aws_profile": "okta"}),
        encoding="utf-8",
    )
    merged, _ = load_user_config(user_path=tmp_path / "absent.json", project_path=cfg_file)
    parser = cli.build_parser()
    apply_config(parser, merged)
    args = parser.parse_args(["ask", "q"])
    assert args.adapter == "bedrock"
    assert args.aws_region == "us-west-2"
    assert args.aws_profile == "okta"
