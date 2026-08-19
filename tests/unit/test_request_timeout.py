"""Per-request model-call timeout: configurable, wired through make_adapter,
and never accidentally disabling the SDK's own default."""

from __future__ import annotations

import pytest

import parsec.cli as cli
from parsec.config import RunConfig
from parsec.gateway.anthropic_adapter import AnthropicAdapter


def test_timeout_set_on_client():
    pytest.importorskip("anthropic")
    adapter = AnthropicAdapter(api_key="k", timeout=120.0)
    assert adapter._client.timeout == 120.0


def test_timeout_unset_keeps_sdk_default():
    anthropic = pytest.importorskip("anthropic")
    adapter = AnthropicAdapter(api_key="k")
    # must be the SDK default, NOT None (None disables timeouts entirely)
    assert adapter._client.timeout == anthropic.DEFAULT_TIMEOUT


def test_make_adapter_passes_config_timeout(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    config = RunConfig(session_id="s", query="q", adapter="anthropic", request_timeout_s=90.0)
    adapter = cli.make_adapter(config)
    assert adapter._client.timeout == 90.0


def test_cli_flag_parses_into_config():
    args = cli.build_parser().parse_args(["ask", "q", "--request-timeout", "45"])
    assert args.request_timeout == 45.0
    assert cli.build_parser().parse_args(["ask", "q"]).request_timeout is None
