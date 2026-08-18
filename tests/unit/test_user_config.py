from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import parsec.cli as cli
from parsec.user_config import apply_config, load_user_config


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_project_config_overrides_user(tmp_path):
    user = _write(tmp_path / "user.json", {"model": "user-model", "max_usd": 1.0})
    project = _write(tmp_path / "proj.json", {"model": "proj-model"})
    merged, sources = load_user_config(user_path=user, project_path=project)
    assert merged["model"] == "proj-model"
    assert merged["max_usd"] == 1.0
    assert sources == [user, project]


def test_env_expansion_and_path_coercion(tmp_path, monkeypatch):
    monkeypatch.setenv("PARSEC_TEST_HOME", "/srv/parsec")
    user = _write(
        tmp_path / "user.json",
        {"data_dir": "${PARSEC_TEST_HOME}/data", "searxng_url": "${PARSEC_TEST_HOME}"},
    )
    merged, _ = load_user_config(user_path=user, project_path=tmp_path / "absent.json")
    assert merged["data_dir"] == Path("/srv/parsec/data")
    assert merged["searxng_url"] == "/srv/parsec"


def test_unknown_key_is_a_typo_error(tmp_path):
    user = _write(tmp_path / "user.json", {"serch_provider": "brave"})
    with pytest.raises(SystemExit, match="serch_provider"):
        load_user_config(user_path=user, project_path=tmp_path / "absent.json")


def test_config_becomes_subcommand_default_but_flags_win(tmp_path):
    parser = cli.build_parser()
    apply_config(parser, {"data_dir": Path("/cfg/data"), "max_usd": 9.5})

    args = parser.parse_args(["ask", "q"])
    assert args.data_dir == Path("/cfg/data")
    assert args.max_usd == 9.5

    args = parser.parse_args(["ask", "q", "--data-dir", "elsewhere", "--max-usd", "2"])
    assert args.data_dir == Path("elsewhere")
    assert args.max_usd == 2.0

    # subcommands that never declared the option are untouched
    args = parser.parse_args(["replay", "some-session"])
    assert args.data_dir == Path("/cfg/data")


def test_compose_in_editor_roundtrip(monkeypatch):
    monkeypatch.setenv("EDITOR", "sh -c 'printf \"edited text\" > \"$0\"'")
    monkeypatch.delenv("VISUAL", raising=False)
    assert cli.compose_in_editor("seed") == "edited text"


def test_compose_in_editor_empty_means_abort(monkeypatch):
    monkeypatch.setenv("EDITOR", "sh -c 'printf \"\" > \"$0\"'")
    monkeypatch.delenv("VISUAL", raising=False)
    assert cli.compose_in_editor("seed") is None


class _FakeLoop:
    def __init__(self, brief=None):
        self.current_brief = brief
        self.steered: list[str] = []

    def steer(self, text: str) -> None:
        self.steered.append(text)


def test_steer_line_edit_opens_editor_at_gate(monkeypatch):
    from parsec.loop.agent import Brief

    monkeypatch.setenv("EDITOR", "sh -c 'printf \"narrow the scope\" > \"$0\"'")
    monkeypatch.delenv("VISUAL", raising=False)
    loop = _FakeLoop(brief=Brief(scope="s", effort="quick", questions=["q1"]))
    cli.handle_steer_line(loop, "edit")
    assert loop.steered == ["narrow the scope"]


def test_steer_line_edit_outside_gate_is_plain_steering():
    loop = _FakeLoop(brief=None)
    cli.handle_steer_line(loop, "edit")
    assert loop.steered == ["edit"]


def test_repl_edit_composes_a_query(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EDITOR", "sh -c 'printf \"long question\" > \"$0\"'")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("/edit\n/exit\n"))
    assert cli.main(["--data-dir", str(tmp_path / "data")]) == cli.EXIT_OK
    out = capsys.readouterr().out
    # composed, then stopped at the key guard (no key in this test env)
    assert "long question" in out
    assert "no ANTHROPIC_API_KEY" in out
