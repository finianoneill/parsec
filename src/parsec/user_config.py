"""Layered CLI configuration: ~/.parsec.json < ./.parsec.json < explicit flags.

Config keys mirror CLI option dests (snake_case). Values land as parser
defaults, so anything passed explicitly on the command line still wins, and
every subcommand that declares the option picks the value up.

String values get ${ENV_VAR} expansion; path keys also expand `~`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

USER_CONFIG = Path("~/.parsec.json")
PROJECT_CONFIG = Path(".parsec.json")

_PATH_KEYS = {"data_dir", "search_fixtures", "calibration"}
_ALLOWED = _PATH_KEYS | {
    "adapter",
    "aws_region",
    "aws_profile",
    "model",
    "cache_mode",
    "search_provider",
    "searxng_url",
    "contact",
    "nli_checker",
    "learned_reliability",
    "judge_model",
    "max_usd",
    "max_tokens",
    "max_seconds",
    "max_turns",
    "max_gap_rounds",
    "max_coverage_gap_rounds",
    "max_turns_per_subagent",
    "request_timeout",
    "parallel",
    "brief_gate",
    "epsilon",
}


def load_user_config(
    user_path: Path | None = None, project_path: Path | None = None
) -> tuple[dict, list[Path]]:
    """Merged config dict plus the files it came from (user first)."""
    merged: dict = {}
    sources: list[Path] = []
    for path in (
        (user_path or USER_CONFIG).expanduser(),
        (project_path or PROJECT_CONFIG),
    ):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid JSON ({exc})")
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}: expected a JSON object")
        unknown = sorted(set(raw) - _ALLOWED)
        if unknown:
            raise SystemExit(
                f"{path}: unknown config keys: {', '.join(unknown)}\n"
                f"known keys: {', '.join(sorted(_ALLOWED))}"
            )
        merged.update(raw)
        sources.append(path)

    for key, value in merged.items():
        if isinstance(value, str):
            value = os.path.expandvars(value)
        if key in _PATH_KEYS:
            value = Path(value).expanduser()
        merged[key] = value
    return merged, sources


def apply_config(parser: argparse.ArgumentParser, config: dict) -> None:
    """Install config values as defaults on the parser and every subparser
    that declares a matching option (set_defaults rewrites action defaults,
    so explicit command-line values still take precedence)."""
    if not config:
        return
    dests = {a.dest for a in parser._actions}
    relevant = {k: v for k, v in config.items() if k in dests}
    if relevant:
        parser.set_defaults(**relevant)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                apply_config(sub, config)
