# Usage notes

Configuration, model providers and the interactive gates. The command reference is in the [README](../README.md#commands).

## Configuration files

Every flag you'd otherwise repeat can live in JSON config, layered as `~/.parsec.json` (user) < `./.parsec.json` (project) < explicit flags. Keys mirror the CLI option names; string values expand `${ENV_VARS}`:

```json
{
  "data_dir": "~/.parsec/data",
  "search_provider": "searxng",
  "searxng_url": "http://localhost:8080",
  "contact": "you@example.com",
  "max_usd": 2.0
}
```

Unknown keys are rejected loudly (typo protection). The interactive banner shows which config files were loaded.

## Claude via Amazon Bedrock

If your org reaches Claude through Bedrock instead of an Anthropic API key, install the signing dependencies and point the adapter at your AWS profile:

```sh
uv sync --extra bedrock              # or: uv tool install --editable ".[bedrock]"
okta-awscli --profile okta           # or aws sso login, etc. — anything that writes
                                     # credentials into the standard AWS chain
```

Then set it once in config (or pass `--adapter bedrock --aws-region … --aws-profile …` per run):

```json
{
  "adapter": "bedrock",
  "aws_region": "us-east-1",
  "aws_profile": "okta"
}
```

Auth is the standard AWS credential chain — env vars, then `~/.aws/credentials` — which is exactly where okta-awscli drops its temporary STS credentials, so re-running `okta-awscli` when they expire is the whole refresh story. Model IDs are prefixed automatically (`claude-opus-5` → `anthropic.claude-opus-5`); no `ANTHROPIC_API_KEY` is needed. Recording, replay, and budgets behave identically to the first-party adapter.

## Editing the research brief in $EDITOR

With `ask --brief-gate`, the run pauses at the proposed research brief. Besides typing `approve` or feedback on stdin, type `edit` to open the proposed brief (scope, effort, subquestions) in `$EDITOR` — the text you save is submitted as the brief-edit steering message, so the gated session still replays byte-identically.

## Gates

Two decision points can pause a live run for your input, both built on the same replay-safe primitive (a journaled proposal blocks for one steering reply, recorded gate-tagged and turn-indexed, so gated sessions replay byte-identically):

- `--brief-gate` pauses at the proposed research brief — `approve` dispatches, `edit` opens it in `$EDITOR`, anything else requests changes.
- `--cost-gate 0.5` pauses **once** when spend crosses that fraction of the USD cap at a dispatch boundary — `approve` continues; any other reply stops dispatching, remaining subquestions resolve as declined, and the writer wraps up with what was gathered.
