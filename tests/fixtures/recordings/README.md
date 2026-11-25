# Frozen recordings

Four sessions recorded by the code at commit `366f1f6`, the last commit before
the remediation work, stored as text corpora (`parsec eval export-case`).
`tests/integration/test_frozen_recordings.py` rebuilds and replays each one on
every run: the answer bytes and the event projection must match what that code
produced.

| Recording | What it exercises |
|---|---|
| `demo` | The built-in offline demo |
| `full` | Rejected premise, `transform_note` premise, near-duplicate corroboration, `deduces` and `temporal` findings, a recorded conflict, a corrected `submit_report`, a writer repair round, and a gap-fill subagent with rewrite |
| `parallel` | Three concurrent subagents joining in the reverse of dispatch order |
| `refresh` | A parent session and its `parsec refresh` after the volatile page changed |

Each directory holds `corpus/` (the export), `aux/` (the search fixtures the
run read, restored next to the data dir on replay) and `recording.json` (the
session ids and their recorded statuses).

**Do not re-record these.** They are only useful because the recording side
never changes. `record_fixtures.py` is kept so the scripts and the fake web
can be audited; a new scenario gets a new recording alongside these. If a
replay fails, a behaviour old recordings depend on was changed in place rather
than behind a behaviour version, and the code is what needs fixing.
