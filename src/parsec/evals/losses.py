"""Loss taxonomy (M15.2): WHY a case lost points, named mechanically.

Scores say a nugget went missing; they do not say whether the gold page
never surfaced in search, surfaced and went unfetched, fetched and would
not extract, extracted and no premise was recorded, was recorded and the
writer never cited it, or was cited in words the gold pattern does not
match. Each of those is a different fix in a different module, and every
one of them is already journaled — so the diagnosis is a pure function of
the recording plus the case's gold. No model, no network, deterministic.

The borrowed idea is Google's agent-evaluation "loss clusters": failures
classified into a fixed pattern taxonomy, then grouped, so triage starts
from the pattern histogram instead of from per-case scores. Google's
patterns are assigned by an autorater over a conversation trace; parsec's
are assigned by walking the event log and the evidence DAG back from each
miss (T2: verification is structural first), which is why they can gate.

Three kinds of subject, one ladder each:

  gold items (must_find entries and nuggets) — the miss is localized to the
    deepest stage the evidence reached: unreachable (its gold docs were
    never read; the per-doc pattern says why) < unrecorded (text sits in a
    fetched doc's spans, no premise) < premise rejected (record_premises
    refused it, reason attached) < uncited (premise exists, no claim) <
    unmatched claim (a claim rests on the premise but the pattern misses
    its wording). Contradictions are their own pattern. A miss that cannot
    be localized says so (declare gold_docs) and a miss whose gold docs
    were all read but contain no match points at the case, not the agent.
  documents (gold_docs and distractor_docs) — the retrieval stage per URL:
    not surfaced by any search, surfaced but not fetched, fetch failed
    (typed outcome or HTTP error), fetched but unminable, read but no
    premise recorded; a planted distractor is flagged when consulted and
    worse when a claim rests on it.
  the run — structural violations per claim, unsupported claims, unresolved
    citations after repair, coverage rows that did not end answered (with
    their recorded reason), and harness incidents: budget halts, subagent
    deaths, model failures and journaled retries, tool errors, compactions,
    premise rejections per stream.

Losses ride in the results file per case (with a per-run histogram), so
`parsec eval analyze` and `eval compare` read them without re-running
anything.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from parsec.evals.case import EvalCase
from parsec.evals.metrics import MetricContext
from parsec.evals.scoring import claim_texts, matches_any
from parsec.evals.support import score_claim_support
from parsec.models.events import EventType
from parsec.retrieval.fetcher import canonicalize_url
from parsec.store.blobs import BlobStore
from parsec.verify.omission import extraction_note
from parsec.verify.structural import verify_session


class LossPattern(StrEnum):
    # retrieval
    DOC_NOT_SURFACED = "doc_not_surfaced"
    DOC_NOT_FETCHED = "doc_not_fetched"
    DOC_FETCH_FAILED = "doc_fetch_failed"
    DOC_UNMINABLE = "doc_unminable"
    DISTRACTOR_FETCHED = "distractor_fetched"
    GOLD_UNREACHABLE = "gold_unreachable"
    # extraction
    DOC_NO_PREMISE = "doc_no_premise"
    GOLD_UNRECORDED = "gold_unrecorded"
    GOLD_PREMISE_REJECTED = "gold_premise_rejected"
    PREMISE_REJECTED = "premise_rejected"
    # writing
    GOLD_UNCITED = "gold_uncited"
    GOLD_UNMATCHED_CLAIM = "gold_unmatched_claim"
    GOLD_CONTRADICTED = "gold_contradicted"
    DISTRACTOR_CITED = "distractor_cited"
    CITATION_UNRESOLVED = "citation_unresolved"
    # verification
    CLAIM_UNFAITHFUL = "claim_unfaithful"
    CLAIM_UNSUPPORTED = "claim_unsupported"
    # coverage
    SUBQUESTION_OPEN = "subquestion_open"
    SUBQUESTION_PARTIAL = "subquestion_partial"
    SUBQUESTION_BLOCKED = "subquestion_blocked"
    SUBQUESTION_DROPPED = "subquestion_dropped"
    # harness
    BUDGET_HALT = "budget_halt"
    RUN_ERROR = "run_error"
    SUBAGENT_FAILED = "subagent_failed"
    MODEL_FAILED = "model_failed"
    MODEL_RETRIED = "model_retried"
    TOOL_ERROR = "tool_error"
    CONTEXT_COMPACTED = "context_compacted"
    # case authoring
    GOLD_PATTERN_ABSENT = "gold_pattern_absent"
    GOLD_UNLOCALIZED = "gold_unlocalized"


CATEGORY: dict[LossPattern, str] = {
    LossPattern.DOC_NOT_SURFACED: "retrieval",
    LossPattern.DOC_NOT_FETCHED: "retrieval",
    LossPattern.DOC_FETCH_FAILED: "retrieval",
    LossPattern.DOC_UNMINABLE: "retrieval",
    LossPattern.DISTRACTOR_FETCHED: "retrieval",
    LossPattern.GOLD_UNREACHABLE: "retrieval",
    LossPattern.DOC_NO_PREMISE: "extraction",
    LossPattern.GOLD_UNRECORDED: "extraction",
    LossPattern.GOLD_PREMISE_REJECTED: "extraction",
    LossPattern.PREMISE_REJECTED: "extraction",
    LossPattern.GOLD_UNCITED: "writing",
    LossPattern.GOLD_UNMATCHED_CLAIM: "writing",
    LossPattern.GOLD_CONTRADICTED: "writing",
    LossPattern.DISTRACTOR_CITED: "writing",
    LossPattern.CITATION_UNRESOLVED: "writing",
    LossPattern.CLAIM_UNFAITHFUL: "verification",
    LossPattern.CLAIM_UNSUPPORTED: "verification",
    LossPattern.SUBQUESTION_OPEN: "coverage",
    LossPattern.SUBQUESTION_PARTIAL: "coverage",
    LossPattern.SUBQUESTION_BLOCKED: "coverage",
    LossPattern.SUBQUESTION_DROPPED: "coverage",
    LossPattern.BUDGET_HALT: "harness",
    LossPattern.RUN_ERROR: "harness",
    LossPattern.SUBAGENT_FAILED: "harness",
    LossPattern.MODEL_FAILED: "harness",
    LossPattern.MODEL_RETRIED: "harness",
    LossPattern.TOOL_ERROR: "harness",
    LossPattern.CONTEXT_COMPACTED: "harness",
    LossPattern.GOLD_PATTERN_ABSENT: "case",
    LossPattern.GOLD_UNLOCALIZED: "case",
}

CATEGORY_ORDER = ("retrieval", "extraction", "writing", "verification", "coverage", "harness", "case")

_RETRIEVAL_STAGES = frozenset(
    {
        LossPattern.DOC_NOT_SURFACED,
        LossPattern.DOC_NOT_FETCHED,
        LossPattern.DOC_FETCH_FAILED,
        LossPattern.DOC_UNMINABLE,
    }
)
_COVERAGE_PATTERN = {
    "open": LossPattern.SUBQUESTION_OPEN,
    "partial": LossPattern.SUBQUESTION_PARTIAL,
    "blocked": LossPattern.SUBQUESTION_BLOCKED,
    "dropped": LossPattern.SUBQUESTION_DROPPED,
}


@dataclass
class Loss:
    pattern: LossPattern
    subject: str  # gold item text | URL | claim id | sq id | stream / tool name
    detail: str = ""
    count: int = 1

    @property
    def category(self) -> str:
        return CATEGORY[self.pattern]

    def to_payload(self) -> dict:
        return {
            "pattern": self.pattern.value,
            "category": self.category,
            "subject": self.subject,
            "detail": self.detail,
            "count": self.count,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Loss":
        return cls(
            LossPattern(payload["pattern"]),
            payload["subject"],
            payload.get("detail", ""),
            int(payload.get("count", 1)),
        )


def histogram(losses: Iterable[Loss]) -> dict[str, int]:
    """pattern -> summed count, in taxonomy order (stable for diffs)."""
    counts: dict[str, int] = {}
    for loss in losses:
        counts[loss.pattern.value] = counts.get(loss.pattern.value, 0) + loss.count
    return {p.value: counts[p.value] for p in LossPattern if p.value in counts}


def histogram_from_results(results: list[dict]) -> dict[str, int]:
    """The same histogram read back from a results file's per-case payloads
    (pre-M15 files carry no losses and read as empty)."""
    losses = [Loss.from_payload(p) for r in results for p in (r.get("losses") or [])]
    return histogram(losses)


def merge_losses(runs: list[list[Loss]]) -> list[Loss]:
    """Across `--runs N`: counts sum per (pattern, subject); the first run's
    detail stands. Order follows first appearance."""
    merged: dict[tuple[str, str], Loss] = {}
    for losses in runs:
        for loss in losses:
            key = (loss.pattern.value, loss.subject)
            if key in merged:
                merged[key].count += loss.count
            else:
                merged[key] = Loss(loss.pattern, loss.subject, loss.detail, loss.count)
    return list(merged.values())


# --- the diagnosis ----------------------------------------------------------


@dataclass
class _DocView:
    url: str
    doc_hash: str
    ok: bool
    outcome: str
    status_code: int


@dataclass
class _Trace:
    """What the event log says happened, in the shape the ladders need."""

    surfaced: set[str]
    attempted: set[str]
    fetch_errors: dict[str, str]
    fetched: dict[str, _DocView]
    spans_by_doc: dict[str, int]
    rejections: list[tuple[str, str, str]]  # (stream, premise text, reason)
    tool_errors: dict[str, tuple[int, str]]
    retries: dict[str, int]
    failed: dict[str, tuple[int, str]]
    compactions: dict[str, int]
    subagent_failures: list[tuple[str, str, str]]
    budget_halt: str | None
    session_errors: list[str]
    unresolved: list[str]


def _blob_json(blobs: BlobStore, sha: str | None) -> dict:
    if not sha or not blobs.exists(sha):
        return {}
    try:
        return json.loads(blobs.get_text(sha))
    except (ValueError, UnicodeDecodeError):
        return {}


def _read_trace(ctx: MetricContext) -> _Trace:
    t = _Trace(set(), set(), {}, {}, {}, [], {}, {}, {}, {}, [], None, [], [])
    intents: dict[str, tuple[str, dict, str]] = {}
    for ev in ctx.event_log.read(ctx.session_id):
        p = ev.payload
        et = ev.event_type
        if et == EventType.TOOL_INTENT:
            intents[p["tool_use_id"]] = (p["tool_name"], p.get("input") or {}, ev.stream_id)
            if p["tool_name"] == "fetch" and p.get("input", {}).get("url"):
                t.attempted.add(canonicalize_url(p["input"]["url"]))
        elif et == EventType.TOOL_RESULT:
            tool_name, input_, stream = intents.get(p["tool_use_id"], ("?", {}, ev.stream_id))
            if not p.get("ok", True):
                error = p.get("error") or ""
                count, first = t.tool_errors.get(tool_name, (0, error))
                t.tool_errors[tool_name] = (count + 1, first)
                if tool_name == "fetch" and input_.get("url"):
                    t.fetch_errors.setdefault(canonicalize_url(input_["url"]), error)
                continue
            if tool_name == "search_broad":
                full = _blob_json(ctx.blobs, p.get("full_blob"))
                for hit in full.get("hits", []):
                    if hit.get("url"):
                        t.surfaced.add(canonicalize_url(hit["url"]))
            elif tool_name == "record_premises":
                full = _blob_json(ctx.blobs, p.get("full_blob"))
                drafts = input_.get("premises") or []
                for r in full.get("results", []):
                    if "error" in r:
                        idx = r.get("index", -1)
                        text = drafts[idx].get("text", "") if 0 <= idx < len(drafts) else ""
                        t.rejections.append((stream, text, r["error"]))
        elif et == EventType.FETCH_PERFORMED:
            url = canonicalize_url(p["url"])
            ok = p.get("outcome", "ok") == "ok" and int(p.get("status_code", 200)) < 400
            view = _DocView(url, p["doc_hash"], ok, p.get("outcome", "ok"), int(p.get("status_code", 0)))
            prior = t.fetched.get(url)
            if prior is None or (ok and not prior.ok):
                t.fetched[url] = view
        elif et == EventType.SPAN_INDEXED:
            n = len(p.get("span_ids") or [])
            t.spans_by_doc[p["doc_hash"]] = max(t.spans_by_doc.get(p["doc_hash"], 0), n)
        elif et == EventType.ERROR:
            kind, detail = p.get("kind", "error"), p.get("detail", "")
            if kind == "budget_exceeded":
                t.budget_halt = detail
            elif p.get("sq_id"):
                t.subagent_failures.append((p["sq_id"], kind, detail))
            else:
                t.session_errors.append(f"{kind}: {detail}")
        elif et == EventType.LLM_FAILED:
            count, first = t.failed.get(ev.stream_id, (0, p.get("kind") or p.get("error_kind") or "failed"))
            t.failed[ev.stream_id] = (count + 1, first)
        elif et == EventType.LLM_RETRY:
            t.retries[ev.stream_id] = t.retries.get(ev.stream_id, 0) + 1
        elif et == EventType.CONTEXT_COMPACTED:
            t.compactions[ev.stream_id] = t.compactions.get(ev.stream_id, 0) + 1
        elif et == EventType.ANSWER_EMITTED:
            t.unresolved = list(p.get("unresolved") or [])
    return t


@dataclass
class _Graph:
    premises: dict[str, str]  # premise id -> text
    cited_premises: set[str]
    docs_with_premise: set[str]
    cited_docs: set[str]
    claims: list[str]


def _read_graph(conn: sqlite3.Connection, session_id: str) -> _Graph:
    nodes: dict[str, tuple[str, dict]] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?", (session_id,)
    ):
        nodes[row["node_id"]] = (row["node_type"], json.loads(row["payload_json"]))
    out_edges: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?", (session_id,)
    ):
        if row["edge_type"] != "contradicts":
            out_edges.setdefault(row["src_node_id"], []).append(row["dst_node_id"])

    premises = {nid: p["text"] for nid, (kind, p) in nodes.items() if kind == "Premise"}
    span_doc = {nid: p["doc_hash"] for nid, (kind, p) in nodes.items() if kind == "SourceSpan"}
    docs_with_premise = {
        span_doc[dst] for pid in premises for dst in out_edges.get(pid, []) if dst in span_doc
    }
    claims = sorted(
        nid for nid, (kind, p) in nodes.items() if kind == "ReportClaim" and not p.get("narrative")
    )
    reachable: set[str] = set()
    stack = list(claims)
    while stack:
        cur = stack.pop()
        for dst in out_edges.get(cur, []):
            if dst in nodes and dst not in reachable:
                reachable.add(dst)
                stack.append(dst)
    return _Graph(
        premises=premises,
        cited_premises={p for p in premises if p in reachable},
        docs_with_premise=docs_with_premise,
        cited_docs={span_doc[n] for n in reachable if n in span_doc},
        claims=claims,
    )


def _doc_stage(
    conn: sqlite3.Connection, url: str, trace: _Trace, graph: _Graph
) -> tuple[LossPattern, str] | None:
    """The deepest stage a document reached, or None when a premise was
    recorded from it (the retrieval + extraction pipeline did its job)."""
    cu = canonicalize_url(url)
    view = trace.fetched.get(cu)
    if view is None:
        if cu in trace.fetch_errors:
            return LossPattern.DOC_FETCH_FAILED, f"fetch tool errored: {trace.fetch_errors[cu]}"
        if cu in trace.surfaced:
            return LossPattern.DOC_NOT_FETCHED, "surfaced by search_broad, never fetched"
        return LossPattern.DOC_NOT_SURFACED, "never appeared in any search_broad result"
    if not view.ok:
        return LossPattern.DOC_FETCH_FAILED, f"outcome {view.outcome}, HTTP {view.status_code}"
    spans = trace.spans_by_doc.get(view.doc_hash, 0)
    if spans == 0:
        note = extraction_note(conn, view.doc_hash)
        return LossPattern.DOC_UNMINABLE, note or "fetched, no indexable text"
    if view.doc_hash not in graph.docs_with_premise:
        return LossPattern.DOC_NO_PREMISE, f"{spans} span(s) indexed, no premise recorded from them"
    return None


def _doc_texts(conn: sqlite3.Connection, trace: _Trace) -> list[tuple[str, str]]:
    """(url, lowercased full text) for every document fetched OK this
    session — the spans table is corpus-wide, so filter by this run's
    fetches, never by what the frozen corpus happens to hold."""
    out = []
    for cu in sorted(trace.fetched):
        view = trace.fetched[cu]
        if not view.ok:
            continue
        rows = conn.execute(
            "SELECT text FROM spans WHERE doc_hash=? ORDER BY char_start", (view.doc_hash,)
        ).fetchall()
        if rows:
            out.append((view.url, " ".join(r["text"] for r in rows).lower()))
    return out


def _localize_miss(
    text: str,
    patterns: list[str],
    case: EvalCase,
    graph: _Graph,
    trace: _Trace,
    doc_texts: list[tuple[str, str]],
    gold_stages: dict[str, tuple[LossPattern, str] | None],
) -> Loss:
    matching = sorted(pid for pid, ptext in graph.premises.items() if matches_any(patterns, [ptext.lower()]))
    if matching:
        cited = [p for p in matching if p in graph.cited_premises]
        if cited:
            return Loss(
                LossPattern.GOLD_UNMATCHED_CLAIM, text,
                f"a claim rests on {cited[0]}, which matches the gold; the claim's wording "
                "does not — loosen the pattern or inspect the claim",
            )
        return Loss(LossPattern.GOLD_UNCITED, text, f"recorded as {matching[0]}; no claim cites it")
    for _stream, ptext, reason in trace.rejections:
        if ptext and matches_any(patterns, [ptext.lower()]):
            return Loss(LossPattern.GOLD_PREMISE_REJECTED, text, reason)
    for url, dtext in doc_texts:
        if matches_any(patterns, [dtext]):
            return Loss(LossPattern.GOLD_UNRECORDED, text, f"stated in fetched {url}; no premise recorded it")
    if not case.gold_docs:
        return Loss(
            LossPattern.GOLD_UNLOCALIZED, text,
            "not in any fetched text and the case declares no gold_docs — declare them to localize",
        )
    unread = [
        f"{url}: {stage[0].value}" for url, stage in gold_stages.items()
        if stage is not None and stage[0] in _RETRIEVAL_STAGES
    ]
    if unread:
        return Loss(LossPattern.GOLD_UNREACHABLE, text, "; ".join(unread))
    return Loss(
        LossPattern.GOLD_PATTERN_ABSENT, text,
        "every gold doc was read and none contains a match — check the case's patterns",
    )


def diagnose_losses(ctx: MetricContext, error: str | None = None) -> list[Loss]:
    """Every named loss for one eval run, gold items first. A clean run with
    full gold recall and an untroubled trajectory returns []."""
    conn, sid, case = ctx.conn, ctx.session_id, ctx.case
    losses: list[Loss] = []
    trace = _read_trace(ctx)
    graph = _read_graph(conn, sid)

    # Documents: gold stage per URL, distractors consulted or cited.
    gold_stages = {url: _doc_stage(conn, url, trace, graph) for url in case.gold_docs}
    doc_losses = [
        Loss(stage[0], url, stage[1]) for url, stage in gold_stages.items() if stage is not None
    ]
    for url in case.distractor_docs:
        view = trace.fetched.get(canonicalize_url(url))
        if view is None:
            continue
        if view.doc_hash in graph.cited_docs:
            doc_losses.append(
                Loss(LossPattern.DISTRACTOR_CITED, url, "a claim rests on a span of this planted source")
            )
        else:
            doc_losses.append(Loss(LossPattern.DISTRACTOR_FETCHED, url, "consulted; nothing cited from it"))

    # Gold items: must_find entries and nuggets, localized by the ladder.
    texts = [t.lower() for t in claim_texts(conn, sid)]
    doc_texts = _doc_texts(conn, trace)
    items = [(entry, [entry], []) for entry in case.must_find]
    items += [(n.text, n.patterns, n.contradiction_patterns) for n in case.nuggets]
    for text, patterns, contradiction in items:
        if contradiction and matches_any(contradiction, texts):
            losses.append(Loss(LossPattern.GOLD_CONTRADICTED, text, "the report asserts the opposite"))
        elif not matches_any(patterns, texts):
            losses.append(_localize_miss(text, patterns, case, graph, trace, doc_texts, gold_stages))
    losses.extend(doc_losses)

    # Claims: structural violations, unsupported claims, citations left unresolved.
    if graph.claims:
        report = verify_session(conn, ctx.blobs, sid)
        claim_ids = set(graph.claims)
        checks: dict[str, set[str]] = {}
        for v in report.violations:
            if v.subject in claim_ids:
                checks.setdefault(v.subject, set()).add(v.check)
        for cid in sorted(checks):
            losses.append(Loss(LossPattern.CLAIM_UNFAITHFUL, cid, ", ".join(sorted(checks[cid]))))
        _, support = score_claim_support(conn, sid, ctx.support_checker)
        for d in support:
            if d.grade == "none":
                losses.append(Loss(LossPattern.CLAIM_UNSUPPORTED, d.claim_id, d.claim_text[:160]))
    if trace.unresolved:
        losses.append(
            Loss(
                LossPattern.CITATION_UNRESOLVED, "writer",
                f"after repair: {trace.unresolved[0]}", count=len(trace.unresolved),
            )
        )

    # Coverage: every row that did not end answered, with its recorded reason.
    for row in conn.execute(
        "SELECT sq_id, status, reason FROM coverage WHERE session_id=? ORDER BY sq_id", (sid,)
    ):
        pattern = _COVERAGE_PATTERN.get(row["status"])
        if pattern is not None:
            losses.append(Loss(pattern, row["sq_id"], row["reason"] or ""))

    # Harness incidents.
    if error:
        losses.append(Loss(LossPattern.RUN_ERROR, "run", error))
    else:
        for detail in trace.session_errors:
            losses.append(Loss(LossPattern.RUN_ERROR, "run", detail))
    if trace.budget_halt:
        losses.append(Loss(LossPattern.BUDGET_HALT, "session", trace.budget_halt))
    for sq_id, kind, detail in trace.subagent_failures:
        losses.append(Loss(LossPattern.SUBAGENT_FAILED, sq_id, f"{kind}: {detail}"))
    for stream in sorted(trace.failed):
        count, kind = trace.failed[stream]
        losses.append(Loss(LossPattern.MODEL_FAILED, stream, kind, count=count))
    for stream in sorted(trace.retries):
        losses.append(Loss(LossPattern.MODEL_RETRIED, stream, "journaled harness retries", count=trace.retries[stream]))
    for tool in sorted(trace.tool_errors):
        count, first = trace.tool_errors[tool]
        losses.append(Loss(LossPattern.TOOL_ERROR, tool, first, count=count))
    for stream in sorted(trace.compactions):
        losses.append(Loss(LossPattern.CONTEXT_COMPACTED, stream, "compaction ladder engaged", count=trace.compactions[stream]))
    by_stream: dict[str, tuple[int, str]] = {}
    for stream, _text, reason in trace.rejections:
        count, first = by_stream.get(stream, (0, reason))
        by_stream[stream] = (count + 1, first)
    for stream in sorted(by_stream):
        count, first = by_stream[stream]
        losses.append(Loss(LossPattern.PREMISE_REJECTED, stream, first, count=count))
    return losses
