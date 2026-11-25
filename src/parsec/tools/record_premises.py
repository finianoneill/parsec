"""record_premises: the model's only way to get facts into the DAG.

Each premise is validated by the harness (T1): every span ref must have
been returned by fetch, the premise must survive the Claimify-style quality
lints (M9 — ambiguous or vague premises are rejected with the reason, not
recorded vaguely), and numbers/quotes must survive the mechanical
containment check (§6 stage 1) — or the premise is rejected back to the
model with the reason. Accepted premises become tier-1 Premise nodes with
`extracts` edges to tier-0 SourceSpan nodes, and their IDs are what the
writer later cites.

On top of the hard gates, the grounded-NLI tier (M9, T9) checks whether the
cited spans actually appear to SUPPORT each accepted premise; a
non-supported verdict is returned to the subagent as an advisory NOTE —
never a rejection, because NLI error rates are real and exact-match is the
floor, not this.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from parsec.config import RunConfig, behaviour_level
from parsec.models.report import PREMISE_MAX_CHARS, PremiseDraft, PremiseDraftV2, PremiseDraftV3
from parsec.retrieval.extract import extract_published
from parsec.store.blobs import BlobStore
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import CURRENT_STREAM
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext
from parsec.verify.containment import (
    check_containment,
    check_premise_verbatim,
    extract_numbers,
)
from parsec.verify.dating import (
    DATE_SOURCE_AS_OF,
    DATE_SOURCE_PUBLISHED,
    DATE_SOURCE_UNKNOWN,
    NOW,
    check_date_containment,
    format_interval,
    published_interval,
)
from parsec.verify.lints import lint_premise
from parsec.verify.nli import GroundedChecker, make_grounded_checker
from parsec.verify.propositions import dup_tokens, near_duplicate

# Near-duplicate gate: the same fact re-recorded in a fresh phrasing
# fragments corroboration across premise nodes and bloats the writer's
# context. What counts as "the same fact" lives in verify.propositions.


class LenientPremiseDraft(PremiseDraft):
    """PremiseDraft without the schema-level length cap: run() checks the cap
    per premise, so one overlong text is rejected alone (with the reason)
    instead of failing the entire batch before the tool runs."""

    text: str = Field(min_length=1)


class RecordPremisesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[LenientPremiseDraft] = Field(min_length=1, max_length=20)


class LenientPremiseDraftV2(PremiseDraftV2):
    """One atomic premise, stated exactly as its spans state it."""

    # (A class docstring becomes the JSON schema's description, so it is
    # written for the model. For maintainers: this is the derivations >= 2
    # draft; length is checked per premise by run(), as for the legacy one.)

    text: str = Field(min_length=1)


class RecordPremisesInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[LenientPremiseDraftV2] = Field(min_length=1, max_length=20)


class LenientPremiseDraftV3(PremiseDraftV3):
    """One atomic premise, stated exactly as its spans state it."""

    # recency >= 2: adds as_of. Length checked per premise by run().
    text: str = Field(min_length=1)


class RecordPremisesInputV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[LenientPremiseDraftV3] = Field(min_length=1, max_length=20)


AnyPremiseDraft = LenientPremiseDraft | LenientPremiseDraftV2 | LenientPremiseDraftV3


class RecordPremisesTool:
    name = "record_premises"
    description = (
        "Record atomic factual premises extracted from fetched spans. Each premise: one "
        "subject, one predicate, quantities stated exactly as the span states them (or add "
        "transform_note explaining a derivation). A premise must stand alone: name the "
        "specific entity (never a bare 'it' or 'the study'), and state specifics instead of "
        "vague terms like 'benefits' or 'significant'. Keep each premise text under "
        f"{PREMISE_MAX_CHARS} characters — longer ones are rejected; split them. span_refs "
        "must be span IDs returned by fetch. Returns premise IDs — your final answer will "
        "cite these, so record every fact you intend to use, as soon as you have its spans. "
        "Rejected premises include the reason; fix and re-record them. A near-duplicate of "
        "an already-recorded premise is not re-recorded: you get the existing premise ID "
        "(with any new spans attached as corroboration) — cite that ID instead of "
        "rephrasing a known fact."
    )
    input_model = RecordPremisesInput
    max_context_chars = 6000

    # derivations >= 2. The text above is frozen (it is prompt bytes for
    # every recording made under it); this is the same text with the
    # transform_note clause replaced.
    _TRANSFORM_CLAUSE = "(or add transform_note explaining a derivation)"
    assert description.count(_TRANSFORM_CLAUSE) == 1
    description_verbatim = description.replace(
        _TRANSFORM_CLAUSE,
        "— a premise never holds a figure you derived: conversions, sums, ratios and the "
        "like go in submit_report as computes findings",
    )

    # recency >= 2 (which presumes derivations >= 2: there is no dated draft
    # with a transform_note).
    description_dated = description_verbatim + (
        " If the span states WHEN the fact held, give that date as as_of — it decides how "
        "the fact is weighed against the time the question asks about."
    )

    def spec_for(self, config: RunConfig) -> tuple[str, type[BaseModel]]:
        if behaviour_level(config.recency) >= 2:
            return self.description_dated, RecordPremisesInputV3
        if behaviour_level(config.derivations) >= 2:
            return self.description_verbatim, RecordPremisesInputV2
        return self.description, self.input_model

    def __init__(self, dag: DagStore, spans: SpanStore, documents: DocumentStore):
        self.dag = dag
        self.spans = spans
        self.documents = documents
        # Grounded checker selected by the session's frozen config, so replay
        # reconstructs the identical advisory behavior. Cached per tool.
        self._nli: GroundedChecker | None = None
        self._nli_name: str | None = None
        self._published_cache: dict[str, str | None] = {}

    def _checker(self, ctx: ToolContext) -> GroundedChecker | None:
        name = ctx.config.nli_checker
        if name != self._nli_name:
            self._nli = make_grounded_checker(name)
            self._nli_name = name
        return self._nli

    def _session_premises(self, sid: str) -> tuple[list[dict], dict[str, str], dict[str, set[str]]]:
        """The dedup working set: recorded premises (with token sets), the
        span_id -> SourceSpan node map, and each premise's attached span
        nodes (from extracts edges — the source of truth for corroboration;
        the payload's span_refs stay as recorded)."""
        premises = []
        for row in self.dag.nodes_for_session(sid, tier=1):
            payload = json.loads(row["payload_json"])
            premises.append(
                {
                    "node_id": row["node_id"],
                    "text": payload["text"],
                    "tokens": dup_tokens(payload["text"]),
                    "numbers": set(extract_numbers(payload["text"])),
                    "target": payload.get("target_time") or NOW,
                }
            )
        span_nodes = {
            json.loads(row["payload_json"])["span_id"]: row["node_id"]
            for row in self.dag.nodes_for_session(sid, tier=0)
        }
        attached: dict[str, set[str]] = {}
        for e in self.dag.edges_for_session(sid):
            if e["edge_type"] == "extracts":
                attached.setdefault(e["src_node_id"], set()).add(e["dst_node_id"])
        return premises, span_nodes, attached

    def _span_node(self, sid: str, row, span_nodes: dict[str, str]) -> str:
        ref = row["span_id"]
        if ref not in span_nodes:
            doc = self.documents.get_document(row["doc_hash"])
            if doc is None:  # spans reference documents by foreign key
                raise LookupError(f"span {ref} has no document row")
            span_nodes[ref] = self.dag.add_node(
                sid,
                "SourceSpan",
                {
                    "span_id": ref,
                    "doc_hash": row["doc_hash"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "text": row["text"],
                    "url": doc["url"],
                    "fetched_ts": doc["fetched_ts"],
                },
            )
        return span_nodes[ref]

    def _published(self, doc_hash: str, blobs: BlobStore) -> str | None:
        """A document's publication date. Documents archived before dates
        were extracted have none in their stored metadata, and a cache hit or
        an identical-byte re-fetch never rewrites that row — so for those the
        date is read from the archived bytes instead. It is a pure function
        of bytes that never change, so a replay computes the same date."""
        if doc_hash in self._published_cache:
            return self._published_cache[doc_hash]
        doc = self.documents.get_document(doc_hash)
        published: str | None = None
        if doc is not None:
            meta = json.loads(doc["meta_json"])
            published = meta.get("published")
            extracted_with = str(meta.get("extractor_version", "0"))
            predates = not extracted_with.isdigit() or int(extracted_with) < 4
            if published is None and predates and "outcome" not in meta:
                try:
                    published = extract_published(blobs.get(doc["raw_blob"]), doc["content_type"])
                except OSError:
                    published = None  # raw bytes gone: corpus-integrity reports that
        self._published_cache[doc_hash] = published
        return published

    def _observation(self, row, as_of: str | None, blobs: BlobStore) -> dict:
        """recency >= 2: when the fact was observed, as far as THIS span can
        say — the payload of the premise's edge to it. The premise's own
        as_of if this span states it; else the document's publication date;
        else unknown. Dated per edge because one span can support two
        premises about different years."""
        if as_of:
            period, _ = check_date_containment(as_of, row["text"])
            if period is not None:
                return {"observed_at": format_interval(period), "date_source": DATE_SOURCE_AS_OF}
        day = published_interval(self._published(row["doc_hash"], blobs))
        if day is not None:
            return {"observed_at": format_interval(day), "date_source": DATE_SOURCE_PUBLISHED}
        return {"date_source": DATE_SOURCE_UNKNOWN}

    def _corroborate(
        self,
        sid: str,
        dup: dict,
        span_rows: list,
        span_nodes: dict[str, str],
        attached: dict[str, set[str]],
        as_of: str | None = None,
        dated: bool = False,
        blobs: BlobStore | None = None,
    ) -> int:
        """Attach a near-duplicate's NEW spans to the existing premise as
        extracts edges — but only if the existing premise's text survives the
        mechanical containment check against its evidence including them
        (the new spans corroborate the recorded wording, not the paraphrase).
        Returns how many spans were attached."""
        already = attached.setdefault(dup["node_id"], set())
        if dated and as_of and blobs is not None:
            self._fill_in_dates(sid, dup["node_id"], span_rows, span_nodes, already, as_of)
        new_rows = [
            r for r in span_rows
            if span_nodes.get(r["span_id"]) is None or span_nodes[r["span_id"]] not in already
        ]
        if not new_rows:
            return 0
        # The new spans must support the recorded text ON THEIR OWN — spans
        # already attached would satisfy any union check, letting a span
        # that never states the fact ride in as "corroboration".
        if check_containment(dup["text"], [r["text"] for r in new_rows], None):
            return 0
        added = 0
        for row in new_rows:
            node = self._span_node(sid, row, span_nodes)
            if node not in already:
                # A corroborating span carries its own observation date.
                payload = (
                    self._observation(row, as_of, blobs) if dated and blobs is not None else None
                )
                self.dag.add_edge(
                    sid, dup["node_id"], node, "extracts", payload, log_payload=dated
                )
                already.add(node)
                added += 1
        return added

    def _fill_in_dates(
        self,
        sid: str,
        premise_id: str,
        span_rows: list,
        span_nodes: dict[str, str],
        already: set[str],
        as_of: str,
    ) -> None:
        """A premise re-recorded WITH a date its span states gets that date
        on the existing edge, when the edge had only a publication date or
        none. A date the premise already gave is never overwritten, so an
        exact retry changes nothing."""
        for row in span_rows:
            node = span_nodes.get(row["span_id"])
            if node is None or node not in already:
                continue
            period, _ = check_date_containment(as_of, row["text"])
            if period is None:
                continue
            stored = self.dag.conn.execute(
                "SELECT payload_json FROM edges WHERE session_id=? AND src_node_id=?"
                " AND dst_node_id=? AND edge_type='extracts'",
                (sid, premise_id, node),
            ).fetchone()
            current = json.loads(stored["payload_json"] or "{}") if stored is not None else {}
            if stored is None or current.get("date_source") == DATE_SOURCE_AS_OF:
                continue
            self.dag.update_edge_payload(
                sid, premise_id, node, "extracts",
                {**current, "observed_at": format_interval(period), "date_source": DATE_SOURCE_AS_OF},
            )

    async def run(
        self,
        input: RecordPremisesInput | RecordPremisesInputV2 | RecordPremisesInputV3,
        ctx: ToolContext,
    ) -> tuple[dict, str]:
        sid = ctx.config.session_id
        checker = self._checker(ctx)
        results: list[dict] = []
        lines: list[str] = []
        known, span_nodes, attached = self._session_premises(sid)
        # claim_credence >= 2: word order is part of what a premise says.
        order_aware = behaviour_level(ctx.config.claim_credence) >= 2
        # derivations >= 2: premises are verbatim, with no transform_note.
        verbatim = behaviour_level(ctx.config.derivations) >= 2
        # recency >= 2: every premise records the time its subquestion asks
        # about, and every edge to a span records when the fact was observed.
        dated = behaviour_level(ctx.config.recency) >= 2
        target = ctx.stream_targets.get(CURRENT_STREAM.get(), NOW) if dated else None

        drafts: list[AnyPremiseDraft] = list(input.premises)
        for i, draft in enumerate(drafts):
            if len(draft.text) > PREMISE_MAX_CHARS:
                error = (
                    f"text is {len(draft.text)} chars (max {PREMISE_MAX_CHARS}); "
                    "split it into atomic premises, one subject and one predicate each"
                )
                results.append({"index": i, "error": error})
                lines.append(f"REJECTED premise {i}: {error}")
                continue
            span_rows = []
            missing = []
            for ref in draft.span_refs:
                row = self.spans.get(ref)
                if row is None:
                    missing.append(ref)
                else:
                    span_rows.append(row)
            if missing:
                error = f"unknown span id(s) (never returned by fetch): {', '.join(missing)}"
                results.append({"index": i, "error": error})
                lines.append(f"REJECTED premise {i}: {error}")
                continue

            problems = lint_premise(draft.text)
            span_texts = [r["text"] for r in span_rows]
            transform_note = getattr(draft, "transform_note", None)  # legacy drafts only
            if verbatim:
                problems += check_premise_verbatim(draft.text, span_texts)
            else:
                problems += check_containment(draft.text, span_texts, transform_note)
            as_of = getattr(draft, "as_of", None) if dated else None
            if as_of:
                # Same floor as numbers: a date the model gives must be a
                # date some cited span states.
                checks = [check_date_containment(as_of, text) for text in span_texts]
                if not any(period is not None for period, _ in checks):
                    problems.append(next(problem for _, problem in checks if problem))
            if problems:
                error = "; ".join(problems)
                results.append({"index": i, "error": error})
                lines.append(f"REJECTED premise {i}: {error}")
                continue

            # Near-duplicate of an already-recorded premise: no new node —
            # attach any genuinely new spans to the existing one instead, so
            # corroboration accrues to a single premise rather than
            # fragmenting across paraphrases.
            tokens = dup_tokens(draft.text)
            dup = next(
                (
                    p
                    for p in known
                    # the same words about a different time frame are a
                    # different premise: evidence for "now" is not evidence
                    # for "in 2019"
                    if (not dated or p["target"] == target)
                    and near_duplicate(draft.text, tokens, p, order_aware)
                ),
                None,
            )
            if dup is not None:
                added = self._corroborate(
                    sid, dup, span_rows, span_nodes, attached,
                    as_of=as_of, dated=dated, blobs=ctx.blobs,
                )
                results.append({"index": i, "premise_id": dup["node_id"], "duplicate": True})
                lines.append(
                    f"[{dup['node_id']}] already recorded (near-duplicate of: {dup['text']}) "
                    "— cite this ID"
                    + (f"; {added} corroborating span(s) attached" if added else "")
                )
                continue

            for row in span_rows:
                self._span_node(sid, row, span_nodes)
            premise_payload: dict = {
                "text": draft.text,
                "span_refs": draft.span_refs,
                "claim_class": draft.claim_class,
            }
            if dated and target != NOW:
                # Only a premise about some OTHER time says so: "now" is the
                # absence of the field. A premise about the present therefore
                # keeps the id it always had, and premises recorded before
                # target times existed read, correctly, as being about now.
                premise_payload["target_time"] = target
            premise_id = self.dag.add_node(sid, "Premise", premise_payload)
            edge_payload = {"transform_note": transform_note} if transform_note else None
            rows_by_ref = {r["span_id"]: r for r in span_rows}
            for ref in draft.span_refs:
                if dated:
                    edge_payload = self._observation(rows_by_ref[ref], as_of, ctx.blobs)
                self.dag.add_edge(
                    sid, premise_id, span_nodes[ref], "extracts", edge_payload, log_payload=dated
                )
                attached.setdefault(premise_id, set()).add(span_nodes[ref])
            known.append(
                {
                    "node_id": premise_id,
                    "text": draft.text,
                    "tokens": tokens,
                    "numbers": set(extract_numbers(draft.text)),
                    "target": target,
                }
            )
            result: dict = {"index": i, "premise_id": premise_id}
            lines.append(f"[{premise_id}] recorded: {draft.text}")

            if checker is not None:
                verdict = checker.check(draft.text, [r["text"] for r in span_rows])
                if verdict.flagged:
                    result["support"] = {
                        "verdict": verdict.verdict,
                        "score": round(verdict.score, 4),
                        "unsupported_terms": list(verdict.unsupported_terms),
                        "checker": verdict.checker,
                    }
                    lines.append(
                        f"NOTE [{premise_id}]: {verdict.describe()} — advisory: the cited "
                        "span may not state this; cite a span that does, or revise the premise"
                    )
            results.append(result)

        return {"results": results}, "\n".join(lines)
