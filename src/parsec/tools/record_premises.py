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
import re

from pydantic import BaseModel, ConfigDict, Field

from parsec.models.report import PREMISE_MAX_CHARS, PremiseDraft
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext
from parsec.verify.containment import check_containment, extract_numbers
from parsec.verify.lints import lint_premise
from parsec.verify.nli import GroundedChecker, make_grounded_checker


# Near-duplicate gate: the same fact re-recorded in a fresh phrasing
# fragments corroboration across premise nodes and bloats the writer's
# context. Token-set Jaccard is deterministic, cheap, and conservative —
# paraphrases of one sentence clear 0.8; a genuinely different subject does
# not. Quantities must match EXACTLY on top of the Jaccard: "90 degrees"
# rephrasing "100 degrees" is a conflict to record, never a duplicate.
_DUP_TOKEN_RE = re.compile(r"[a-z0-9]+")
DUP_JACCARD = 0.8


def _dup_tokens(text: str) -> frozenset[str]:
    return frozenset(_DUP_TOKEN_RE.findall(text.lower()))


def _near_duplicate(text_a: str, tokens_a: frozenset[str], entry_b: dict) -> bool:
    a, b = tokens_a, entry_b["tokens"]
    if not a or not b:
        return False
    if len(a & b) / len(a | b) < DUP_JACCARD:
        return False
    return set(extract_numbers(text_a)) == entry_b["numbers"]


class LenientPremiseDraft(PremiseDraft):
    """PremiseDraft without the schema-level length cap: run() checks the cap
    per premise, so one overlong text is rejected alone (with the reason)
    instead of failing the entire batch before the tool runs."""

    text: str = Field(min_length=1)


class RecordPremisesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[LenientPremiseDraft] = Field(min_length=1, max_length=20)


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

    def __init__(self, dag: DagStore, spans: SpanStore, documents: DocumentStore):
        self.dag = dag
        self.spans = spans
        self.documents = documents
        # Grounded checker selected by the session's frozen config, so replay
        # reconstructs the identical advisory behavior. Cached per tool.
        self._nli: GroundedChecker | None = None
        self._nli_name: str | None = None

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
                    "tokens": _dup_tokens(payload["text"]),
                    "numbers": set(extract_numbers(payload["text"])),
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

    def _corroborate(
        self,
        sid: str,
        dup: dict,
        span_rows: list,
        span_nodes: dict[str, str],
        attached: dict[str, set[str]],
    ) -> int:
        """Attach a near-duplicate's NEW spans to the existing premise as
        extracts edges — but only if the existing premise's text survives the
        mechanical containment check against its evidence including them
        (the new spans corroborate the recorded wording, not the paraphrase).
        Returns how many spans were attached."""
        already = attached.setdefault(dup["node_id"], set())
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
                self.dag.add_edge(sid, dup["node_id"], node, "extracts", None)
                already.add(node)
                added += 1
        return added

    async def run(self, input: RecordPremisesInput, ctx: ToolContext) -> tuple[dict, str]:
        sid = ctx.config.session_id
        checker = self._checker(ctx)
        results: list[dict] = []
        lines: list[str] = []
        known, span_nodes, attached = self._session_premises(sid)

        for i, draft in enumerate(input.premises):
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
            problems += check_containment(
                draft.text, [r["text"] for r in span_rows], draft.transform_note
            )
            if problems:
                error = "; ".join(problems)
                results.append({"index": i, "error": error})
                lines.append(f"REJECTED premise {i}: {error}")
                continue

            # Near-duplicate of an already-recorded premise: no new node —
            # attach any genuinely new spans to the existing one instead, so
            # corroboration accrues to a single premise rather than
            # fragmenting across paraphrases.
            tokens = _dup_tokens(draft.text)
            dup = next((p for p in known if _near_duplicate(draft.text, tokens, p)), None)
            if dup is not None:
                added = self._corroborate(sid, dup, span_rows, span_nodes, attached)
                results.append({"index": i, "premise_id": dup["node_id"], "duplicate": True})
                lines.append(
                    f"[{dup['node_id']}] already recorded (near-duplicate of: {dup['text']}) "
                    "— cite this ID"
                    + (f"; {added} corroborating span(s) attached" if added else "")
                )
                continue

            for row in span_rows:
                self._span_node(sid, row, span_nodes)
            premise_id = self.dag.add_node(
                sid,
                "Premise",
                {
                    "text": draft.text,
                    "span_refs": draft.span_refs,
                    "claim_class": draft.claim_class,
                },
            )
            edge_payload = (
                {"transform_note": draft.transform_note} if draft.transform_note else None
            )
            for ref in draft.span_refs:
                self.dag.add_edge(sid, premise_id, span_nodes[ref], "extracts", edge_payload)
                attached.setdefault(premise_id, set()).add(span_nodes[ref])
            known.append(
                {
                    "node_id": premise_id,
                    "text": draft.text,
                    "tokens": tokens,
                    "numbers": set(extract_numbers(draft.text)),
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
