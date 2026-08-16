"""record_premises: the model's only way to get facts into the DAG.

Each premise is validated by the harness (T1): every span ref must have
been returned by fetch, and numbers/quotes must survive the mechanical
containment check (§6 stage 1) — or the premise is rejected back to the
model with the reason. Accepted premises become tier-1 Premise nodes with
`extracts` edges to tier-0 SourceSpan nodes, and their IDs are what the
writer later cites.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from parsec.models.report import PremiseDraft
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext
from parsec.verify.containment import check_containment


class RecordPremisesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premises: list[PremiseDraft] = Field(min_length=1, max_length=20)


class RecordPremisesTool:
    name = "record_premises"
    description = (
        "Record atomic factual premises extracted from fetched spans. Each premise: one "
        "subject, one predicate, quantities stated exactly as the span states them (or add "
        "transform_note explaining a derivation). span_refs must be span IDs returned by fetch. "
        "Returns premise IDs — your final answer will cite these, so record every fact you "
        "intend to use. Rejected premises include the reason; fix and re-record them."
    )
    input_model = RecordPremisesInput
    max_context_chars = 6000

    def __init__(self, dag: DagStore, spans: SpanStore, documents: DocumentStore):
        self.dag = dag
        self.spans = spans
        self.documents = documents

    async def run(self, input: RecordPremisesInput, ctx: ToolContext) -> tuple[dict, str]:
        sid = ctx.config.session_id
        results: list[dict] = []
        lines: list[str] = []
        span_node_ids: dict[str, str] = {}

        for i, draft in enumerate(input.premises):
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

            problems = check_containment(
                draft.text, [r["text"] for r in span_rows], draft.transform_note
            )
            if problems:
                error = "; ".join(problems)
                results.append({"index": i, "error": error})
                lines.append(f"REJECTED premise {i}: {error}")
                continue

            for row in span_rows:
                ref = row["span_id"]
                if ref not in span_node_ids:
                    doc = self.documents.get_document(row["doc_hash"])
                    span_node_ids[ref] = self.dag.add_node(
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
                self.dag.add_edge(sid, premise_id, span_node_ids[ref], "extracts", edge_payload)
            results.append({"index": i, "premise_id": premise_id})
            lines.append(f"[{premise_id}] recorded: {draft.text}")

        return {"results": results}, "\n".join(lines)
