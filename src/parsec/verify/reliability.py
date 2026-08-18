"""Learned source reliability via truth-discovery iteration (WS-D.3).

The static tier table stays the PRIOR; this module adjusts it from the
agreement patterns already present in the session's own evidence graph:

- a domain whose premises are independently corroborated by OTHER domains
  earns trust (leave-one-out, so a domain never vouches for itself);
- a domain whose premises are contradicted by well-supported opposing
  premises loses trust.

Risk discipline (plan §4.3 — truth discovery on a small corpus can
self-reinforce): the adjustment is hard-capped at ±`cap` around the prior,
iterations are fixed, and every estimate carries a provenance string so a
learned number is never mistaken for a measured one. Opt-in via
`RunConfig.learned_reliability` until calibration data justifies more.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from parsec.verify.credence import domain_of, noisy_or, source_tier

CAP = 0.1
ITERATIONS = 3
RATE = 0.5


@dataclass(frozen=True)
class ReliabilityEstimate:
    domain: str
    prior: float
    adjusted: float
    provenance: str

    @property
    def delta(self) -> float:
        return self.adjusted - self.prior


def learn_source_reliability(
    conn: sqlite3.Connection,
    session_id: str,
    source_tiers: dict[str, float] | None = None,
    cap: float = CAP,
    iterations: int = ITERATIONS,
) -> dict[str, ReliabilityEstimate]:
    """Truth-discovery over the session graph. Deterministic: fixed iteration
    count, sorted traversal, and each round re-anchors to the prior (the
    learned signal can move a domain at most ±cap, ever)."""
    premise_domains, conflicts = _load_graph(conn, session_id)
    domains = sorted({d for doms in premise_domains.values() for d in doms})
    priors = {d: source_tier(f"https://{d}/", source_tiers) for d in domains}
    reliability = dict(priors)

    premises_of: dict[str, list[str]] = {d: [] for d in domains}
    for pid in sorted(premise_domains):
        for d in sorted(premise_domains[pid]):
            premises_of[d].append(pid)
    opponents: dict[str, set[str]] = {}
    for a, b in conflicts:
        opponents.setdefault(a, set()).add(b)
        opponents.setdefault(b, set()).add(a)

    counted: dict[str, int] = {d: 0 for d in domains}
    for _ in range(iterations):
        truth = {
            pid: noisy_or(sorted(reliability[d] for d in premise_domains[pid]))
            for pid in premise_domains
        }
        for dom in domains:
            signals: list[float] = []
            for pid in premises_of[dom]:
                others = sorted(
                    reliability[d] for d in premise_domains[pid] if d != dom
                )
                contra = max(
                    (truth[q] for q in sorted(opponents.get(pid, ())) if q in truth),
                    default=0.0,
                )
                if not others and contra == 0.0:
                    continue  # no independent signal about this premise
                support = noisy_or(others) if others else 0.5
                # A contradiction hurts only to the extent the OPPOSING side
                # is better supported than this premise's independent backing:
                # a lone contrarian cannot drag down a corroborated consensus.
                excess = max(0.0, contra - support)
                signals.append(support * (1.0 - excess))
            counted[dom] = len(signals)
            if not signals:
                continue
            score = sum(signals) / len(signals)
            delta = max(-cap, min(cap, RATE * (score - 0.5)))
            reliability[dom] = min(1.0, max(0.0, priors[dom] + delta))

    out: dict[str, ReliabilityEstimate] = {}
    for dom in domains:
        delta = reliability[dom] - priors[dom]
        out[dom] = ReliabilityEstimate(
            domain=dom,
            prior=priors[dom],
            adjusted=round(reliability[dom], 6),
            provenance=(
                f"prior {priors[dom]:.2f}, adjusted {delta:+.3f} by observed agreement "
                f"across {counted[dom]} premise(s) (cap ±{cap})"
            ),
        )
    return out


def _load_graph(
    conn: sqlite3.Connection, session_id: str
) -> tuple[dict[str, set[str]], list[tuple[str, str]]]:
    """premise -> supporting domains (via extracts spans), and the
    premise-level contradicts pairs."""
    span_domain: dict[str, str] = {}
    premise_ids: set[str] = set()
    for row in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?", (session_id,)
    ):
        if row["node_type"] == "SourceSpan":
            span_domain[row["node_id"]] = domain_of(json.loads(row["payload_json"])["url"])
        elif row["node_type"] == "Premise":
            premise_ids.add(row["node_id"])

    premise_domains: dict[str, set[str]] = {}
    conflicts: list[tuple[str, str]] = []
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?"
        " ORDER BY src_node_id, dst_node_id",
        (session_id,),
    ):
        if row["edge_type"] == "extracts" and row["dst_node_id"] in span_domain:
            premise_domains.setdefault(row["src_node_id"], set()).add(
                span_domain[row["dst_node_id"]]
            )
        elif (
            row["edge_type"] == "contradicts"
            and row["src_node_id"] in premise_ids
            and row["dst_node_id"] in premise_ids
        ):
            conflicts.append((row["src_node_id"], row["dst_node_id"]))
    return premise_domains, conflicts
