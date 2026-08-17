"""Credence model (§2.1, T3): premises carry credence, never presumption.

Root prior per Premise = f(source_tier, corroboration, volatility):
  - source_tier: a small static domain table (overridable per run) — no
    source is presumed true, the tier is a prior, not a verdict.
  - corroboration: independent source CLUSTERS asserting the premise.
    v1 clusters by URL domain (§10's sanctioned cut of minhash/syndication
    clustering): twelve spans from one domain count once; two domains
    corroborate via noisy-OR.
  - volatility: volatile claims take a flat penalty in v1. Real
    recency-decay needs a clock, and a clock read here would break
    byte-identical replay; it lands with calibration (M5).

Propagation:
  - single derivation path: min(parent credences) × edge_penalty(edge_type)
  - independent paths into one node: noisy-OR (1 − Π(1 − pᵢ))
so a long chain from one shaky source visibly decays and genuine
corroboration genuinely raises confidence.

Numbers stay internal (§10.3): render_tier() maps credence to the coarse
labels users see; raw values live on the nodes for calibration later.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# Domain suffix -> prior. Longest matching suffix wins; config merges on top.
DEFAULT_SOURCE_TIERS: dict[str, float] = {
    ".gov": 0.9,
    ".edu": 0.9,
    "arxiv.org": 0.85,
    "nature.com": 0.85,
    "wikipedia.org": 0.8,
    "reuters.com": 0.8,
    "apnews.com": 0.8,
    "medium.com": 0.4,
    "blogspot.com": 0.4,
    "wordpress.com": 0.4,
    "substack.com": 0.4,
    "reddit.com": 0.4,
}
DEFAULT_TIER = 0.6

EDGE_PENALTY: dict[str, float] = {
    "extracts": 1.0,    # mechanically checked at record time
    "deduces": 0.95,
    "induces": 0.85,    # generalization is the leakiest derivation
    "temporal": 0.9,
    "aggregates": 1.0,  # structural merge, no new inference
}

HIGH, MODERATE = 0.8, 0.6


def render_tier(credence: float) -> str:
    """The user-facing register (§10.3): tiers, never raw numbers."""
    if credence >= HIGH:
        return "high"
    if credence >= MODERATE:
        return "moderate"
    return "low"


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def source_tier(url: str, overrides: dict[str, float] | None = None) -> float:
    table = dict(DEFAULT_SOURCE_TIERS)
    if overrides:
        table.update(overrides)
    domain = domain_of(url)
    best_len, best = -1, DEFAULT_TIER
    for suffix, tier in table.items():
        s = suffix.lower().lstrip()
        if (domain == s.lstrip(".") or domain.endswith(s if s.startswith(".") else "." + s)) and len(s) > best_len:
            best_len, best = len(s), tier
    return best


def noisy_or(values: list[float]) -> float:
    p = 1.0
    for v in values:
        p *= 1.0 - v
    return 1.0 - p


@dataclass
class NodeCredence:
    credence: float
    domains: frozenset[str]  # independent source domains transitively behind this node

    @property
    def single_source(self) -> bool:
        return len(self.domains) <= 1


@dataclass
class CredenceReport:
    nodes: dict[str, NodeCredence] = field(default_factory=dict)
    flagged_claims: list[str] = field(default_factory=list)  # below stakes threshold

    def tier(self, node_id: str) -> str:
        return render_tier(self.nodes[node_id].credence)


def compute_credences(
    conn: sqlite3.Connection,
    session_id: str,
    source_tiers: dict[str, float] | None = None,
    stakes_threshold: float = 0.7,
    volatile_penalty: float = 0.85,
    persist: bool = True,
) -> CredenceReport:
    """Recompute credence over the whole session graph (§6 stage 3), bottom-up
    by tier, and (optionally) persist values onto the nodes."""
    nodes: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, tier, payload_json FROM nodes WHERE session_id=?",
        (session_id,),
    ):
        nodes[row["node_id"]] = {
            "type": row["node_type"],
            "tier": row["tier"],
            "payload": json.loads(row["payload_json"]),
        }
    out_edges: dict[str, list[dict]] = {}
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?",
        (session_id,),
    ):
        out_edges.setdefault(row["src_node_id"], []).append(dict(row))

    report = CredenceReport()

    for nid in sorted(nodes, key=lambda n: (nodes[n]["tier"], n)):
        node = nodes[nid]
        if node["type"] == "SourceSpan":
            tier = source_tier(node["payload"]["url"], source_tiers)
            report.nodes[nid] = NodeCredence(tier, frozenset([domain_of(node["payload"]["url"])]))
        elif node["type"] == "Premise":
            report.nodes[nid] = _premise_credence(nid, node, out_edges, report, volatile_penalty)
        elif node["type"] == "Finding":
            report.nodes[nid] = _derived_credence(nid, out_edges, report, joint=True)
        elif node["type"] in ("Synthesis", "ReportClaim"):
            # cited refs are independent supporting paths -> noisy-OR
            report.nodes[nid] = _derived_credence(nid, out_edges, report, joint=False)

    for nid, node in nodes.items():
        if node["type"] == "ReportClaim" and not node["payload"].get("narrative"):
            if report.nodes[nid].credence < stakes_threshold:
                report.flagged_claims.append(nid)
    report.flagged_claims.sort()

    if persist:
        conn.executemany(
            "UPDATE nodes SET credence=? WHERE session_id=? AND node_id=?",
            [(round(nc.credence, 6), session_id, nid) for nid, nc in report.nodes.items()],
        )
    return report


def _premise_credence(
    nid: str, node: dict, out_edges: dict, report: CredenceReport, volatile_penalty: float
) -> NodeCredence:
    """Noisy-OR over independent domain clusters; volatile claims penalized."""
    clusters: dict[str, float] = {}  # domain -> best span tier in cluster
    domains: set[str] = set()
    for e in out_edges.get(nid, []):
        if e["edge_type"] != "extracts" or e["dst_node_id"] not in report.nodes:
            continue
        span = report.nodes[e["dst_node_id"]]
        for d in span.domains:
            clusters[d] = max(clusters.get(d, 0.0), span.credence)
            domains.add(d)
    if not clusters:
        return NodeCredence(0.0, frozenset())
    credence = noisy_or(sorted(clusters.values()))
    if node["payload"].get("claim_class") == "volatile":
        credence *= volatile_penalty
    return NodeCredence(credence, frozenset(domains))


def _derived_credence(
    nid: str, out_edges: dict, report: CredenceReport, joint: bool
) -> NodeCredence:
    """joint=True: one derivation over all parents (min × penalty).
    joint=False: each ref is an independent path (noisy-OR of per-path values)."""
    paths: list[float] = []
    domains: set[str] = set()
    penalties: list[float] = []
    for e in out_edges.get(nid, []):
        if e["edge_type"] == "contradicts" or e["dst_node_id"] not in report.nodes:
            continue
        parent = report.nodes[e["dst_node_id"]]
        penalty = EDGE_PENALTY.get(e["edge_type"], 0.9)
        paths.append(parent.credence * penalty)
        penalties.append(penalty)
        domains.update(parent.domains)
    if not paths:
        return NodeCredence(0.0, frozenset())
    if joint:
        # single derivation path: min of parents, one penalty application
        raw_parents = [p / pen for p, pen in zip(paths, penalties)]
        credence = min(raw_parents) * min(penalties)
    else:
        credence = noisy_or(sorted(paths))
    return NodeCredence(credence, frozenset(domains))
