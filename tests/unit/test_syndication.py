"""Content-based syndication clustering (M10, WS-D.1)."""

from parsec.retrieval.embeddings import HashedNgramEmbedder
from parsec.verify.syndication import cluster_spans

EMBED = HashedNgramEmbedder().embed

WIRE = (
    "Water boils at 100 degrees Celsius at sea level, according to the wire "
    "service report distributed to member outlets on Monday."
)
INDEPENDENT = (
    "Our laboratory verified the boiling point of purified water as one "
    "hundred degrees under one atmosphere of pressure."
)


def _span(domain: str, text: str) -> dict:
    return {"domain": domain, "text": text}


def test_same_domain_always_merges():
    clusters = cluster_spans([_span("a.example", WIRE), _span("a.example", INDEPENDENT)], EMBED)
    assert clusters == [[0, 1]]


def test_near_dup_content_merges_across_domains():
    lightly_edited = WIRE + " Editors appended a local note."
    clusters = cluster_spans([_span("a.example", WIRE), _span("b.example", lightly_edited)], EMBED)
    assert clusters == [[0, 1]]


def test_independent_content_stays_separate():
    clusters = cluster_spans([_span("a.example", WIRE), _span("b.example", INDEPENDENT)], EMBED)
    assert clusters == [[0], [1]]


def test_transitive_merge_and_deterministic_order():
    spans = [
        _span("a.example", WIRE),
        _span("b.example", INDEPENDENT),
        _span("c.example", WIRE + " Minor edit."),   # near-dup of 0
        _span("b.example", "Unrelated market commentary about cereal futures."),  # domain of 1
    ]
    clusters = cluster_spans(spans, EMBED)
    assert clusters == [[0, 2], [1, 3]]


def test_empty():
    assert cluster_spans([], EMBED) == []
