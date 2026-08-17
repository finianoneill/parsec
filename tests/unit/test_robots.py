import httpx
import pytest

from parsec.retrieval.robots import RobotsPolicy, _parse_rsl_license

ROBOTS_TXT = """User-agent: *
Disallow: /private/

User-agent: parsec-research-harness
Disallow: /no-agents/

License: https://example.test/license.xml
"""


def robots_transport(counter: dict, text: str = ROBOTS_TXT, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] = counter.get("calls", 0) + 1
        counter["ua"] = request.headers.get("user-agent")
        return httpx.Response(status, text=text)

    return httpx.MockTransport(handler)


@pytest.fixture
def policy_factory(db, clock):
    def make(counter, **kwargs):
        return RobotsPolicy(
            db, clock, "parsec-research-harness/0.1 (+local-first; polite)",
            transport=robots_transport(counter, **kwargs),
        )

    return make


async def test_disallow_for_our_agent(policy_factory):
    counter: dict = {}
    policy = policy_factory(counter)
    assert not (await policy.check("https://example.test/no-agents/page")).allowed
    assert (await policy.check("https://example.test/allowed/page")).allowed
    assert counter["ua"].startswith("parsec-research-harness")


async def test_specific_group_supersedes_wildcard(policy_factory):
    # correct robots semantics: our named agent group exists, so the * group
    # (which disallows /private/) does not apply to us
    counter: dict = {}
    policy = policy_factory(counter)
    assert (await policy.check("https://example.test/private/page")).allowed


async def test_wildcard_applies_without_specific_group(policy_factory, db, clock):
    from parsec.retrieval.robots import RobotsPolicy

    counter: dict = {}
    policy = RobotsPolicy(
        db, clock, "parsec-research-harness/0.1",
        transport=robots_transport(counter, text="User-agent: *\nDisallow: /private/\n"),
    )
    assert not (await policy.check("https://other.test/private/page")).allowed
    assert (await policy.check("https://other.test/public/page")).allowed


async def test_rsl_license_surfaced(policy_factory):
    counter: dict = {}
    decision = await policy_factory(counter).check("https://example.test/anything")
    assert decision.license_url == "https://example.test/license.xml"


async def test_robots_cached_per_domain(policy_factory, db):
    counter: dict = {}
    policy = policy_factory(counter)
    await policy.check("https://example.test/a")
    await policy.check("https://example.test/b")
    assert counter["calls"] == 1
    assert db.execute("SELECT COUNT(*) FROM robots_cache").fetchone()[0] == 1


async def test_missing_robots_allows(policy_factory):
    counter: dict = {}
    policy = policy_factory(counter, text="not found", status=404)
    decision = await policy.check("https://example.test/anything")
    assert decision.allowed and decision.license_url is None


def test_parse_rsl():
    assert _parse_rsl_license("License: https://x.example/l.xml") == "https://x.example/l.xml"
    assert _parse_rsl_license("User-agent: *\nDisallow:") is None
