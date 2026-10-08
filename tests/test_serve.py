"""Server-sent events, guardrails, and settings."""

import json
import threading

import pytest

from cqa.errors import ConfigError, RepoNotAllowedError
from cqa.serve.guardrails import RateLimiter, SpendCap, normalize_repo_url, repo_allowed
from cqa.serve.settings import ServerSettings
from cqa.serve.sse import format_event, to_sse
from cqa.trace import Event


def test_an_event_is_one_frame():
    frame = format_event(Event("token", {"text": "line one\nline two"}))
    assert frame.startswith("event: token\ndata: ") and frame.endswith("\n\n")
    (data,) = [line for line in frame.splitlines() if line.startswith("data: ")]
    assert json.loads(data.removeprefix("data: ")) == {"text": "line one\nline two"}


def test_a_failing_stream_ends_with_an_error_event():
    def events():
        yield Event("sources", {"sources": [], "low_confidence": False})
        raise RuntimeError("generator unavailable")

    frames = list(to_sse(events()))
    assert frames[0].startswith("event: sources")
    assert frames[-1].startswith("event: error") and len(frames) == 2


def test_the_rate_limit_is_a_sliding_window():
    limiter = RateLimiter(per_minute=2)
    assert limiter.allow("1.2.3.4", 0.0) and limiter.allow("1.2.3.4", 10.0)
    assert not limiter.allow("1.2.3.4", 20.0)
    assert limiter.allow("5.6.7.8", 20.0)
    assert limiter.allow("1.2.3.4", 61.0)


def test_urls_normalize():
    assert normalize_repo_url("https://user:token@GitHub.com/acme/toy.git/") == "https://github.com/acme/toy"
    assert normalize_repo_url("https://github.com/acme/toy") == "https://github.com/acme/toy"


@pytest.mark.parametrize(
    "url", ["/srv/repos/toy", "file:///srv/repos/toy", "ext::sh -c id", "http://github.com/acme/toy"]
)
def test_only_https_urls_are_repositories(url):
    with pytest.raises(RepoNotAllowedError):
        normalize_repo_url(url)
    assert not repo_allowed(url, None)


def test_the_allowlist_compares_normalized_urls():
    allow = frozenset({"https://github.com/acme/toy"})
    assert repo_allowed("https://github.com/acme/anything", None)
    assert repo_allowed("https://GitHub.com/acme/toy.git", allow)
    assert not repo_allowed("https://github.com/acme/other", allow)
    assert not repo_allowed("https://evil.example/acme/toy.git", allow)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Settings tests see only the variables they set, never the developer's own."""
    for name in (
        "CQA_DATA_DIR",
        "CQA_CONFIG",
        "CQA_RATE_LIMIT_PER_MIN",
        "CQA_DAILY_SPEND_CAP_USD",
        "CQA_REPO_ALLOWLIST",
        "CQA_ALLOW_ANY_REPO",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", "https://github.com/acme/toy, https://github.com/acme/tiger.git")
    monkeypatch.setenv("CQA_RATE_LIMIT_PER_MIN", "3")
    monkeypatch.delenv("CQA_DAILY_SPEND_CAP_USD", raising=False)
    settings = ServerSettings.from_env()
    assert settings.data_dir == tmp_path
    assert settings.repo_allowlist == frozenset(
        {"https://github.com/acme/toy", "https://github.com/acme/tiger"}
    )
    assert settings.rate_limit_per_min == 3 and settings.daily_spend_cap_usd == 1.0


def test_an_empty_allowlist_refuses_to_start(monkeypatch):
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", " , ")
    with pytest.raises(ConfigError, match="CQA_ALLOW_ANY_REPO=1"):
        ServerSettings.from_env()


def test_any_repository_only_on_explicit_request(monkeypatch):
    monkeypatch.setenv("CQA_ALLOW_ANY_REPO", "true")
    assert ServerSettings.from_env().repo_allowlist is None
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", "https://github.com/acme/toy")
    with pytest.raises(ConfigError, match="not both"):
        ServerSettings.from_env()


@pytest.mark.parametrize("value", ["maybe", "2", "on"])
def test_an_unclear_opt_in_is_an_error(monkeypatch, value):
    monkeypatch.setenv("CQA_ALLOW_ANY_REPO", value)
    with pytest.raises(ConfigError, match="CQA_ALLOW_ANY_REPO"):
        ServerSettings.from_env()


def test_defaults(monkeypatch):
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", "https://github.com/acme/toy")
    s = ServerSettings.from_env()
    assert (str(s.data_dir), str(s.config_path)) == ("data", "configs/base.yaml")
    assert (s.rate_limit_per_min, s.daily_spend_cap_usd) == (10, 1.0)


@pytest.mark.parametrize(
    "name, value",
    [
        ("CQA_RATE_LIMIT_PER_MIN", "ten"),
        ("CQA_RATE_LIMIT_PER_MIN", "0"),
        ("CQA_RATE_LIMIT_PER_MIN", "2.5"),
        ("CQA_DAILY_SPEND_CAP_USD", "-1"),
        ("CQA_DAILY_SPEND_CAP_USD", "inf"),
        ("CQA_DAILY_SPEND_CAP_USD", "nan"),
    ],
)
def test_a_bad_number_is_an_error(monkeypatch, name, value):
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", "https://github.com/acme/toy")
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        ServerSettings.from_env()


def test_a_bad_allowlist_entry_is_an_error(monkeypatch):
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", "toy")
    with pytest.raises(ConfigError):
        ServerSettings.from_env()


@pytest.mark.parametrize(
    "url, expected",
    [
        ("HTTPS://GitHub.COM/acme/toy", "https://github.com/acme/toy"),
        ("https://git.example.com:8443/team/repo.git", "https://git.example.com:8443/team/repo"),
        ("  https://github.com/acme/Toy  ", "https://github.com/acme/Toy"),
    ],
)
def test_more_spellings_normalize(url, expected):
    assert normalize_repo_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com",
        "https://github.com/",
        "https://github.com/acme/toy?ref=main",
        "https://github.com/acme/toy#readme",
        "https://github.com/acme/toy/../evil",
        "https://github.com/acme//toy",
        "https://github.com:notaport/acme/toy",
    ],
)
def test_ambiguous_urls_are_refused(url):
    with pytest.raises(RepoNotAllowedError):
        normalize_repo_url(url)
    assert not repo_allowed(url, None)


def test_concurrent_requests_never_exceed_the_limit():
    limiter = RateLimiter(per_minute=25)
    allowed = []
    lock = threading.Lock()

    def hammer():
        for _ in range(20):
            ok = limiter.allow("9.9.9.9", 1.0)
            with lock:
                allowed.append(ok)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sum(allowed) == 25


def test_idle_clients_are_forgotten():
    limiter = RateLimiter(per_minute=5, max_clients=2)
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        limiter.allow(ip, 0.0)
    limiter.allow("4.4.4.4", 100.0)
    assert set(limiter._hits) == {"4.4.4.4"}


@pytest.fixture
def queries(main_db):
    def spend(created_at: str, cost: float | None) -> None:
        with main_db:
            main_db.execute(
                "INSERT INTO queries (index_id, question, config_hash, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("idx", "q", "h", cost, created_at),
            )

    return main_db, spend


def test_the_spend_cap_counts_only_today(queries):
    conn, spend = queries
    cap = SpendCap(daily_usd=1.0)
    assert not cap.exceeded(conn, "2026-10-07")
    spend("2026-10-06T23:59:59.999999Z", 5.0)
    spend("2026-10-07T00:00:00.000000Z", 0.60)
    spend("2026-10-07T12:00:00.000000Z", None)
    assert not cap.exceeded(conn, "2026-10-07")
    spend("2026-10-07T23:59:59.999999Z", 0.40)
    assert cap.exceeded(conn, "2026-10-07")
    spend("2026-10-08T00:00:00.000000Z", 0.10)
    assert not cap.exceeded(conn, "2026-10-08")


def test_a_zero_cap_refuses_everything(queries):
    conn, _ = queries
    assert SpendCap(daily_usd=0.0).exceeded(conn, "2026-10-07")


def test_the_spend_cap_needs_a_date(queries):
    conn, _ = queries
    with pytest.raises(ValueError):
        SpendCap(daily_usd=1.0).exceeded(conn, "today")
