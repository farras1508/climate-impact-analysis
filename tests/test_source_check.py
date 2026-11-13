"""Source checks run offline: HTTP calls are replaced by a fake session."""

import requests

from climate_impact.config import load_datasets
from climate_impact.ingest.source_check import (
    USER_AGENT,
    check_source,
    coverage_matrix,
    resolve_url,
)

HTTP_META = {
    "access": "http",
    "url": "https://example.org/data.csv",
    "priority": "core",
    "questions": ["Q1", "Q3"],
}


class FakeResponse:
    def __init__(self, status_code, headers=None, url="https://final.example.org/data.csv"):
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def close(self):
        pass


class FakeSession:
    def __init__(self, head_status=200, get_status=206, error=None, head_headers=None):
        self.head_status = head_status
        self.get_status = get_status
        self.error = error
        self.head_headers = head_headers or {
            "Content-Length": "1234",
            "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT",
        }
        self.calls = []

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", kwargs))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.head_status, self.head_headers)

    def get(self, url, **kwargs):
        self.calls.append(("GET", kwargs))
        return FakeResponse(self.get_status, {"Content-Range": "bytes 0-0/98765"})


def test_reachable_source_records_metadata():
    status = check_source("demo", HTTP_META, FakeSession())
    assert status.outcome == "ok"
    assert status.status_code == 200
    assert status.size_bytes == "1234"
    assert status.final_url == "https://final.example.org/data.csv"
    assert status.last_modified.startswith("Thu, 01 Oct 2026")


def test_rejected_head_falls_back_to_one_byte_get():
    session = FakeSession(head_status=405)
    status = check_source("demo", HTTP_META, session)
    assert [method for method, _ in session.calls] == ["HEAD", "GET"]
    assert session.calls[1][1]["headers"]["Range"] == "bytes=0-0"
    assert status.outcome == "ok"
    assert status.size_bytes == "98765"


def test_http_error_is_reported_as_failed():
    assert check_source("demo", HTTP_META, FakeSession(head_status=404)).outcome == "failed"


def test_network_error_is_recorded_not_raised():
    status = check_source("demo", HTTP_META, FakeSession(error=requests.ConnectionError()))
    assert status.outcome == "failed"
    assert status.note == "ConnectionError"


def test_every_request_has_timeout_and_user_agent():
    session = FakeSession(head_status=405)
    check_source("demo", HTTP_META, session)
    for _, kwargs in session.calls:
        assert kwargs["timeout"] > 0
        assert kwargs["headers"]["User-Agent"] == USER_AGENT


def test_non_http_source_needs_manual_check_and_makes_no_request():
    session = FakeSession()
    status = check_source("era5", {"access": "cdsapi", "priority": "core"}, session)
    assert status.outcome == "manual"
    assert session.calls == []


def test_url_template_is_filled_with_check_value():
    meta = {
        "access": "http",
        "url_template": "https://x.org/{country}-TAVG-Trend.txt",
        "template_check_value": "indonesia",
    }
    assert resolve_url(meta) == "https://x.org/indonesia-TAVG-Trend.txt"


def test_coverage_matrix_marks_questions():
    lines = coverage_matrix({"demo": HTTP_META})
    assert lines[2] == "| demo | core | ● |  | ● |  |  |"


def test_every_http_entry_in_catalogue_resolves_to_a_url():
    for name, meta in load_datasets().items():
        if meta.get("access") == "http":
            assert resolve_url(meta), (
                f"{name}: no checkable URL (url or url_template + check value)"
            )
