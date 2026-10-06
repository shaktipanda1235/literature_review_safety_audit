from ui.review_app import normalize_api_url, parse_sse


def test_parse_sse_yields_progress_and_status_events():
    events = list(
        parse_sse(
            [
                "id: 0",
                "event: progress",
                'data: {"retrieve_node": {"documents": []}}',
                "",
                "event: status",
                'data: {"status": "pending_review"}',
                "",
            ]
        )
    )

    assert events == [
        ("progress", '{"retrieve_node": {"documents": []}}'),
        ("status", '{"status": "pending_review"}'),
    ]


def test_normalize_api_url_trims_whitespace_and_trailing_slashes():
    assert normalize_api_url("  http://127.0.0.1:8000/// ") == "http://127.0.0.1:8000"