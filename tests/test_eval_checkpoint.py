from tools.diagnostics.eval_checkpoint import parse_json_from_text


def test_parse_json_from_text_recovers_trailing_payload() -> None:
    text = "noise before\n{\n  \"status\": \"completed\",\n  \"metrics\": {\"success_rate\": 1.0}\n}\n"
    payload = parse_json_from_text(text)
    assert payload == {"status": "completed", "metrics": {"success_rate": 1.0}}
