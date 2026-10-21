"""Generator: parsiranje odgovora i format konteksta. Bez modela i bez GPU-a."""
from generate import PROMPTS, format_context, parse_response, prompt_fingerprint

CHUNK = {"section": "5.2", "heading_path": "5 > 5.2", "status": "valid", "text": "T", "overridden_by": ["PT-1"]}
FIGURE = {"figure": "Figure 3", "section": "5.6", "heading_path": "H", "status": "valid", "text": "U"}


def test_prompt_fingerprints_frozen():
    """Promena teksta prompta menja heš; stari rezultati tada više ne odgovaraju kodu (v1 je kontrola)."""
    assert {v: prompt_fingerprint(v) for v in PROMPTS} == {
        "v1": "03d3e8f8", "v2a": "61cbff9f", "v2b": "a57a7880", "v2c": "63f1bba3"}


def test_format_context_numbered_v1_unchanged():
    assert format_context([CHUNK, FIGURE], "numbered") == (
        "[1] Source 5.2 | 5 > 5.2 | status: valid; amended by: PT-1\nT\n\n"
        "[2] Source Figure 3 | H | status: valid\nU")


def test_format_context_labeled_has_no_block_numbers():
    out = format_context([CHUNK, FIGURE], "labeled")
    assert out.startswith("Source: 5.2 | Section: 5 > 5.2 | Status: valid; amended by: PT-1\nT")
    assert "\n\n---\n\n" in out and "[1]" not in out


def test_parse_valid_json_with_text_around():
    r = parse_response('Here: {"answer": "220 EUR", "citations": ["5.2"], "abstained": false} done')
    assert (r["answer"], r["citations"], r["parse_error"], r["repaired"]) == ("220 EUR", ["5.2"], False, False)


def test_parse_repairs_unquoted_answer():
    """Greška Qwen 14B: vrednost polja answer bez navodnika."""
    r = parse_response('{"answer": The Orion room has 12 seats., "citations": ["3"], "abstained": false}')
    assert r["answer"] == "The Orion room has 12 seats." and r["repaired"] and not r["parse_error"]


def test_parse_strips_source_prefix_and_header_from_citations():
    r = parse_response('{"answer": "x", "citations": ["Source: 5.2", "6.1 | Section: 6.1 Purchase | Status: valid"]}')
    assert r["citations"] == ["5.2", "6.1"]


def test_parse_keeps_evidence():
    r = parse_response('{"evidence": ["5.2: Singapore: 220"], "answer": "220", "citations": ["5.2"], "abstained": false}')
    assert r["evidence"] == ["5.2: Singapore: 220"]


def test_parse_garbage_marks_error_and_keeps_text():
    r = parse_response("thought thought thought")
    assert r["parse_error"] and r["answer"] == "thought thought thought" and r["citations"] == []
