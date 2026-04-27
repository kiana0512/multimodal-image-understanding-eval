from mm_eval.evaluation.report_builder import build_markdown_report

def test_report_builder_generates_markdown():
    md=build_markdown_report(title="Demo Report")
    assert "# Demo Report" in md
    assert "placeholder" in md
