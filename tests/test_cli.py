import re

from typer.testing import CliRunner

from geoIR.cli import app

runner = CliRunner()
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    """Strip ANSI sequences that Rich may emit in CI terminals."""
    return ANSI_RE.sub("", output)


def test_top_level_help_lists_supported_and_transitional_commands() -> None:
    result = runner.invoke(app, ["--help"])
    output = plain_output(result.output)

    assert result.exit_code == 0
    assert "encode" in output
    assert "audit" in output
    assert "search" in output
    assert "eval" not in output
    assert "report-save" not in output


def test_search_help_marks_command_as_deprecated() -> None:
    result = runner.invoke(app, ["search", "--help"])
    output = plain_output(result.output)

    assert result.exit_code == 0
    assert "deprecated" in output.lower()


def test_search_exits_with_clear_deprecation_message() -> None:
    result = runner.invoke(app, ["search", "demo-model", "demo-corpus.txt", "--query", "hello"])
    output = plain_output(result.output)

    assert result.exit_code == 2
    assert "deprecated" in output.lower()
    assert "not supported in cli v0" in output.lower()


def test_audit_help_marks_plot_as_experimental() -> None:
    result = runner.invoke(app, ["audit", "--help"])
    output = plain_output(result.output)

    assert result.exit_code == 0
    assert "--plot" in output
    assert "experimental:" in output.lower()
