import re

from typer.testing import CliRunner

from geoIR.cli import app

runner = CliRunner()
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    """Strip ANSI sequences that Rich may emit in CI terminals."""
    return ANSI_RE.sub("", output)


def test_top_level_help_lists_only_supported_commands() -> None:
    result = runner.invoke(app, ["--help"])
    output = plain_output(result.output)

    assert result.exit_code == 0
    assert "encode" in output
    assert "audit" in output
    assert "search" not in output
    assert "eval" not in output
    assert "report-save" not in output


def test_audit_help_marks_plot_as_experimental() -> None:
    result = runner.invoke(app, ["audit", "--help"])
    output = plain_output(result.output)

    assert result.exit_code == 0
    assert "--plot" in output
    assert "experimental:" in output.lower()
