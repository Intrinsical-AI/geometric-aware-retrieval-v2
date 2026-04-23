from typer.testing import CliRunner

from geoIR.cli import app


runner = CliRunner()


def test_top_level_help_lists_supported_and_transitional_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "encode" in result.output
    assert "audit" in result.output
    assert "search" in result.output
    assert "eval" not in result.output
    assert "report-save" not in result.output


def test_search_help_marks_command_as_deprecated() -> None:
    result = runner.invoke(app, ["search", "--help"])

    assert result.exit_code == 0
    assert "deprecated" in result.output.lower()


def test_search_exits_with_clear_deprecation_message() -> None:
    result = runner.invoke(app, ["search", "demo-model", "demo-corpus.txt", "--query", "hello"])

    assert result.exit_code == 2
    assert "deprecated" in result.output.lower()
    assert "not supported in cli v0" in result.output.lower()


def test_audit_help_marks_plot_as_experimental() -> None:
    result = runner.invoke(app, ["audit", "--help"])

    assert result.exit_code == 0
    assert "--plot" in result.output
    assert "experimental:" in result.output.lower()
