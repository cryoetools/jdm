"""Tests for `cryojdm`.cli module."""

from typing import List

import pytest
from typer.testing import CliRunner

import cryojdm
from cryojdm import cli

runner = CliRunner()

output_expected = "Missing option.\n"


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], ""),
        (["--help"], "Usage: "),
        (
            ["--version"],
            f"CryoJDM, version {cryojdm.__version__}\n",
        ),
        (["preprocess"], output_expected),
        (["train"], output_expected),
        (["infer"], output_expected),
    ],
)
def test_command_line_interface(options: List[str], expected: str) -> None:
    """Test the CLI."""
    result = runner.invoke(cli.app, options)
    assert result.exit_code == 0
    assert expected in result.stdout
