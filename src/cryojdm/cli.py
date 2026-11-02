# Copyright (c) CryoJDM
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Console script for CryoJDM."""

import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from cryojdm import __version__

app = typer.Typer()
CORE_PATH = Path(__file__).resolve().parent
run_python = sys.executable
step_1 = str(CORE_PATH / "preprocess.py")
step_2 = str(CORE_PATH / "main.py")


def version_callback(value: bool) -> None:
    """Callback function for the --version option."""
    if value:
        typer.echo(f"CryoJDM, version {__version__}")
        raise typer.Exit()


@app.command("preprocess")
def preprocess(
    config_file: Annotated[
        Optional[str],
        typer.Option("--config_file", help="Path to the pre-processing configuration file."),
    ] = None,
) -> None:
    """Executes the pre-processing step for CryoJDM."""
    if config_file:
        cmd = f"{run_python} {step_1} --config_file {config_file}"
        print(f"Run: {cmd}")
        os.system(cmd)  # noqa: S605
    else:
        print("Missing option.")


@app.command("train")
def train(
    config_file: Annotated[
        Optional[str],
        typer.Option("--config_file", help="Path to the training configuration file."),
    ] = None,
) -> None:
    """Executes the training step for CryoJDM."""
    if config_file:
        cmd = f"{run_python} {step_2} --config_file {config_file}"
        print(f"Run: {cmd}")
        os.system(cmd)  # noqa: S605
    else:
        print("Missing option.")


@app.command("infer")
def infer(
    config_file: Annotated[
        Optional[str],
        typer.Option("--config_file", help="Path to the inference configuration file."),
    ] = None,
    infer: Annotated[
        Optional[str],
        typer.Option(
            "--infer",
            help="Specifies the path or version of the checkpoint to use for inference.",
        ),
    ] = None,
) -> None:
    """Executes the inference step for CryoJDM."""
    if config_file:
        infer = "0" if not infer else infer
        cmd = f"{run_python} {step_2} --config_file {config_file} --infer {infer}"
        print(f"Run: {cmd}")
        os.system(cmd)  # noqa: S605
    else:
        print("Missing option.")


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Console script for CryoJDM."""


if __name__ == "__main__":
    app()  # pragma: no cover
