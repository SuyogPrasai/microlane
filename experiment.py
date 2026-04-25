# Main Code for the pipeline which automates all the pipeline notebooks

"""Microlane pipeline CLI — entry point and global flags."""

from __future__ import annotations

import logging, sys, click
from rich.console import Console

from pipeline.commands.run import run_cmd
from pipeline.commands.continue_ import continue_cmd
from pipeline.commands.batch import batch_cmd
from pipeline.commands.compare import compare_cmd
from pipeline.commands.check import check_cmd
from pipeline.commands.visualize import visualize_cmd
from pipeline.commands.info import info_cmd

