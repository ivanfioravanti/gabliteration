# Copyright © 2025 Gökdeniz Gülmez

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "automate",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand in subcommands:
        submodule = importlib.import_module(f"gabliterate.{subcommand}")
    elif subcommand == "--version":
        from .__init__ import __version__

        print(__version__)
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    