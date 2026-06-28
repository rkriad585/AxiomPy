"""AxiomPy CLI tool.

Usage::

    axiompy shell      Interactive REPL with ``Axiom`` pre-imported
    axiompy demo       Run through all example scripts
    axiompy info       Print version, backend, config
    axiompy --help     Show this message
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from ._facade import Axiom


def cmd_shell():
    """Launch an interactive REPL with ``Axiom`` pre-imported."""
    banner = (
        "AxiomPy interactive shell — use ``A`` as the Axiom facade.\n"
        "  Example: A.Vector([1, 2, 3]) @ A.Matrix([[1,0],[0,1],[0,0]])\n"
        "  Type help(A) for available methods.\n"
    )
    # Build a PYTHONSTARTUP script that pre-imports Axiom as A
    startup_code = "from axiompy import Axiom as A\n"
    env = os.environ.copy()
    env["PYTHONSTARTUP"] = ""
    try:
        import IPython  # noqa: F401
        subprocess.run(
            [sys.executable, "-m", "IPython", "--no-banner", "-c", startup_code + "import IPython; IPython.embed()"],
            env=env,
        )
    except ImportError:
        # Fall back to standard interactive Python
        import code
        ns = {"A": Axiom, "__version__": __version__}
        code.interact(banner=banner, local=ns)


def cmd_demo():
    """Run through all example scripts in ``examples/``."""
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    scripts = sorted(examples_dir.glob("*.py"))
    if not scripts:
        print("No example scripts found.")
        return
    for script in scripts:
        print(f"\n{'=' * 60}")
        print(f"Running: {script.name}")
        print(f"{'=' * 60}")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    print(f"  stderr: {line}")
        else:
            print(result.stdout.strip()[:500])


def cmd_info():
    """Print version, backend, and config information."""
    config = Axiom.config
    backend = getattr(Axiom, "backend", None)
    backend_name = type(backend).__name__ if backend else "unknown"
    print(f"AxiomPy version:    {__version__}")
    print(f"Python version:     {sys.version.split()[0]}")
    print(f"Platform:           {sys.platform}")
    print(f"Active backend:     {backend_name}")
    print(f"Config — precision: {config.precision}")
    print(f"Config — dtype:     {config.dtype}")
    print(f"Config — verbose:   {config.verbose}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="axiompy",
        description="AxiomPy CLI — shell, demo, and info.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("shell", help="Interactive REPL with Axiom pre-imported")
    sub.add_parser("demo", help="Run through all example scripts")
    sub.add_parser("info", help="Print version, backend, config")

    args = parser.parse_args(argv)

    if args.command == "shell":
        cmd_shell()
    elif args.command == "demo":
        cmd_demo()
    elif args.command == "info":
        cmd_info()


if __name__ == "__main__":
    main()
