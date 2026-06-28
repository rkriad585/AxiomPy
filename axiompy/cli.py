"""AxiomPy CLI tool.

Run ``axiompy --help`` for available commands.

Examples::

    axiompy shell          Interactive REPL
    axiompy info           Version & config
    axiompy eval "2+3*4"   Evaluate expression
    axiompy factors 84     Prime factorization
    axiompy constants      List built-in constants
    axiompy convert 42     Number conversions
    axiompy stats          Cache & system stats
    axiompy help           Detailed guidance
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from ._facade import Axiom


def cmd_shell():
    """Launch an interactive REPL with ``Axiom`` pre-imported.

    Tries IPython first; falls back to standard interactive Python.
    """
    banner = (
        "AxiomPy interactive shell — use ``A`` as the Axiom facade.\n"
        "  Example: A.Vector([1, 2, 3]) @ A.Matrix([[1,0],[0,1],[0,0]])\n"
        "  Type help(A) for available methods.\n"
    )
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


def cmd_eval(expression: str):
    """Evaluate a math expression using the BasicMath PEMDAS engine.

    Args:
        expression: A math expression string like ``"2 + 3 * 4"``.

    Examples:
        >>> cmd_eval("2 + 3 * 4")  # doctest: +SKIP
        # 2 + 3 * 4 = 14
    """
    try:
        result = Axiom.math.evaluate(expression)
        parts = expression.strip().split()
        if len(parts) == 1:
            print(f"{expression} = {result}")
        else:
            print(f"# {expression} = {result}")
    except Exception as e:
        print(f"Error evaluating '{expression}': {e}", file=sys.stderr)
        sys.exit(1)


def cmd_factors(n: str):
    """Show prime factorization of an integer.

    Args:
        n: Integer to factor.

    Examples:
        >>> cmd_factors("84")  # doctest: +SKIP
        # 84 = 2 × 2 × 3 × 7
    """
    try:
        num = int(n)
        factors = Axiom.number_theory.prime_factors(num)
        formatted = " × ".join(str(f) for f in factors)
        print(f"# {num} = {formatted}")
    except ValueError:
        print(f"Error: '{n}' is not a valid integer.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_constants(search: Optional[str] = None):
    """List or search built-in constants.

    Args:
        search: Optional substring to search for (e.g. ``"mass"``).

    Examples:
        >>> cmd_constants()  # doctest: +SKIP
        >>> cmd_constants("planck")  # doctest: +SKIP
    """
    if search:
        results = Axiom.constants.find(search)
        if not results:
            print(f"No constants found matching '{search}'.")
            return
        print(f"Constants matching '{search}':")
        for name, val in sorted(results.items()):
            print(f"  {name}: {val}")
    else:
        cats = Axiom.constants.list_by_category()
        for cat, consts in cats.items():
            print(f"\n{cat}:")
            for name, val in sorted(consts.items()):
                print(f"  {name}: {val}")
        total = sum(len(v) for v in cats.values())
        print(f"\nTotal: {total} constants")


def cmd_convert(n: str):
    """Show a number in multiple formats.

    Args:
        n: Number to convert (integer).

    Examples:
        >>> cmd_convert("42")  # doctest: +SKIP
    """
    try:
        num = int(n)
        print(f"Number: {num}")
        print(f"  Binary:      {bin(num)}")
        print(f"  Octal:       {oct(num)}")
        print(f"  Hex:         {hex(num)}")
        print(f"  Roman:       {Axiom.magic.roman_numeral(num)}")
        print(f"  In words:    {Axiom.magic.number_to_words(num)}")
        print(f"  Digit sum:   {Axiom.magic.digit_sum(num)}")
        print(f"  Digital root:{Axiom.magic.digital_root(num)}")
        print(f"  Palindrome:  {'yes' if Axiom.magic.is_palindrome(num) else 'no'}")
    except ValueError:
        print(f"Error: '{n}' is not a valid integer.", file=sys.stderr)
        sys.exit(1)


def cmd_stats():
    """Print cache and system statistics.

    Examples:
        >>> cmd_stats()  # doctest: +SKIP
    """
    print("=== System Info ===")
    print(f"  Version:  {__version__}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")

    print("\n=== Cache Stats ===")
    print(f"  Entries:  {Axiom.cache.size}")
    print(f"  Hits:     {Axiom.cache.hits}")
    print(f"  Misses:   {Axiom.cache.misses}")
    print(f"  Hit rate: {Axiom.cache.hit_ratio:.1%}")

    print("\n=== Backend ===")
    backend = getattr(Axiom, "backend", None)
    print(f"  Active:   {type(backend).__name__ if backend else 'unknown'}")


def cmd_help(topic: Optional[str] = None):
    """Show detailed help on an AxiomPy topic.

    Args:
        topic: One of ``math``, ``constants``, ``cache``, ``io``, ``magic``,
               ``cli``, or None for general help.

    Examples:
        >>> cmd_help("cache")  # doctest: +SKIP
    """
    guides = {
        "math": Axiom.math.help,
        "constants": lambda: (
            "Axiom.constants provides built-in math and physics constants.\n"
            "  Use Axiom.constants.list_all() to see all constants.\n"
            "  Use Axiom.constants.find('mass') to search.\n"
            "  Example: Axiom.constants.PI, Axiom.constants.C"
        ),
        "cache": Axiom.cache.help,
        "io": Axiom.io.help,
        "magic": Axiom.magic.help,
    }

    if topic is None:
        print("AxiomPy CLI — available topics:")
        for t in sorted(guides):
            print(f"  axiompy help {t}")
        print("\nAlso try:")
        print("  axiompy shell     — interactive REPL")
        print("  axiompy info      — version & config")
        print("  axiompy eval      — evaluate expression")
        print("  axiompy factors   — prime factorization")
        print("  axiompy constants — list constants")
        print("  axiompy convert   — number conversions")
        print("  axiompy stats     — system & cache stats")
    elif topic in guides:
        print(guides[topic]())
    else:
        print(f"Unknown topic '{topic}'. Try: {', '.join(sorted(guides))}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="axiompy",
        description="AxiomPy CLI — math, constants, cache, and more.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("shell", help="Interactive REPL with Axiom pre-imported")
    sub.add_parser("demo", help="Run through all example scripts")
    sub.add_parser("info", help="Print version, backend, config")

    eval_parser = sub.add_parser("eval", help="Evaluate a math expression")
    eval_parser.add_argument("expression", type=str, help="Expression to evaluate (e.g. '2 + 3 * 4')")

    factors_parser = sub.add_parser("factors", help="Show prime factorization")
    factors_parser.add_argument("n", type=str, help="Integer to factor")

    const_parser = sub.add_parser("constants", help="List or search built-in constants")
    const_parser.add_argument("search", nargs="?", type=str, default=None, help="Optional substring to search")

    convert_parser = sub.add_parser("convert", help="Show number in multiple formats")
    convert_parser.add_argument("n", type=str, help="Number to convert")

    sub.add_parser("stats", help="Show cache and system statistics")

    help_parser = sub.add_parser("help", help="Show detailed guidance on a topic")
    help_parser.add_argument("topic", nargs="?", type=str, default=None, help="Topic: math, constants, cache, io, magic")

    args = parser.parse_args(argv)

    if args.command == "shell":
        cmd_shell()
    elif args.command == "demo":
        cmd_demo()
    elif args.command == "info":
        cmd_info()
    elif args.command == "eval":
        cmd_eval(args.expression)
    elif args.command == "factors":
        cmd_factors(args.n)
    elif args.command == "constants":
        cmd_constants(args.search)
    elif args.command == "convert":
        cmd_convert(args.n)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "help":
        cmd_help(args.topic)


if __name__ == "__main__":
    main()
