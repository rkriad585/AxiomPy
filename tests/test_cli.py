"""Tests for the ``axiompy`` CLI tool."""

from unittest.mock import patch

import pytest

from axiompy.cli import (
    cmd_constants,
    cmd_convert,
    cmd_eval,
    cmd_factors,
    cmd_help,
    cmd_info,
    cmd_stats,
    main,
)


class TestCmdInfo:
    def test_info_output(self, capsys):
        cmd_info()
        captured = capsys.readouterr()
        assert "AxiomPy version:" in captured.out
        assert "Python version:" in captured.out
        assert "Active backend:" in captured.out
        assert "precision:" in captured.out

    def test_info_main(self, capsys):
        main(["info"])
        captured = capsys.readouterr()
        assert "AxiomPy version:" in captured.out


class TestCmdDemo:
    def test_demo_runs(self):
        main(["demo"])

    def test_demo_output(self, capsys):
        main(["demo"])
        captured = capsys.readouterr()
        assert "Running:" in captured.out


class TestCmdShell:
    def test_shell_requires_no_arguments(self):
        with patch("code.interact"):
            main(["shell"])

    def test_shell_fallback(self):
        with patch("code.interact"):
            main(["shell"])


class TestCmdEval:
    def test_eval_simple(self, capsys):
        cmd_eval("2+2")
        captured = capsys.readouterr()
        assert "4" in captured.out

    def test_eval_with_expr(self, capsys):
        cmd_eval("3 * 7 + 1")
        captured = capsys.readouterr()
        assert "22" in captured.out

    def test_eval_invalid(self):
        with pytest.raises(SystemExit):
            cmd_eval("not an expr")


class TestCmdFactors:
    def test_factors_84(self, capsys):
        cmd_factors("84")
        captured = capsys.readouterr()
        assert "2 × 2 × 3 × 7" in captured.out

    def test_factors_prime(self, capsys):
        cmd_factors("17")
        captured = capsys.readouterr()
        assert "17" in captured.out

    def test_factors_invalid(self):
        with pytest.raises(SystemExit):
            cmd_factors("abc")


class TestCmdConstants:
    def test_constants_all(self, capsys):
        cmd_constants()
        captured = capsys.readouterr()
        assert "Math:" in captured.out
        assert "PI:" in captured.out
        assert "Total:" in captured.out

    def test_constants_search(self, capsys):
        cmd_constants("mass")
        captured = capsys.readouterr()
        assert "SOLAR_MASS" in captured.out
        assert "EARTH_MASS" in captured.out

    def test_constants_search_no_match(self, capsys):
        cmd_constants("xyznonexistent")
        captured = capsys.readouterr()
        assert "No constants found" in captured.out


class TestCmdConvert:
    def test_convert_42(self, capsys):
        cmd_convert("42")
        captured = capsys.readouterr()
        assert "Binary:" in captured.out
        assert "0b101010" in captured.out
        assert "Roman:" in captured.out
        assert "XLII" in captured.out

    def test_convert_invalid(self):
        with pytest.raises(SystemExit):
            cmd_convert("abc")


class TestCmdStats:
    def test_stats_output(self, capsys):
        cmd_stats()
        captured = capsys.readouterr()
        assert "System Info" in captured.out
        assert "Version:" in captured.out
        assert "Cache Stats" in captured.out
        assert "Hit rate:" in captured.out


class TestCmdHelp:
    def test_help_general(self, capsys):
        cmd_help()
        captured = capsys.readouterr()
        assert "AxiomPy CLI" in captured.out

    def test_help_magic(self, capsys):
        cmd_help("magic")
        captured = capsys.readouterr()
        assert "Decorators:" in captured.out
        assert "pipe" in captured.out

    def test_help_constants(self, capsys):
        cmd_help("constants")
        captured = capsys.readouterr()
        assert "list_all" in captured.out

    def test_help_unknown_topic(self, capsys):
        cmd_help("xyznonexistent")
        captured = capsys.readouterr()
        assert "Unknown topic" in captured.out


class TestArgParsing:
    def test_no_args(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command(self):
        with pytest.raises(SystemExit):
            main(["unknown"])

    def test_help(self):
        with pytest.raises(SystemExit):
            main(["--help"])

    def test_info_on_unknown(self):
        main(["info"])
