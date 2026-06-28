"""Tests for the ``axiompy`` CLI tool."""

from unittest.mock import patch

import pytest

from axiompy.cli import cmd_info, main


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
