from axiompy._facade import AxiomPy


class TestVisualization:
    def test_plot_ascii_basic(self, capsys):
        AxiomPy().viz.plot_ascii([0, 1, 2], [0, 1, 4])
        captured = capsys.readouterr()
        assert '*' in captured.out

    def test_plot_ascii_title(self, capsys):
        AxiomPy().viz.plot_ascii([0, 1], [0, 1], title="Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_plot_ascii_labels(self, capsys):
        AxiomPy().viz.plot_ascii([0, 1], [0, 1], xlabel="X", ylabel="Y")
        captured = capsys.readouterr()
        assert "X" in captured.out
        assert "Y" in captured.out

    def test_plot_ascii_extra_series(self, capsys):
        AxiomPy().viz.plot_ascii(
            [0, 1], [0, 1],
            extra_series=[([0, 1], [1, 0], "second")]
        )
        captured = capsys.readouterr()
        assert "second" in captured.out

    def test_plot_histogram(self, capsys):
        AxiomPy().viz.plot_histogram([1, 2, 2, 3, 3, 3, 4, 4, 5], bins=3)
        captured = capsys.readouterr()
        assert '#' in captured.out

    def test_plot_bar(self, capsys):
        AxiomPy().viz.plot_bar(["A", "B", "C"], [3, 7, 5])
        captured = capsys.readouterr()
        assert '#' in captured.out
        assert 'A' in captured.out

    def test_plot_scatter(self, capsys):
        AxiomPy().viz.plot_scatter([0, 1, 2], [0, 1, 4], title="Scatter")
        captured = capsys.readouterr()
        assert '*' in captured.out
