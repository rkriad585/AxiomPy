from axiompy import Axiom, AxiomConfig


class TestConfig:
    def test_default_precision(self):
        cfg = AxiomConfig.load()
        assert cfg.precision == 6
        assert cfg.dtype == "float64"
        assert cfg.verbose is False

    def test_configure_precision(self):
        AxiomConfig.configure(precision=3)
        try:
            cfg = AxiomConfig.load()
            assert cfg.precision == 3
        finally:
            AxiomConfig.reset()

    def test_configure_reset(self):
        AxiomConfig.configure(precision=10)
        AxiomConfig.reset()
        cfg = AxiomConfig.load()
        assert cfg.precision == 6

    def test_facade_config(self):
        assert hasattr(Axiom, 'config')
        assert Axiom.config.precision == 6

    def test_vector_repr_precision(self):
        AxiomConfig.configure(precision=2)
        try:
            v = Axiom.Vector([1.23456, 2.34567])
            s = repr(v)
            assert "1.23" in s
            assert "2.35" in s
        finally:
            AxiomConfig.reset()
