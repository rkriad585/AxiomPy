# Contributing

Contributions are welcome!  Whether it's a bug fix, new feature, or
documentation improvement, feel free to open an issue or pull request.

## Development setup

```bash
git clone https://github.com/rkriad585/AxiomPy.git
cd AxiomPy
uv sync
```

## Guidelines

- **No new external dependencies.** `numpy` is the sole allowed dependency.
  All other math must be hand-rolled.
- **Follow existing code style.** Type hints, `typing` imports, `numpy` +
  `math` for numeric work, flat class hierarchy.
- **Use the facade pattern.** Add new public methods through the appropriate
  sub-module class (e.g., `LinearAlgebra`, `GraphAnalysis`), then wire it in
  `_facade.py` if it needs a new property.
- **Keep the `Axiom` API stable.** Changes to the public interface
  (`Axiom.Vector`, `Axiom.linalg`, etc.) are breaking changes.
- **No tests currently exist in the repo.** If you add a test, place it in a
  `tests/` directory at the project root.

## Building

```bash
uv build          # builds sdist + wheel
uv run python -c "from axiompy import Axiom; print(Axiom)"  # smoke test
```

## Publishing

A GitHub Release triggers the publish workflow (see
`.github/workflows/publish.yml`).  No manual `twine upload` is needed.

## Project conventions

| Aspect | Convention |
|---|---|
| Python | ≥ 3.9 |
| Package manager | `uv` |
| Build backend | `hatchling` |
| Config | `pyproject.toml` only (no `setup.py`) |
| PyPI name | `axiom-math` (imported as `axiompy`) |
| License | MIT |
