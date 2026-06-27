# AxiomPy — AGENTS.md

## Architecture

- **Multi-module package:** `axiompy/` is split by domain into `vector.py`, `matrix.py`, `graph.py`, `linalg.py`, `autodiff.py`, `number_theory.py`, `electromagnetism.py`, `visualization.py`, `stats.py`. The `_facade.py` wires the `Axiom` singleton; `_base.py` holds shared types/errors.
- **Public API:** `from axiompy import Axiom` — a singleton `AxiomPy` façade instance providing `Vector`, `Matrix`, `Graph`, and sub-module properties (`linalg`, `graph_analysis`, `number_theory`, `autodiff`, `electromagnetism`, `viz`, `stats`).
- **Dependency:** `numpy>=1.20.0` is the sole external dependency. All math beyond numpy arrays is hand-rolled.
- **Python compatibility:** `>=3.9` (matches numpy support). CI tests 3.9–3.11.

## Key commands / workflow

| Action | Command |
|---|---|
| Install dev | `uv sync` |
| Run demo | `uv run python example.py` |
| Build | `uv build` |
| Publish | Triggered by GitHub Release — see `.github/workflows/publish.yml` |

- **Package managed with `uv`.** All metadata in `pyproject.toml` (PEP 621). No `setup.py`.
- **PyPI package name differs from import name:** installed as `pip install axiom-math`, imported as `axiompy`.
- **No tests anywhere in the repo.** No test framework, runner, or directory.
- **No linter, formatter, or type checker** config present (no `.pylintrc`, `ruff.toml`, `mypy.ini`, `setup.cfg`, etc.).
- **No CI on push/PR** — only a publish workflow that fires on GitHub Release.
- **`.gitignore`** exists — covers `__pycache__/`, `*.egg-info/`, `dist/`, `.venv/`, etc.
- **No archived version snapshots** — git history is the canonical archive of past versions.

## Conventions to follow

- Only code under `axiompy/` is shipped in the wheel; repo-root files are for development/docs.
- Use numpy for array operations; avoid adding new external dependencies.
- Follow existing style: flat class structure, type hints from `typing`, `numpy` + `math` imports.
- `AxiomError` is the custom exception base class, defined in `_base.py`.
