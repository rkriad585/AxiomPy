FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY axiompy/ axiompy/

RUN uv sync --no-dev

COPY examples/ examples/

CMD ["uv", "run", "python", "examples/linear_algebra.py"]
