FROM ghcr.io/astral-sh/uv:0.12.1-python3.13-trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git just tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group memory-eval
COPY src/ src/
COPY evals/suites/ evals/suites/
CMD ["python", "-m", "evals.suites.memory.runner", "--help"]
