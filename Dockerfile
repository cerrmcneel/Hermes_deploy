FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md /app/
COPY src /app/src

# Install package and dependencies
RUN uv pip install --system -e .

# Create workspace directory for user execution
RUN mkdir -p /workspace && chown -R 1000:1000 /workspace /app

# Set non-root user for execution safety
USER 1000:1000

ENV WORKSPACE_DIR=/workspace
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["hermes-loop"]
CMD ["--help"]
