FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV PATH="/app/.venv/bin:/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV APP_PORT=7777

COPY --chown=user ./ /app/
RUN chmod +x /app/set_github_user.sh

ENV UV_COMPILE_BYTECODE=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv sync --no-dev --frozen
RUN sudo playwright install --with-deps chromium

ENTRYPOINT ["/app/app/set_github_user.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7777"]
EXPOSE 7777