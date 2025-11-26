FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git libglib2.0-0t64 libnspr4 libnss3 libdbus-1-3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libxkbcommon0 libatspi2.0-0t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libcairo2 libpango-1.0-0 libasound2t64 && \
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
RUN python -m playwright install chromium

ENTRYPOINT ["/app/app/set_github_user.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7777"]
EXPOSE 7777