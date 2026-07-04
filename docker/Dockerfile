ARG DRESSAGE_BASE_PLATFORM=linux/amd64
FROM --platform=${DRESSAGE_BASE_PLATFORM} slimerl/slime:nightly-dev-20260430b

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OPENCODE_VERSION=1.17.4
ARG OPENCLAW_VERSION=2026.6.6
ARG CLAUDE_CODE_CHANNEL=stable

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    PATH=/root/.opencode/bin:/root/.openclaw/bin:/root/.local/bin:/root/.claude/local:${PATH} \
    DRESSAGE_BLACKBOX_RUNNER_MODE=bwrap \
    DRESSAGE_BLACKBOX_BWRAP_BIN=bwrap \
    OPENCODE_BIN=/usr/local/bin/opencode \
    OPENCLAW_BIN=/usr/local/bin/openclaw \
    CLAUDE_CODE_BIN=/usr/local/bin/claude

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      bubblewrap \
      ca-certificates \
      curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://opencode.ai/install | bash -s -- --version "${OPENCODE_VERSION}" && \
    ln -sf "${HOME}/.opencode/bin/opencode" /usr/local/bin/opencode && \
    /usr/local/bin/opencode --version

RUN curl -fsSL https://openclaw.ai/install.sh | bash -s -- --version "${OPENCLAW_VERSION}" --no-onboard && \
    openclaw_resolved="$(command -v openclaw)" && \
    ln -sf "${openclaw_resolved}" /usr/local/bin/openclaw && \
    /usr/local/bin/openclaw --version

RUN curl -fsSL https://claude.ai/install.sh | bash -s "${CLAUDE_CODE_CHANNEL}" && \
    claude_resolved="$(command -v claude || true)" && \
    if [[ -z "${claude_resolved}" && -x "${HOME}/.local/bin/claude" ]]; then \
      claude_resolved="${HOME}/.local/bin/claude"; \
    fi && \
    if [[ -z "${claude_resolved}" && -x "${HOME}/.claude/local/claude" ]]; then \
      claude_resolved="${HOME}/.claude/local/claude"; \
    fi && \
    test -n "${claude_resolved}" && \
    ln -sf "${claude_resolved}" /usr/local/bin/claude && \
    /usr/local/bin/claude --version

WORKDIR /root/Dressage
COPY . /root/Dressage

RUN python -m pip install --no-cache-dir --no-build-isolation -e "." && \
    python -m pip install --no-cache-dir --no-build-isolation ./blackbox_server

CMD ["bash"]
