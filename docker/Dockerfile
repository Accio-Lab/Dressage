ARG DRESSAGE_BASE_PLATFORM=linux/amd64
FROM --platform=${DRESSAGE_BASE_PLATFORM} slimerl/slime:nightly-dev-20260430b

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OPENCODE_VERSION=1.17.4
ARG OPENCLAW_VERSION=2026.6.6
ARG CLAUDE_CODE_VERSION=2.1.191
ARG CODEX_VERSION=0.142.5

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    PATH=/root/.opencode/bin:/root/.openclaw/bin:/root/.local/bin:/root/.claude/local:${PATH} \
    DRESSAGE_BLACKBOX_RUNNER_MODE=bwrap \
    DRESSAGE_BLACKBOX_BWRAP_BIN=bwrap \
    OPENCODE_BIN=/usr/local/bin/opencode \
    OPENCLAW_BIN=/usr/local/bin/openclaw \
    CLAUDE_CODE_BIN=/usr/local/bin/claude \
    CODEX_BIN=/usr/local/bin/codex

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

RUN curl -fsSL https://claude.ai/install.sh | bash -s "${CLAUDE_CODE_VERSION}" && \
    claude_resolved="$(command -v claude || true)" && \
    if [[ -z "${claude_resolved}" && -x "${HOME}/.local/bin/claude" ]]; then \
      claude_resolved="${HOME}/.local/bin/claude"; \
    fi && \
    if [[ -z "${claude_resolved}" && -x "${HOME}/.claude/local/claude" ]]; then \
      claude_resolved="${HOME}/.claude/local/claude"; \
    fi && \
    test -n "${claude_resolved}" && \
    ln -sf "${claude_resolved}" /usr/local/bin/claude && \
    /usr/local/bin/claude --version && \
    test "$(/usr/local/bin/claude --version | awk '{print $1}')" = "${CLAUDE_CODE_VERSION}"

RUN curl -fsSL https://chatgpt.com/codex/install.sh | \
      CODEX_NON_INTERACTIVE=1 CODEX_INSTALL_DIR=/usr/local/bin sh -s -- --release "${CODEX_VERSION}" && \
    codex_resolved="$(command -v codex)" && \
    codex_real="$(readlink -f "${codex_resolved}")" && \
    if [[ "${codex_real}" != "/usr/local/bin/codex" || -L /usr/local/bin/codex ]]; then \
      rm -f /usr/local/bin/codex && \
      install -m 0755 "${codex_real}" /usr/local/bin/codex; \
    fi && \
    test -x /usr/local/bin/codex && \
    test ! -L /usr/local/bin/codex && \
    /usr/local/bin/codex --version | grep -F "${CODEX_VERSION}"

WORKDIR /root/Dressage
COPY . /root/Dressage

RUN python -m pip install --no-cache-dir --no-build-isolation -e "." && \
    python -m pip install --no-cache-dir --no-build-isolation ./blackbox_server

CMD ["bash"]
