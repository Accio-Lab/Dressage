ARG NODE_RUNTIME_IMAGE
FROM ${NODE_RUNTIME_IMAGE}

ARG CLAUDE_CODE_VERSION=2.1.207
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && installed="$(claude --version)" \
    && test "${installed%% *}" = "${CLAUDE_CODE_VERSION}"

ENV PATH="/usr/local/bin:${PATH}"
