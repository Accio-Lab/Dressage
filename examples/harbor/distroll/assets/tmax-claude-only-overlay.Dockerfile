ARG NODE_RUNTIME_IMAGE
ARG TMAX_SOURCE_IMAGE
FROM ${NODE_RUNTIME_IMAGE} AS node_runtime

FROM ${TMAX_SOURCE_IMAGE}

USER root
COPY --from=node_runtime /usr/local/ /usr/local/

ARG CLAUDE_CODE_VERSION=2.1.207
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && installed="$(claude --version)" \
    && test "${installed%% *}" = "${CLAUDE_CODE_VERSION}"

ENV PATH="/usr/local/bin:${PATH}"
