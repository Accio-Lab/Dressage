ARG CLAUDE_RUNTIME_IMAGE
ARG TMAX_SOURCE_IMAGE
FROM ${CLAUDE_RUNTIME_IMAGE} AS claude_runtime

FROM ${TMAX_SOURCE_IMAGE}

USER root
COPY --from=claude_runtime /usr/local/ /usr/local/

ARG CLAUDE_CODE_VERSION=2.1.207
RUN test "$(command -v node)" = "/usr/local/bin/node" \
    && node -e 'process.stdout.write(process.versions.node)' \
    && ldd /usr/local/bin/node > /tmp/dressage-node.ldd \
    && ! grep -q 'not found' /tmp/dressage-node.ldd \
    && rm /tmp/dressage-node.ldd \
    && test "$(command -v claude)" = "/usr/local/bin/claude" \
    && installed="$(claude --version)" \
    && test "${installed%% *}" = "${CLAUDE_CODE_VERSION}"

ENV PATH="/usr/local/bin:${PATH}"
