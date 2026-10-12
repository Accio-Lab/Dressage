ARG NODE_RUNTIME_IMAGE
ARG TMAX_SOURCE_IMAGE
FROM ${NODE_RUNTIME_IMAGE} AS node_runtime

FROM ${TMAX_SOURCE_IMAGE}

USER root
COPY --from=node_runtime /usr/local/ /usr/local/

ARG CLAUDE_CODE_VERSION=2.1.207
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       bash build-essential ca-certificates curl git procps python3 python3-pip \
       python3-pytest python3-requests \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version \
    && python3 -c "import pytest, requests" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache /root/.npm

ENV PATH="/usr/local/bin:${PATH}"
