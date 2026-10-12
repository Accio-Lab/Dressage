# syntax=docker/dockerfile:1.7
ARG DRESSAGE_BASE_IMAGE
FROM ${DRESSAGE_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DRESSAGE_EXPECTED_REVISION
ARG SLIME_EXPECTED_REVISION
ARG MEGATRON_EXPECTED_REVISION
ARG DRESSAGE_BASE_DIGEST

LABEL org.opencontainers.image.title="dressage-distroll GPU dependency environment" \
      org.opencontainers.image.description="Fixed dependencies for host-mounted Dressage project source" \
      org.opencontainers.image.base.digest="${DRESSAGE_BASE_DIGEST}" \
      io.dressage-distroll.dressage.revision="${DRESSAGE_EXPECTED_REVISION}" \
      io.dressage-distroll.slime.revision="${SLIME_EXPECTED_REVISION}" \
      io.dressage-distroll.megatron.revision="${MEGATRON_EXPECTED_REVISION}"

ENV GPU_RUNTIME_BIN=/opt/dressage-distroll/bin

WORKDIR /opt/dressage-distroll

COPY configs/gpu-runtime-lock.json /opt/dressage-distroll/configs/gpu-runtime-lock.json
COPY tools/check_gpu_image_environment.py /opt/dressage-distroll/tools/check_gpu_image_environment.py
RUN mkdir -p /opt/dressage-distroll/bin && \
    for command_name in python ray torchrun hf; do \
      resolved="$(command -v "${command_name}")"; \
      test -n "${resolved}"; \
      ln -s "${resolved}" "/opt/dressage-distroll/bin/${command_name}"; \
    done && \
    /opt/dressage-distroll/bin/python \
      /opt/dressage-distroll/tools/check_gpu_image_environment.py \
      --lock /opt/dressage-distroll/configs/gpu-runtime-lock.json

WORKDIR /opt/dressage-distroll
CMD ["bash"]
