#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${DRESSAGE_REGISTRY_CONTAINER:-dressage-registry}
IMAGE=${DRESSAGE_REGISTRY_IMAGE:-registry:2}
LISTEN_IP=${DRESSAGE_REGISTRY_LISTEN_IP:?CPU Head内网IP待确定，请设置 DRESSAGE_REGISTRY_LISTEN_IP}
PORT=${DRESSAGE_REGISTRY_PORT:-5000}
DATA_ROOT=${DRESSAGE_REGISTRY_DATA_ROOT:?set the persistent Registry data directory}

[[ $(id -u) -eq 0 ]] || { echo "run as root" >&2; exit 1; }
command -v docker >/dev/null
command -v jq >/dev/null
install -d -o root -g root -m 0755 "$DATA_ROOT"

if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -p "${LISTEN_IP}:${PORT}:5000" \
    -e REGISTRY_HTTP_ADDR=0.0.0.0:5000 \
    -v "${DATA_ROOT}:/var/lib/registry" \
    "$IMAGE" >/dev/null
fi

inspection=$(docker container inspect "$CONTAINER")
jq -e \
  --arg image "$IMAGE" \
  --arg listen_ip "$LISTEN_IP" \
  --arg port "$PORT" \
  --arg data_root "$DATA_ROOT" '
    .[0].Config.Image == $image
    and .[0].HostConfig.RestartPolicy.Name == "unless-stopped"
    and .[0].HostConfig.PortBindings["5000/tcp"][0].HostIp == $listen_ip
    and .[0].HostConfig.PortBindings["5000/tcp"][0].HostPort == $port
    and any(.[0].Mounts[];
      .Type == "bind"
      and .Source == $data_root
      and .Destination == "/var/lib/registry"
      and .RW == true)
    and any(.[0].Config.Env[]; . == "REGISTRY_HTTP_ADDR=0.0.0.0:5000")
    and all(.[0].Config.Env[]; startswith("REGISTRY_HTTP_TLS_") | not)
  ' <<<"$inspection" >/dev/null || {
    echo "existing $CONTAINER configuration does not match the expected shared HTTP Registry" >&2
    exit 1
  }

if [[ $(jq -r '.[0].State.Running' <<<"$inspection") != true ]]; then
  docker start "$CONTAINER" >/dev/null
fi
curl -fsS --noproxy '*' "http://${LISTEN_IP}:${PORT}/v2/" >/dev/null
echo "shared HTTP Registry ready: container=$CONTAINER endpoint=${LISTEN_IP}:${PORT} data=$DATA_ROOT"
