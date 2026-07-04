# Dressage Docker

This directory builds a local and remote blackbox environment on top of
`slimerl/slime:nightly-dev-20260430b`. The image installs bubblewrap, opencode, openclaw, Claude Code, Codex CLI,
Dressage, and the standalone `dressage-blackbox-server` package from
`blackbox_server/`.

Dressage's default dependencies include E2B support, Ray `2.55.1`, and
transformers `5.3.0`. The current default installer outputs are opencode
`1.17.4`, OpenClaw `2026.6.6`, Claude Code `2.1.191`, and the latest Codex CLI
from the official installer.

## Build

The scripts set `VERSION=v0.1.0` by default and derive `IMAGE_NAME` as
`dressage:${VERSION}`, so the default image tag is `dressage:v0.1.0`.

```bash
docker/build.sh
```

Override the version if needed:

```bash
VERSION=0.2.0 docker/build.sh
```

Override the tag if needed:

```bash
IMAGE_NAME=my-dressage:dev docker/build.sh
```

The default build platform is `linux/amd64`, matching the published slime GPU
image. Override it only if you have a compatible base image:

```bash
DOCKER_PLATFORM=linux/amd64 docker/build.sh
```

## Run

```bash
docker/run.sh
```

The run script uses `--gpus all --network host --ipc host --privileged` by
default. `--privileged` is required for the containerized bubblewrap runner in
the default configuration.

The image includes this repository at `/root/Dressage`. The run script also
mounts the host repository at `/root/Dressage`. Model and checkpoint files are
mounted when `HOST_CHECKPOINT_DIR` or `BASE_FOLDER` points to an existing host
directory:

```bash
HOST_CHECKPOINT_DIR=/data/checkpoints docker/run.sh
```

Pass a command after the script to run it inside the container:

```bash
docker/run.sh bash
docker/run.sh python -m blackbox_server.main
```

## Verify Tools

Inside the container:

```bash
bwrap --version
opencode --version
openclaw --version
claude --version
codex --version
python -c "import ray; print(ray.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import e2b; print('e2b ok')"
blackbox-server --help
```

The Codex blackbox backend runs with a sandbox-local `CODEX_HOME` created by
BlackboxServer. Do not mount host `~/.codex` into blackbox slots; that directory
can contain real Codex authentication state.

Start BlackboxServer and check health:

```bash
BBS_HOST=127.0.0.1 BBS_PORT=23456 python -m blackbox_server.main
curl http://127.0.0.1:23456/health
```

Start a local blackbox bwrap pool after configuring Ray and Dressage
environment variables:

```bash
dressage-local-bwrap-start
dressage-local-bwrap-status
dressage-local-bwrap-stop
```
