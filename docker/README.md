# Dressage Docker

This directory builds a local and remote blackbox environment on top of
`slimerl/slime:nightly-dev-20260430b`. The image installs bubblewrap, opencode, openclaw, Claude Code, Codex CLI,
Dressage, and the standalone `dressage-blackbox-server` package from
`blackbox_server/`.

Dressage's default dependencies include E2B support, Ray `2.55.1`, and
transformers `5.3.0`. The current default installer outputs are opencode
`1.17.4`, OpenClaw `2026.6.6`, Claude Code `2.1.191`, and Codex CLI
`0.142.5`.

## Build

The scripts default to the Docker Hub repository `huang3eng/dressage` and use a
Slime-style UTC date tag: `nightly-dev-YYYYMMDDa`. For example, a build on
2026-07-04 defaults to `huang3eng/dressage:nightly-dev-20260704a`.

```bash
docker/build.sh
```

Override the generated date tag if needed:

```bash
IMAGE_TAG_DATE=20260704 IMAGE_TAG_SUFFIX=b docker/build.sh
IMAGE_TAG=nightly-dev-20260704b docker/build.sh
```

Override the full image name if needed:

```bash
IMAGE_NAME=my-dressage:dev docker/build.sh
```

`VERSION` remains available as a compatibility alias for older local workflows:

```bash
VERSION=v0.1.0 docker/build.sh
```

Push a local build to Docker Hub by setting `PUSH=1`. Add `TAG_LATEST=1` to
also tag and push `huang3eng/dressage:latest`:

```bash
PUSH=1 docker/build.sh
PUSH=1 TAG_LATEST=1 docker/build.sh
```

The default build platform is `linux/amd64`, matching the published slime GPU
image. Override it only if you have a compatible base image:

```bash
DOCKER_PLATFORM=linux/amd64 docker/build.sh
```

## Automatic Publishing

Docker publishing is owned by the canonical repository
`Accio-Lab/Dressage`. Merging a pull request into `Accio-Lab/Dressage:main`
creates the push event that builds and publishes the image. The workflow is
guarded with `github.repository == 'Accio-Lab/Dressage'`, so the same workflow
file can exist in forks without publishing images from fork pushes or manual
runs.

The workflow publishes `huang3eng/dressage:nightly-dev-YYYYMMDDa/b/c...` and
updates `huang3eng/dressage:latest`. When multiple releases happen on the same
UTC day, it queries Docker Hub and uses the first unused suffix from `a` to `z`.
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` must be configured as repository
secrets in `Accio-Lab/Dressage`, or as organization secrets available to that
repository.

## Run

```bash
docker/run.sh
```

The run script uses the same image-name resolution as the build script. Use
`IMAGE_TAG=latest docker/run.sh` to run the latest published image, or pin a
dated tag with `IMAGE_TAG=nightly-dev-20260704a`.

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

The Docker image pins Codex CLI to `0.142.5`.

The canonical blackbox backend entrypoints are all exposed under
`/usr/local/bin`: `opencode`, `openclaw`, `claude`, and `codex`.

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
