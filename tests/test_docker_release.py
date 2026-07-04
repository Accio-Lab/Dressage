from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_IMAGE_TAG_HELPER = _REPO_ROOT / "docker" / "image_tag.sh"
_DOCKER_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "docker.yml"


def _resolve_image_tag(env: dict[str, str]) -> tuple[str, str, str, str]:
    command = (
        "source docker/image_tag.sh; "
        'printf "%s\\n%s\\n%s\\n%s\\n" '
        '"${DRESSAGE_IMAGE_NAME}" '
        '"${DRESSAGE_IMAGE_REPOSITORY}" '
        '"${DRESSAGE_IMAGE_TAG}" '
        '"${DRESSAGE_LATEST_IMAGE_NAME}"'
    )
    clean_env = {"PATH": os.environ["PATH"], **env}
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=_REPO_ROOT,
        env=clean_env,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = result.stdout.rstrip("\n").split("\n")
    assert len(lines) == 4
    return lines[0], lines[1], lines[2], lines[3]


def test_docker_shell_scripts_are_valid() -> None:
    for script in (
        _IMAGE_TAG_HELPER,
        _REPO_ROOT / "docker" / "build.sh",
        _REPO_ROOT / "docker" / "run.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_default_image_tag_uses_nightly_date_format() -> None:
    image_name, repository, tag, latest = _resolve_image_tag({})

    assert repository == "huang3eng/dressage"
    assert re.fullmatch(r"nightly-dev-\d{8}a", tag)
    assert image_name == f"{repository}:{tag}"
    assert latest == "huang3eng/dressage:latest"


def test_image_tag_date_and_suffix_override_default_tag() -> None:
    image_name, repository, tag, latest = _resolve_image_tag(
        {"IMAGE_TAG_DATE": "20260704", "IMAGE_TAG_SUFFIX": "b"}
    )

    assert repository == "huang3eng/dressage"
    assert tag == "nightly-dev-20260704b"
    assert image_name == "huang3eng/dressage:nightly-dev-20260704b"
    assert latest == "huang3eng/dressage:latest"


def test_image_tag_override_wins_over_generated_tag() -> None:
    image_name, repository, tag, _ = _resolve_image_tag(
        {
            "IMAGE_TAG": "nightly-dev-20260704c",
            "IMAGE_TAG_DATE": "20260704",
            "IMAGE_TAG_SUFFIX": "b",
        }
    )

    assert repository == "huang3eng/dressage"
    assert tag == "nightly-dev-20260704c"
    assert image_name == "huang3eng/dressage:nightly-dev-20260704c"


def test_image_name_complete_override_has_highest_priority() -> None:
    image_name, repository, tag, latest = _resolve_image_tag(
        {"IMAGE_NAME": "my-dressage:dev", "IMAGE_TAG": "ignored"}
    )

    assert image_name == "my-dressage:dev"
    assert repository == "huang3eng/dressage"
    assert tag == "ignored"
    assert latest == "huang3eng/dressage:latest"


def test_version_remains_a_compatibility_alias() -> None:
    image_name, repository, tag, _ = _resolve_image_tag({"VERSION": "v0.1.0"})

    assert repository == "huang3eng/dressage"
    assert tag == "v0.1.0"
    assert image_name == "huang3eng/dressage:v0.1.0"


def test_docker_workflow_publishes_only_from_canonical_repository() -> None:
    workflow = _DOCKER_WORKFLOW.read_text(encoding="utf-8")

    assert "push:\n    branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "github.repository == 'Accio-Lab/Dressage'" in workflow


def test_docker_workflow_uses_docker_hub_release_settings() -> None:
    workflow = _DOCKER_WORKFLOW.read_text(encoding="utf-8")

    assert "secrets.DOCKERHUB_USERNAME" in workflow
    assert "secrets.DOCKERHUB_TOKEN" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "provenance: false" in workflow
    assert "nightly-dev-${tag_date}${suffix}" in workflow
    assert "{a..z}" in workflow
    assert "${IMAGE_REPOSITORY}:latest" in workflow


def test_docker_workflow_uses_expected_actions() -> None:
    workflow = _DOCKER_WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@v7" in workflow
    assert "docker/login-action@v4" in workflow
    assert "docker/setup-buildx-action@v4" in workflow
    assert "docker/build-push-action@v7" in workflow
