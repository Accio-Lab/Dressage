# Runtime dependencies

The example uses the outer Dressage checkout. It does not contain another
Dressage copy or change its default installation. `configs/gpu-runtime-lock.json`
records the reviewed GPU image, dependency versions, source revisions and hashes.

| Component | Pinned version/revision |
| --- | --- |
| Dressage base | `3e3142fe8ea07e4504c3b20a936a4c201a3de44c` |
| Runtime Slime | `ea9819f88caa5e043eb8aea992b0969ffe79aa8e` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087`, with the locked Slime patch |
| CPU Harbor | `0.22.0` |
| TransferQueue | `0.1.10.dev0`, with the request-socket adjustment below |
| Claude Code in task images | `2.1.207` |

The GPU base image is
`huang3eng/dressage@sha256:db8809be3e3164088208f1d2ad127f86ce2623702286b3d00080ae2bb17af058`.
The small derived image adds stable `/opt/dressage-distroll/bin` command paths
and validates installed package versions. Application and training sources are
mounted separately.

## Source preparation

`scripts/gpu/bootstrap_runtime_sources.sh` reads the pinned commits from existing
Git repositories, normally `/root/slime` and `/root/Megatron-LM` in the image.
It creates `runtime-src/slime` and `runtime-src/Megatron-LM` inside the example,
applies only the hash-checked Slime Megatron patch to the new copy, and checks
the locked source hashes. Existing target directories are verified, not replaced.
It performs no network fetch and does not change the repositories it reads.

The official Dressage `slime/` submodule is not silently changed. This recipe
uses its separately pinned runtime revision; compatibility with other revisions
must be verified before changing the lock.

`check_gpu_runtime.py` checks actual imports, reviewed core files and package
versions on each Ray node. Feature commits adding the example are allowed above
the locked Dressage base; changes to the reviewed Dressage/BlackboxServer core
are rejected. All GPU nodes must see the same paths and sources.

## Scoped adaptations

The following are required by this pinned recipe. They are application/dependency
adaptations, not extra trajectory auditing or backups.

| Location | Purpose |
| --- | --- |
| `gpu_adapter/.../protocol_proxy.py` | Unique IDs for independent responses; preserve input usage needed by Claude's context accounting. Network replay reuses the original complete response. |
| `gpu_adapter/.../proxy_source.py` | Reject a TITO incremental candidate when the changed message history is not append-compatible. Build a new lineage with prior history masked as context. |
| `tools/prepare_tq_linger_overlay.py` | Copy the pinned TQ package to a private directory and change request-socket close to `linger=0`, avoiding an indefinite close during error recovery. |
| `gpu_adapter/.../native_tq_actor.py` | Validate real hydrated logprobs, retain source keys until every rank consumes them, and preserve the non-overlapped multi-node gradient synchronization boundary. |

Proxy preparation verifies the exact upstream input hash and writes a new private
entry file. The original checkout remains unchanged. TQ preparation verifies the
original file and the resulting adjustment. Both tools reject unknown source
versions and conflicting existing output. `TQ_SOURCE_ROOT` points to this private
dependency package, not to a trajectory data directory.

The TITO candidate fix and protocol behavior should be reviewed independently for
upstream integration. Once equivalent fixes are available in an admitted upstream
version, remove the corresponding adaptation after regression; do not keep two
versions active or silently bypass the hash check.

## CPU and GPU environments

CPU nodes install only the contract and rollout-service packages. They do not
need Dressage, Ray or Torch. Do not combine this CPU environment with the parent
Dressage Harbor extra, which pins a different Harbor version.

GPU nodes use the outer Dressage source, the example's contract/GPU adapter,
and the pinned Slime/Megatron sources through explicit runtime paths. Optional
TensorBoard figure rendering uses system PyGObject/librsvg and is not a training
dependency.
