# GPU training adapter

`rollout.py` fetches candidate groups and submits logical slots through
`execution.py`. Admission precedes context preparation; claim precedes route
binding. Group selection prefers full groups and fills with qualified partial
groups. Explicit early stop cancels and settles remaining work before update.

`persistent_train.py` runs synchronous rollout, native training, source-key
release, checkpoint and weight publication. `native_tq_actor.py` hydrates native
references and preserves the multi-node gradient synchronization boundary.
`training_resume.py` restores metadata-only sampler state alongside native
model/optimizer/RNG checkpoints.

`policy_service.py` owns route actions, model request idempotency and verified
generation cleanup. `protocol_proxy.py` provides transparent response identity
and usage semantics. `proxy_source.py` rejects unsafe incremental TITO candidates
before building a new lineage. Tensor export and pure-rollout entry points are
not part of this package.
