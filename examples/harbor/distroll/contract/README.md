# Execution contract

`agentic-rollout/v4` defines single-execution admission and Policy context identity.
CPU operations are capacity, reserve, activate, query, cancel and result ACK.
Internal Worker operations cover registration, heartbeat, claim, phase transitions,
fixed execution deadlines, completion, cleanup and recovery evidence.
Worker heartbeats remain independent.

RL groups and weight-version orchestration belong to the GPU side. CPU
`client_scope` is an opaque diagnostic label, not an RL owner or timed session.
GPU Head and CPU Head communicate over direct HTTP. Retries reuse the same
identity; no transport success implies training commit.
