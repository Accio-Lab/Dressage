# CPU rollout service

Control reserves inventory resources transactionally and activates one immutable
execution. Worker claims are fenced by incarnation and execution identity. Each execution
has a fixed run deadline and a fixed settlement deadline. Harbor executes the
preinstalled Agent, verifier and exact sandbox cleanup. Results persist in an
outbox before delivery and require an explicit RL ACK.

Image Cleaner respects in-use locks, immutable image identity and disk pull
floor. Sandbox Guard enforces individual resource stop-loss without blacklisting
a task. Uncertain executions fail and quarantine resources until cleanup is
proven; healthy Workers can continue.

Dashboard and metadata logs are production observability, not a substitute for
durable result, cleanup or training commit evidence. Verifier execution is part
of the task's reward calculation and remains enabled.
