CREATE SCHEMA IF NOT EXISTS rollout_v4;
CREATE TABLE IF NOT EXISTS rollout_v4.workers (
    worker_id text PRIMARY KEY,
    registration jsonb NOT NULL,
    state text NOT NULL,
    last_heartbeat timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS rollout_v4.executions (
    execution_id text PRIMARY KEY,
    client_scope text NOT NULL,
    state text NOT NULL,
    worker_id text,
    cleanup_verified boolean NOT NULL DEFAULT false,
    body jsonb NOT NULL,
    reservation_keys text[] NOT NULL DEFAULT '{}',
    claim_request_id text,
    claim jsonb,
    lease_expires_at timestamptz,
    sequence_no bigserial UNIQUE
);
ALTER TABLE rollout_v4.workers ADD COLUMN IF NOT EXISTS active_execution_ids jsonb NOT NULL DEFAULT '[]';
CREATE INDEX IF NOT EXISTS executions_scope_idx ON rollout_v4.executions(client_scope, execution_id);
CREATE INDEX IF NOT EXISTS executions_worker_idx ON rollout_v4.executions(worker_id, state);
CREATE UNIQUE INDEX IF NOT EXISTS executions_claim_idx
    ON rollout_v4.executions(worker_id, (claim->>'worker_incarnation'), claim_request_id)
    WHERE claim_request_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS rollout_v4.recovery_audit (
    execution_id text NOT NULL,
    evidence_digest text NOT NULL,
    recovered_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (execution_id, evidence_digest)
);
