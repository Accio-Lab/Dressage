#!/usr/bin/env python3
"""Read-only V4 installation gate; preserve old records and refuse live ownership."""

import asyncio
import os

import asyncpg


async def main():
    connection = await asyncpg.connect(os.environ["ROLLOUT_DATABASE_URL"], timeout=10)
    try:
        checks = {
            "rollout_attempts": "state IN ('QUEUED','ASSIGNED','PREPARING','WAITING_ROUTE','AGENT_RUNNING','VERIFYING','UNKNOWN')",
            "rollout_groups": "state NOT IN ('ACKED','ZERO_GRAD','CANCELLED','FAILED')",
            "rollout_outbox": "consumed=false",
            "rollout_admission_batches": "state='QUEUED'",
        }
        blocked = {}
        for table, predicate in checks.items():
            if await connection.fetchval("SELECT to_regclass($1)", "public." + table):
                count = await connection.fetchval(
                    f"SELECT count(*) FROM public.{table} WHERE {predicate}"
                )
                if count:
                    blocked[table] = count
        for schema in ("rollout_v3", "rollout_v4"):
            if not await connection.fetchval(
                "SELECT to_regclass($1)", schema + ".executions"
            ):
                continue
            count = await connection.fetchval(f"""SELECT count(*) FROM {schema}.executions
                WHERE NOT cleanup_verified OR (body->'result' IS DISTINCT FROM 'null'::jsonb
                AND NOT (body->>'result_acknowledged')::boolean)""")
            if count:
                blocked[schema + ".executions"] = count
        if blocked:
            raise SystemExit(
                f"V4 upgrade blocked by outstanding ownership: {blocked}; drain using the running version first"
            )
        print(
            "database ownership is drained; also confirm GPU submissions and old Worker/Policy processes are stopped"
        )
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
