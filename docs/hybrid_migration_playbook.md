# LegacyCoinTrader Hybrid Migration Playbook

## Overview

This playbook describes how to migrate tenant data from LegacyCoinTrader 1.0 into the multi-tenant microservices platform while
operating both stacks in parallel. It complements the new automation delivered in `tools/migration` and the hybrid deployment
options added to `docker-compose` and the Helm charts.

Key objectives:

* Maintain customer availability throughout the transition.
* Provide a reversible migration with clearly defined cutover gates.
* Give the Change Advisory Board (CAB) structured feedback checkpoints and reporting artefacts.

---

## Phased Rollout Plan

| Phase | Target Duration | Scope | CAB Checkpoint | Hybrid Toggles |
| ----- | --------------- | ----- | -------------- | -------------- |
| **Preparation** | Week 0 | Baseline health checks, capture tenant inventory, back up legacy DB | CAB review of migration plan and risk register | All hybrid flags off |
| **Shadow Read** | Week 1 | Enable `HYBRID_MODE_ENABLED` for read-only mirroring on pilot tenants | CAB sign-off on data parity metrics (run `tools/migration/tenant_migration.py audit`) | `HYBRID_MODE_ENABLED=true`, `DUAL_WRITE_STRATEGY=legacy-primary`, `HYBRID_READ_STRATEGY=compare` |
| **Dual Write** | Weeks 2–3 | Expand to majority of tenants with dual write enabled, monitor drift | Weekly CAB checkpoint with diff report and reconciliation sign-off | `DUAL_WRITE_STRATEGY=mirror`, `MIGRATION_DRIFT_TOLERANCE=<per tenant>` |
| **Cutover Read** | Week 4 | Flip read preference to microservices, legacy remains as safety net | CAB go/no-go with rollback decision tree | `HYBRID_READ_STRATEGY=prefer-modern`, `CUTOVER_COMPLETED=false` |
| **Final Cutover** | Week 5 | Disable legacy writes, freeze new onboarding into 1.0 | Emergency CAB call to confirm stabilisation | `CUTOVER_COMPLETED=true`, `CUTOVER_GUARDRAIL_ENABLED=true` |
| **Decommission** | Week 6+ | Archive legacy data, dismantle hybrid infrastructure | CAB closure report | Hybrid flags removed |

Operational guidance:

* **Tenant Cohorts:** Start with low-volume tenants, graduate to strategic accounts, then the long-tail. Maintain a per-cohort
  rollback window (24–48 hours) before progressing.
* **Data Audits:** After each cohort migration, run `make migration-audit` (or directly invoke the audit CLI) and attach the
  generated JSON to the CAB minutes.
* **Performance Monitors:** Use the new docker-compose hybrid environment variables to surface dashboards that report dual-write
  latency and drift counters (`MIGRATION_DRIFT_TOLERANCE`).

---

## Rollback Procedures

1. **Immediate Drift Response**
   * Triggered when `tools/migration/tenant_migration.py audit --fail-on-drift` exits non-zero or when the dual-read shim raises a
     guardrail exception.
   * Actions:
     * Set `CUTOVER_COMPLETED=false` and `DUAL_WRITE_STRATEGY=legacy-only` via Helm values / docker-compose overrides.
     * Revert tenant routing in the API Gateway (`LEGACY_SERVICE_BASE_URL` environment variable) to point exclusively at 1.0.
     * Notify CAB and affected customer segments with the “Incident Head-Up” template below.

2. **Controlled Rollback (Cohort)**
   * Roll back the most recent cohort by replaying the migration plan in reverse:
     1. Pause ingestion for the impacted tenants (`DualWriteStrategy.LEGACY_ONLY`).
     2. Run the schema migrator with `generate-migration-plan` to capture undo scripts (`tenant_migration.py migration-plan ...`).
     3. Execute the rollback SQL against the multi-tenant schema.
     4. Run the audit in verification mode to confirm the legacy dataset is authoritative.

3. **Full Rollback / Abort**
   * If CAB directs a full abort, execute the following within a change window:
     * `docker compose down` (or Helm rollback) for all microservices.
     * Redeploy the legacy monolith via `start_integrated.sh --legacy-only --cutover-ready` (restores legacy-only mode).
     * Archive hybrid configuration artefacts and submit a CAB incident closure report.

Monitoring and alerting updates:

* Both start scripts export hybrid environment variables so runtime logs clearly state the current strategy and tolerance.
* The dual-read shim emits audit callbacks that can be wired to PagerDuty or Slack for rapid detection.

---

## CAB Feedback Loop

* **Weekly CAB Pack:**
  * Migration status dashboard (number of tenants migrated, drift detected, outstanding tasks).
  * Data integrity audit summary (use `tools/migration/post_migration_validator.py` outputs).
  * Feature flag matrix pulled from `tenant_migration.py flags` for transparency.

* **Emergency CAB Call Criteria:**
  * Any failed audit with `fail-on-drift`.
  * More than 10% latency regression on dual writes.
  * Customer-facing incident attributed to hybrid routing.

* **Decision Log:** Store approvals and rollbacks in `docs/cab_decision_log.md` (create if absent) with direct references to the
  generated audit reports.

---

## Customer Communication Templates

### 1. Migration Heads-Up (T-7 Days)

```
Subject: Upcoming LegacyCoinTrader platform enhancements

Hi {{customer_name}},

We are preparing to migrate your LegacyCoinTrader workspace to our new multi-tenant infrastructure during the week of {{date}}.
Trading and monitoring services will remain available; the migration runs in a hybrid mode where we verify all data in parallel.

Key points:
- No action is required on your side.
- We will send a completion notice once the cutover is finalised.
- Support remains available 24/7 via the usual channels.

If you have CAB review requirements, please share them so we can align schedules.

Thank you,
LegacyCoinTrader Operations
```

### 2. Incident Heads-Up (Drift Detected)

```
Subject: Temporary failback to LegacyCoinTrader 1.0

Hi {{customer_name}},

During our hybrid migration validation we detected a data inconsistency affecting your tenant. We have switched trading reads back
to LegacyCoinTrader 1.0 while we reconcile the dataset. There is no impact to order execution.

What happens next:
- Our engineers will re-run the migration validator and share the reconciliation summary.
- We will notify you before re-enabling the multi-tenant path.

Please reach out if you require additional reporting for your CAB.

Regards,
LegacyCoinTrader Operations
```

### 3. Cutover Confirmation

```
Subject: LegacyCoinTrader migration complete

Hi {{customer_name}},

Your tenant has been successfully migrated to the new multi-tenant platform. The legacy stack is now in read-only standby mode and
will be retired after {{retirement_window}}. All monitoring dashboards and API endpoints remain unchanged.

If you observe any discrepancies please contact support referencing ticket “Hybrid-Cutover”.

Best,
LegacyCoinTrader Operations
```

---

## Artefact Checklist

* ✅ SQL schema and migration plans generated via `tools/migration/tenant_migration.py`.
* ✅ Config translation outputs stored under `tools/migration/transformed_configs/` or a secure bucket.
* ✅ Automated audits integrated into CI with `make migration-audit` (see Makefile target).
* ✅ CAB feedback artefacts archived in the `/docs` directory alongside this playbook.
