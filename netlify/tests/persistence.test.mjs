import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { PGlite } from '@electric-sql/pglite';
import { drizzle } from 'drizzle-orm/pglite';
import { createRepository } from '../server/index.mjs';

test('the migrated Postgres schema preserves incidents and a single decision across reopen', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'traffic-review-test-'));
  let postgres = new PGlite(directory);
  try {
    const migrations = new URL('../database/migrations/', import.meta.url);
    for (const migration of (await readdir(migrations)).sort()) {
      await postgres.exec(await readFile(new URL(`${migration}/migration.sql`, migrations), 'utf8'));
    }
    const repository = createRepository(() => drizzle({ client: postgres }));
    const incident = { incident_key: 'a'.repeat(64), source: 'fixture', camera_id: 'CAM-01', tracking_id: 1,
      motorcycle_tracking_id: 2, violation_type: 'no_helmet', frame_number: 3, timestamp: 0.1,
      detection_confidence: 0.8, violation_confidence: 0.7, explanation: 'test incident' };
    const { id } = await repository.insert(incident);
    assert.equal(await repository.insert(incident), undefined);
    assert.equal((await repository.list({ review_status: 'pending', page: 1, page_size: 20 })).total, 1);
    const decisions = await Promise.all(['confirmed', 'rejected'].map((review_status) =>
      repository.decide(id, { review_status, reviewer_id: 'reviewer' })));
    assert.equal(decisions.filter(Boolean).length, 1);
    await postgres.close();
    postgres = new PGlite(directory);
    const row = await repository.find(id);
    assert.equal(row.frame_number, 3);
    assert.equal(row.reviewer_id, 'reviewer');
    assert.equal(row.review_status, decisions.find(Boolean).review_status);
    assert.ok(row.reviewed_at);
  } finally {
    await postgres.close();
    await rm(directory, { recursive: true, force: true });
  }
});
