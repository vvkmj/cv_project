import assert from 'node:assert/strict';
import test from 'node:test';
import { createReviewApi } from '../server/review.mjs';

const endpoint = '/api/violations';
const jpeg = Uint8Array.of(255, 216, 7, 255, 217);
const auth = { Authorization: 'test-ingest' };
const reviewer = { Authorization: 'test-reviewer' };

function harness() {
  const rows = new Map();
  const images = new Map();
  const state = {
    authenticate: async (request) => {
      const credential = request.headers.get('Authorization');
      if (credential === 'test-ingest') return { id: 'worker', roles: ['traffic_ingest'] };
      if (credential === 'test-reviewer') return { id: 'human-verified', roles: ['traffic_reviewer'] };
      if (request.headers.has('Cookie')) return { id: 'human-verified', roles: ['traffic_reviewer'] };
      return null;
    },
    repository: {
      list: async ({ review_status }) => {
        const items = [...rows.values()].filter((row) => !review_status || row.review_status === review_status);
        return { total: items.length, items };
      },
      find: async (id) => rows.get(id),
      findByKey: async (key) => [...rows.values()].find((row) => row.incident_key === key),
      insert: async (record) => {
        if ([...rows.values()].some((row) => row.incident_key === record.incident_key)) return undefined;
        const row = { ...record, review_status: 'pending' };
        rows.set(row.id, row);
        return row;
      },
      decide: async (id, review) => {
        const row = rows.get(id);
        if (row?.review_status !== 'pending') return undefined;
        Object.assign(row, review, { reviewed_at: new Date().toISOString() });
        return row;
      },
    },
    blobs: () => ({
      get: async (key) => images.get(key),
      set: async (key, bytes) => {
        if (images.has(key)) return { modified: false };
        images.set(key, bytes);
        return { modified: true };
      },
      delete: async (key) => images.delete(key),
    }),
  };
  return { app: createReviewApi(state), images };
}

function submission(override = {}, original = jpeg) {
  const form = new FormData();
  form.set('metadata', JSON.stringify({
    incident_key: 'a'.repeat(64), source: 'Recording 01', camera_id: 'CAM-01', tracking_id: 1,
    motorcycle_tracking_id: 2, violation_type: 'no_helmet', timestamp: 3, frame_number: 90,
    detection_confidence: 0.8, violation_confidence: 0.7, explanation: 'Pending observation', ...override,
  }));
  for (const variant of ['original', 'crop', 'annotated']) {
    form.set(variant, new File([variant === 'original' ? original : jpeg], `${variant}.jpg`, { type: 'image/jpeg' }));
  }
  return { method: 'POST', headers: auth, body: form };
}

test('authorizes, preserves evidence on retries, and attributes one terminal review', async () => {
  const { app, images } = harness();
  assert.equal((await app.request(endpoint)).status, 401);
  assert.equal((await app.request(endpoint, { headers: auth })).status, 403);
  const response = await app.request(`${endpoint}/ingest`, submission());
  assert.equal(response.status, 201);
  const { id } = await response.json();
  const duplicate = await app.request(`${endpoint}/ingest`, submission({}, Uint8Array.of(255, 216, 99, 255, 217)));
  assert.deepEqual(await duplicate.json(), { id, duplicate: true });
  assert.equal(images.size, 3);
  const queue = await app.request(`${endpoint}?review_status=pending`, { headers: reviewer });
  assert.equal(queue.status, 200);
  const [item] = (await queue.json()).items;
  assert.equal(item.review_status, 'pending');
  assert.equal(item.incident_key, undefined);
  assert.equal((await app.request(item.evidence_path)).status, 401);

  const path = `${endpoint}/${id}/status`;
  const body = JSON.stringify({ review_status: 'false_positive', review_note: 'Helmet visible' });
  assert.equal((await app.request(path, { method: 'PATCH', headers: { Cookie: 'session', Origin: 'https://other.invalid' }, body })).status, 403);
  const request = () => app.request(path, { method: 'PATCH', headers: reviewer, body });
  const decisions = await Promise.all([request(), request()]);
  assert.deepEqual(decisions.map((entry) => entry.status).sort(), [200, 409]);
  const reviewed = await decisions.find((entry) => entry.ok).json();
  assert.equal(reviewed.reviewer_id, 'human-verified');
  assert.ok(reviewed.reviewed_at);
  const evidence = await app.request(item.evidence_path, { headers: reviewer });
  assert.equal(evidence.headers.get('cache-control'), 'no-store');
  assert.deepEqual(new Uint8Array(await evidence.arrayBuffer()), jpeg);
});

test('rejects invalid metadata or evidence before creating storage entries', async () => {
  const { app, images } = harness();
  assert.equal((await app.request(`${endpoint}/ingest`, submission({ review_status: 'confirmed' }))).status, 400);
  assert.equal((await app.request(`${endpoint}/ingest`, submission({}, Uint8Array.of(1, 2, 3, 4)))).status, 400);
  assert.equal(images.size, 0);
});
