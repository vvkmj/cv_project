import { getDeployStore, getStore } from '@netlify/blobs';
import { getIdentityConfig, getUser } from '@netlify/identity';
import { getContext } from '@netlify/server';
import { and, count, desc, eq, gte, sql } from 'drizzle-orm';
import { drizzle } from 'drizzle-orm/netlify-db';
import * as schema from '../../db/schema.ts';
import { createReviewApi } from './review.mjs';

const { incidents } = schema;
let database;
const db = () => database ??= drizzle({ schema });

export const createRepository = (connect) => ({
  async list({ page, page_size, camera_id, review_status, violation_type, min_confidence }) {
    const where = and(
      camera_id && eq(incidents.camera_id, camera_id),
      review_status && eq(incidents.review_status, review_status),
      violation_type && eq(incidents.violation_type, violation_type),
      min_confidence !== undefined ? gte(incidents.violation_confidence, min_confidence) : undefined,
    );
    const [[result], items] = await Promise.all([
      connect().select({ total: count() }).from(incidents).where(where),
      connect().select().from(incidents).where(where).orderBy(desc(incidents.created_at), desc(incidents.id))
        .offset((page - 1) * page_size).limit(page_size),
    ]);
    return { total: result.total, items };
  },
  async find(id) {
    const [row] = await connect().select().from(incidents).where(eq(incidents.id, id)).limit(1);
    return row;
  },
  async findByKey(key) {
    const [row] = await connect().select({ id: incidents.id }).from(incidents).where(eq(incidents.incident_key, key)).limit(1);
    return row;
  },
  async insert(record) {
    const [row] = await connect().insert(incidents).values(record).onConflictDoNothing({ target: incidents.incident_key })
      .returning({ id: incidents.id });
    return row;
  },
  async decide(id, decision) {
    const [row] = await connect().update(incidents).set({ ...decision, reviewed_at: sql`now()` })
      .where(and(eq(incidents.id, id), eq(incidents.review_status, 'pending'))).returning();
    return row;
  },
});

async function authenticate(request) {
  const authorization = request.headers.get('Authorization');
  if (!authorization) return getUser();
  if (!/^Bearer \S+$/i.test(authorization)) return null;
  const identityUrl = getIdentityConfig()?.url ?? (process.env.URL && new URL('/.netlify/identity', process.env.URL).href);
  if (!identityUrl) return null;
  const result = await fetch(`${identityUrl.replace(/\/$/, '')}/user`, {
    headers: { Authorization: authorization }, redirect: 'error', signal: AbortSignal.timeout(5000),
  });
  if (!result.ok) return null;
  const user = await result.json();
  return { id: user.id, roles: Array.isArray(user.app_metadata?.roles) ? user.app_metadata.roles : [] };
}

function blobs() {
  const options = { name: 'traffic-incident-evidence', consistency: 'strong' };
  return getContext().deploy.context === 'production' ? getStore(options) : getDeployStore(options);
}

export const config = { path: ['/api/violations', '/api/violations/*'] };
export default createReviewApi({ authenticate, repository: createRepository(db), blobs });
