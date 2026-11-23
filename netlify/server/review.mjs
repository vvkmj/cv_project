import { Hono } from 'hono';
import { bodyLimit } from 'hono/body-limit';
import { z } from 'zod';

const prefix = '/api/violations';
const variants = { original: 'evidence_path', crop: 'crop_path', annotated: 'annotated_evidence_path' };
const maxImage = 5 * 1024 * 1024;
const uuid = z.uuid();
const status = z.enum(['pending', 'confirmed', 'false_positive', 'rejected']);
const trackingId = z.int().min(1).max(2147483647);
const confidence = z.number().min(0).max(1);
const label = z.string().trim().regex(/^[\p{L}\p{N} ._()-]+$/u).min(1).max(120);
const metadata = z.strictObject({
  incident_key: z.string().regex(/^[a-f0-9]{64}$/), source: label, camera_id: label,
  tracking_id: trackingId, motorcycle_tracking_id: trackingId.optional(),
  violation_type: z.enum(['no_helmet', 'multiple_riders', 'wrong_way']),
  timestamp: z.number().min(0), frame_number: z.int().min(0).max(2147483647),
  detection_confidence: confidence, violation_confidence: confidence,
  explanation: z.string().trim().min(1).max(1500),
});
const decision = z.strictObject({
  review_status: status.exclude(['pending']), review_note: z.string().trim().max(2000).optional(),
});
const filters = z.object({
  page: z.coerce.number().int().min(1).max(1000).default(1),
  page_size: z.coerce.number().int().min(1).max(100).default(20),
  review_status: status.optional(),
  camera_id: label.optional(),
  violation_type: metadata.shape.violation_type.optional(),
  min_confidence: z.coerce.number().min(0).max(1).optional(),
});

function publicIncident({ incident_key, ...row }) {
  return { ...row, ...Object.fromEntries(Object.entries(variants).map(([variant, field]) =>
    [field, `${prefix}/${row.id}/evidence/${variant}`])) };
}

async function imageBytes(part) {
  if (!(part instanceof File) || part.type !== 'image/jpeg' || part.size < 4 || part.size > maxImage) return null;
  const bytes = new Uint8Array(await part.arrayBuffer());
  return bytes[0] === 255 && bytes[1] === 216 && bytes.at(-2) === 255 && bytes.at(-1) === 217 ? bytes : null;
}

export function createReviewApi({ authenticate, repository, blobs }) {
  const app = new Hono().basePath(prefix);
  app.onError((error, context) => {
    console.error('Traffic review API request failed', error.name);
    return context.json({ detail: 'Review service unavailable' }, 503);
  });
  app.use('*', async (context, next) => {
    context.header('Cache-Control', 'no-store');
    context.header('X-Content-Type-Options', 'nosniff');
    const user = await authenticate(context.req.raw);
    if (!user?.id) return context.json({ detail: 'Authentication required' }, 401);
    const method = context.req.method;
    const requiredRole = method === 'POST' ? 'traffic_ingest' : 'traffic_reviewer';
    if (!user.roles?.includes(requiredRole)) return context.json({ detail: 'Insufficient permissions' }, 403);
    if (method !== 'GET' && method !== 'HEAD' && !context.req.header('Authorization') &&
        context.req.header('Origin') !== new URL(context.req.url).origin) {
      return context.json({ detail: 'Same-origin request required' }, 403);
    }
    context.set('user', user);
    await next();
  });

  app.get('/', async (context) => {
    const query = filters.safeParse(context.req.query());
    if (!query.success) return context.json({ detail: 'Invalid review filters' }, 400);
    const { page, page_size } = query.data;
    const { total, items } = await repository.list(query.data);
    return context.json({ total, page, page_size, items: items.map(publicIncident) });
  });

  app.get('/:id', async (context) => {
    const id = uuid.safeParse(context.req.param('id'));
    const row = id.success ? await repository.find(id.data) : null;
    return row ? context.json(publicIncident(row)) : context.json({ detail: 'Incident not found' }, 404);
  });

  app.get('/:id/evidence/:variant', async (context) => {
    const id = uuid.safeParse(context.req.param('id'));
    const variant = context.req.param('variant');
    if (!id.success || !Object.hasOwn(variants, variant) || !await repository.find(id.data)) {
      return context.json({ detail: 'Evidence not found' }, 404);
    }
    const stream = await blobs().get(`${id.data}/${variant}`, { type: 'stream' });
    return stream ? context.body(stream, 200, { 'Content-Type': 'image/jpeg', 'Content-Security-Policy': 'sandbox' }) :
      context.json({ detail: 'Evidence not found' }, 404);
  });

  app.patch('/:id/status', bodyLimit({ maxSize: 8192 }), async (context) => {
    const id = uuid.safeParse(context.req.param('id'));
    if (!id.success) return context.json({ detail: 'Incident not found' }, 404);
    const parsed = decision.safeParse(await context.req.json().catch(() => null));
    if (!parsed.success) return context.json({ detail: 'Invalid review decision' }, 400);
    const row = await repository.decide(id.data, { ...parsed.data, reviewer_id: context.get('user').id });
    if (row) return context.json(publicIncident(row));
    return await repository.find(id.data) ? context.json({ detail: 'Incident already reviewed' }, 409) :
      context.json({ detail: 'Incident not found' }, 404);
  });

  app.post('/ingest', bodyLimit({ maxSize: 3 * maxImage + 65536 }), async (context) => {
    const form = await context.req.formData().catch(() => null);
    let raw = null;
    try { raw = JSON.parse(form?.get('metadata')); } catch { /* Invalid input is rejected below. */ }
    const parsed = metadata.safeParse(raw);
    if (!parsed.success || (parsed.data.violation_type === 'no_helmet' && !parsed.data.motorcycle_tracking_id)) {
      return context.json({ detail: 'Invalid incident metadata' }, 400);
    }
    const existing = await repository.findByKey(parsed.data.incident_key);
    if (existing) return context.json({ id: existing.id, duplicate: true });
    const images = await Promise.all(Object.keys(variants).map(async (variant) => [variant, await imageBytes(form.get(variant))]));
    if (images.some(([, bytes]) => !bytes)) return context.json({ detail: 'Three valid JPEG evidence images required' }, 400);
    const id = crypto.randomUUID();
    const store = blobs();
    const writes = await Promise.allSettled(images.map(([variant, bytes]) =>
      store.set(`${id}/${variant}`, bytes, { onlyIfNew: true })));
    if (writes.some((write) => write.status === 'rejected' || !write.value.modified)) {
      await Promise.allSettled(writes.flatMap((write, index) => write.status === 'fulfilled' && write.value.modified ?
        [store.delete(`${id}/${images[index][0]}`)] : []));
      throw new Error('Evidence write failed');
    }
    const row = await repository.insert({ ...parsed.data, id });
    if (row) return context.json({ id: row.id, duplicate: false }, 201);
    await Promise.allSettled(images.map(([variant]) => store.delete(`${id}/${variant}`)));
    const duplicate = await repository.findByKey(parsed.data.incident_key);
    if (!duplicate) throw new Error('Concurrent incident not found');
    return context.json({ id: duplicate.id, duplicate: true });
  });
  return app;
}
