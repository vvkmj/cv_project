import { doublePrecision, index, integer, pgEnum, pgTable, text, timestamp, uuid } from 'drizzle-orm/pg-core';

export const reviewState = pgEnum('review_state', ['pending', 'confirmed', 'false_positive', 'rejected']);
export const incidents = pgTable('traffic_incidents', {
  id: uuid('id').primaryKey().defaultRandom(),
  incident_key: text('incident_key').notNull().unique(),
  source: text('source').notNull(),
  camera_id: text('camera_id').notNull(),
  tracking_id: integer('tracking_id').notNull(),
  motorcycle_tracking_id: integer('motorcycle_tracking_id'),
  violation_type: text('violation_type').notNull(),
  frame_number: integer('frame_number').notNull(),
  timestamp: doublePrecision('timestamp').notNull(),
  detection_confidence: doublePrecision('detection_confidence').notNull(),
  violation_confidence: doublePrecision('violation_confidence').notNull(),
  explanation: text('explanation').notNull(),
  review_status: reviewState('review_status').notNull().default('pending'),
  reviewer_id: text('reviewer_id'),
  reviewed_at: timestamp('reviewed_at', { withTimezone: true, mode: 'string' }),
  review_note: text('review_note'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'string' }).notNull().defaultNow(),
}, (table) => [index('traffic_review_queue').on(table.review_status, table.created_at, table.id)]);
