CREATE TYPE "review_state" AS ENUM('pending', 'confirmed', 'false_positive', 'rejected');--> statement-breakpoint
CREATE TABLE "traffic_incidents" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	"incident_key" text NOT NULL UNIQUE,
	"source" text NOT NULL,
	"camera_id" text NOT NULL,
	"tracking_id" integer NOT NULL,
	"motorcycle_tracking_id" integer,
	"violation_type" text NOT NULL,
	"frame_number" integer NOT NULL,
	"timestamp" double precision NOT NULL,
	"detection_confidence" double precision NOT NULL,
	"violation_confidence" double precision NOT NULL,
	"explanation" text NOT NULL,
	"review_status" "review_state" DEFAULT 'pending'::"review_state" NOT NULL,
	"reviewer_id" text,
	"reviewed_at" timestamp with time zone,
	"review_note" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "traffic_review_queue" ON "traffic_incidents" ("review_status","created_at","id");