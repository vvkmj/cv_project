export type ViolationType = 'no_helmet' | 'multiple_riders' | 'wrong_way';
export type ReviewStatus = 'pending' | 'confirmed' | 'rejected';
export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ExpectedDirection {
  angle_deg?: number;
  dx?: number;
  dy?: number;
  description?: string;
}

export interface Camera {
  id: string;
  name: string;
  location_label: string;
  expected_direction: ExpectedDirection;
  passenger_limit: number;
  enabled_violations: {
    no_helmet: boolean;
    multiple_riders: boolean;
    wrong_way: boolean;
  };
  created_at: string;
}

export interface Violation {
  id: string;
  job_id: string;
  camera_id: string;
  tracking_id?: number | null;
  violation_type: ViolationType;
  timestamp: number;
  detection_confidence: number;
  violation_confidence: number;
  evidence_path: string;
  annotated_evidence_path: string;
  crop_path?: string | null;
  review_status: ReviewStatus;
  explanation: string;
  created_at: string;
}

export interface Detection {
  id?: number;
  frame_number: number;
  timestamp: number;
  tracking_id?: number | null;
  object_class: string;
  confidence: number;
  bounding_box: BoundingBox;
}

export interface JobSummary {
  total_vehicles_detected: number;
  violations_count: number;
  no_helmet_count: number;
  multiple_riders_count: number;
  wrong_way_count: number;
}

export interface AnalysisJob {
  id: string;
  filename: string;
  media_type: 'image' | 'video';
  camera_id: string;
  status: JobStatus;
  progress: number;
  total_frames: number;
  processed_frames: number;
  summary?: JobSummary | null;
  annotated_path?: string | null;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
  violations?: Violation[];
  detections?: Detection[];
}

export interface ModelStatus {
  vehicle_detector: {
    name: string;
    model_path: string;
    version: string;
    is_loaded: boolean;
    device: string;
    target_classes: string[];
    confidence_threshold: number;
  };
  helmet_detector: {
    name: string;
    weights_path: string;
    version: string;
    is_available: boolean;
    device: string;
    confidence_threshold: number;
    status_message: string;
    guidance: string;
  };
  hardware_device: string;
  pipeline_operational: boolean;
}

export interface SystemStatus {
  system: {
    cpu_usage_pct: number;
    memory_usage_pct: number;
    memory_available_mb: number;
    disk_free_gb: number;
  };
  workload: {
    active_processing_jobs: number;
    completed_jobs: number;
    total_violations_recorded: number;
  };
  timestamp: string;
}

