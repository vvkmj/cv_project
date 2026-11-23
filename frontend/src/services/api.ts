import {
  Camera,
  Violation,
  AnalysisJob,
  ModelStatus,
  SystemStatus,
  ReviewStatus
} from '../types';

const BASE_URL = '/api';

export const api = {
  // Health & System
  getHealth: async (): Promise<{ status: string; timestamp: string }> => {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error('Failed to fetch health status');
    return res.json();
  },

  getSystemStatus: async (): Promise<SystemStatus> => {
    const res = await fetch(`${BASE_URL}/system/status`);
    if (!res.ok) throw new Error('Failed to fetch system telemetry');
    return res.json();
  },

  getModelStatus: async (): Promise<ModelStatus> => {
    const res = await fetch(`${BASE_URL}/models/status`);
    if (!res.ok) throw new Error('Failed to fetch model status');
    return res.json();
  },

  // Cameras
  getCameras: async (): Promise<Camera[]> => {
    const res = await fetch(`${BASE_URL}/cameras`);
    if (!res.ok) throw new Error('Failed to fetch cameras');
    return res.json();
  },

  createCamera: async (camera: Partial<Camera>): Promise<Camera> => {
    const res = await fetch(`${BASE_URL}/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(camera),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to create camera');
    }
    return res.json();
  },

  updateCamera: async (cameraId: string, update: Partial<Camera>): Promise<Camera> => {
    const res = await fetch(`${BASE_URL}/cameras/${cameraId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(update),
    });
    if (!res.ok) throw new Error('Failed to update camera');
    return res.json();
  },

  deleteCamera: async (cameraId: string): Promise<void> => {
    const res = await fetch(`${BASE_URL}/cameras/${cameraId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete camera');
  },

  // Media Upload & Analysis
  uploadMedia: async (file: File): Promise<{ filename: string; media_type: 'image' | 'video' }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/media/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to upload media file');
    }
    return res.json();
  },

  analyzeImage: async (filename: string, cameraId: string): Promise<AnalysisJob> => {
    const formData = new FormData();
    formData.append('filename', filename);
    formData.append('camera_id', cameraId);

    const res = await fetch(`${BASE_URL}/analysis/image`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to analyze image');
    }
    return res.json();
  },

  analyzeVideo: async (filename: string, cameraId: string): Promise<AnalysisJob> => {
    const formData = new FormData();
    formData.append('filename', filename);
    formData.append('camera_id', cameraId);

    const res = await fetch(`${BASE_URL}/analysis/video`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to initiate video analysis');
    }
    return res.json();
  },

  getJobStatus: async (jobId: string): Promise<AnalysisJob> => {
    const res = await fetch(`${BASE_URL}/analysis/${jobId}`);
    if (!res.ok) throw new Error('Failed to fetch job status');
    return res.json();
  },

  getJobResults: async (jobId: string): Promise<AnalysisJob> => {
    const res = await fetch(`${BASE_URL}/analysis/${jobId}/results`);
    if (!res.ok) throw new Error('Failed to fetch job results');
    return res.json();
  },

  // Violations
  getViolations: async (params?: {
    violation_type?: string;
    camera_id?: string;
    review_status?: string;
    min_confidence?: number;
    page?: number;
    page_size?: number;
  }): Promise<{ total: number; page: number; page_size: number; items: Violation[] }> => {
    const url = new URL(`${window.location.origin}${BASE_URL}/violations`);
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null && v !== '') {
          url.searchParams.append(k, String(v));
        }
      });
    }
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error('Failed to fetch violations');
    return res.json();
  },

  getViolation: async (violationId: string): Promise<Violation> => {
    const res = await fetch(`${BASE_URL}/violations/${violationId}`);
    if (!res.ok) throw new Error('Failed to fetch violation details');
    return res.json();
  },

  updateViolationStatus: async (
    violationId: string,
    status: ReviewStatus
  ): Promise<Violation> => {
    const res = await fetch(`${BASE_URL}/violations/${violationId}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: status }),
    });
    if (!res.ok) throw new Error('Failed to update violation status');
    return res.json();
  },

  getExportUrl: (format: 'csv' | 'json', filters?: { violation_type?: string; review_status?: string }) => {
    const url = new URL(`${window.location.origin}${BASE_URL}/violations/export`);
    url.searchParams.append('format', format);
    if (filters?.violation_type) url.searchParams.append('violation_type', filters.violation_type);
    if (filters?.review_status) url.searchParams.append('review_status', filters.review_status);
    return url.toString();
  }
};

