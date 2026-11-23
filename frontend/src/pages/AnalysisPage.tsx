import React, { useState, useEffect } from 'react';
import {
  Upload,
  Video,
  Image as ImageIcon,
  CheckCircle,
  AlertTriangle,
  Play,
  Layers,
  Settings,
  Clock,
  Eye,
  FileCheck
} from 'lucide-react';
import { api } from '../services/api';
import { Camera, AnalysisJob, Violation } from '../types';
import { EvidenceModal } from '../components/EvidenceModal';

export const AnalysisPage: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-01');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [currentJob, setCurrentJob] = useState<AnalysisJob | null>(null);
  const [activeTab, setActiveTab] = useState<'annotated' | 'original'>('annotated');
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Analysis settings
  const [confidence, setConfidence] = useState<number>(0.35);
  const [passengerLimit, setPassengerLimit] = useState<number>(2);

  // Fetch available cameras
  useEffect(() => {
    api.getCameras().then((cams) => {
      setCameras(cams);
      if (cams.length > 0) setSelectedCameraId(cams[0].id);
    }).catch(console.error);
  }, []);

  // Polling for video analysis progress
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await api.getJobStatus(currentJob.id);
        setCurrentJob(updated);

        // If newly completed, load full results with detections and violations
        if (updated.status === 'completed') {
          const fullResults = await api.getJobResults(currentJob.id);
          setCurrentJob(fullResults);
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error polling job status:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [currentJob?.id, currentJob?.status]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMessage(null);
    }
  };

  const loadSampleMedia = async (samplePath: string, fileName: string, mimeType: string) => {
    try {
      setErrorMessage(null);
      const res = await fetch(`http://127.0.0.1:8000/${samplePath}`);
      if (!res.ok) throw new Error(`Could not load sample from ${samplePath}`);
      const blob = await res.blob();
      const loadedFile = new File([blob], fileName, { type: mimeType });
      setFile(loadedFile);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(`Failed to load preset sample media: ${err.message}`);
    }
  };

  const handleStartAnalysis = async () => {
    if (!file) {
      setErrorMessage('Please choose an image or video file first.');
      return;
    }

    try {
      setUploading(true);
      setErrorMessage(null);

      // 1. Upload media
      const uploadRes = await api.uploadMedia(file);

      // 2. Dispatch analysis
      if (uploadRes.media_type === 'image') {
        const job = await api.analyzeImage(uploadRes.filename, selectedCameraId);
        setCurrentJob(job);
      } else {
        const job = await api.analyzeVideo(uploadRes.filename, selectedCameraId);
        setCurrentJob(job);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || 'Error occurred while processing media.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Upload & Media Analysis</h1>
        <p className="text-sm text-slate-400">Process traffic feeds, track vehicles, and inspect potential infractions.</p>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Upload and Configuration Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* File Dropzone */}
        <div className="lg:col-span-2 bg-slate-800/30 border border-slate-700/60 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              <Upload className="w-4 h-4 text-sky-400" /> Select Traffic Media (Image or Video)
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Supported formats: MP4, AVI, MOV, MKV, JPG, PNG. Video files are queued for background processing with live frame telemetry.
            </p>

            <label className="border-2 border-dashed border-slate-700 hover:border-sky-500/60 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition bg-slate-900/40 hover:bg-slate-900/70">
              <input
                type="file"
                className="hidden"
                accept="video/*,image/*"
                onChange={handleFileChange}
              />
              <div className="w-12 h-12 rounded-full bg-sky-500/10 flex items-center justify-center text-sky-400 mb-3">
                {file?.type.includes('video') ? <Video className="w-6 h-6" /> : <ImageIcon className="w-6 h-6" />}
              </div>
              {file ? (
                <div className="text-center">
                  <span className="text-sm font-semibold text-white">{file.name}</span>
                  <p className="text-xs text-slate-400 mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              ) : (
                <div className="text-center">
                  <span className="text-sm font-medium text-slate-300">Click to browse or drag & drop</span>
                  <p className="text-xs text-slate-500 mt-1">Traffic camera clips, intersection snapshots, highway feeds</p>
                </div>
              )}
            </label>

            {/* Quick Sample Presets */}
            <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-slate-700/50">
              <span className="text-xs text-slate-400 font-medium">Quick Preset Samples:</span>
              <button
                type="button"
                onClick={() => loadSampleMedia('sample_media/motorcycle_helmet_violation.mp4', 'motorcycle_helmet_violation.mp4', 'video/mp4')}
                className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-sky-400 border border-slate-700 hover:border-sky-500/50 rounded-lg transition"
              >
                🎬 Helmet & Traffic Video (MP4)
              </button>
              <button
                type="button"
                onClick={() => loadSampleMedia('sample_media/traffic_highway.mp4', 'traffic_highway.mp4', 'video/mp4')}
                className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 border border-slate-700 hover:border-emerald-500/50 rounded-lg transition"
              >
                🚗 Highway Wrong-Way Feed (MP4)
              </button>
              <button
                type="button"
                onClick={() => loadSampleMedia('sample_media/bus.jpg', 'bus.jpg', 'image/jpeg')}
                className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-amber-400 border border-slate-700 hover:border-amber-500/50 rounded-lg transition"
              >
                📸 Urban Traffic Keyframe (JPG)
              </button>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <button
              onClick={handleStartAnalysis}
              disabled={!file || uploading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-sm bg-sky-600 hover:bg-sky-500 text-white transition disabled:opacity-50 shadow-md"
            >
              <Play className="w-4 h-4" />
              {uploading ? 'Uploading & Queuing...' : 'Execute Analysis'}
            </button>
          </div>
        </div>

        {/* Camera and Parameters Panel */}
        <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-6 space-y-5">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Settings className="w-4 h-4 text-sky-400" /> Feed & Rule Parameters
          </h3>

          <div className="space-y-4 text-xs">
            {/* Camera Select */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Target Camera Stream</label>
              <select
                value={selectedCameraId}
                onChange={(e) => setSelectedCameraId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-sky-500"
              >
                {cameras.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id} — {c.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Confidence Threshold */}
            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Detection Confidence</span>
                <span className="font-mono text-sky-400">{Math.round(confidence * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-sky-500 cursor-pointer"
              />
            </div>

            {/* Passenger Limit */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Motorcycle Passenger Limit</label>
              <select
                value={passengerLimit}
                onChange={(e) => setPassengerLimit(parseInt(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-sky-500"
              >
                <option value={1}>1 Rider Max (Single Rider Only)</option>
                <option value={2}>2 Riders Max (Standard Lawful Limit)</option>
                <option value={3}>3 Riders Max</option>
              </select>
            </div>

            {/* Camera Rule Snapshot */}
            {cameras.find((c) => c.id === selectedCameraId) && (
              <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-700/60 text-[11px] space-y-1">
                <span className="text-slate-400 uppercase font-semibold">Active Stream Profile</span>
                <p className="text-slate-200">
                  Legal Heading: {cameras.find((c) => c.id === selectedCameraId)?.expected_direction?.description || 'Southbound'}
                </p>
                <p className="text-slate-400">
                  Location: {cameras.find((c) => c.id === selectedCameraId)?.location_label}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Progress & Results Section */}
      {currentJob && (
        <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-6 space-y-6">
          {/* Job Status Banner */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-700/60 pb-4">
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-base font-bold text-white font-mono">{currentJob.id}</h3>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                  currentJob.status === 'completed'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : currentJob.status === 'failed'
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    : 'bg-sky-500/20 text-sky-400 border border-sky-500/30 animate-pulse'
                }`}>
                  {currentJob.status.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Media: <span className="text-slate-200">{currentJob.filename}</span> | Frames: {currentJob.processed_frames}/{currentJob.total_frames}
              </p>
            </div>

            {/* Progress Bar for video */}
            <div className="w-full sm:w-64">
              <div className="flex justify-between text-xs text-slate-400 mb-1 font-mono">
                <span>Progress</span>
                <span>{currentJob.progress.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-700">
                <div
                  className="h-full bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-300"
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
            </div>
          </div>

          {/* Completed Results Display */}
          {currentJob.status === 'completed' && (
            <div className="space-y-6">
              {/* Telemetry Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/50">
                  <span className="text-xs text-slate-400">Total Detections</span>
                  <div className="text-xl font-bold text-white font-mono mt-1">
                    {currentJob.summary?.total_vehicles_detected ?? currentJob.detections?.length ?? 0}
                  </div>
                </div>
                <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/50">
                  <span className="text-xs text-slate-400">Violations Flagged</span>
                  <div className="text-xl font-bold text-rose-400 font-mono mt-1">
                    {currentJob.summary?.violations_count ?? currentJob.violations?.length ?? 0}
                  </div>
                </div>
                <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/50">
                  <span className="text-xs text-slate-400">Multiple Riders</span>
                  <div className="text-xl font-bold text-orange-400 font-mono mt-1">
                    {currentJob.summary?.multiple_riders_count ?? 0}
                  </div>
                </div>
                <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-700/50">
                  <span className="text-xs text-slate-400">Wrong-Way Driving</span>
                  <div className="text-xl font-bold text-red-500 font-mono mt-1">
                    {currentJob.summary?.wrong_way_count ?? 0}
                  </div>
                </div>
              </div>

              {/* Media Output Preview */}
              <div className="bg-slate-950 rounded-xl border border-slate-800 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <Eye className="w-4 h-4 text-sky-400" /> Visual Verification Stream
                  </h4>
                  {currentJob.media_type === 'video' && (
                    <a
                      href={`/api/analysis/${currentJob.id}/annotated-video`}
                      download
                      className="text-xs text-sky-400 hover:text-sky-300 font-medium"
                    >
                      Download Processed Video
                    </a>
                  )}
                </div>

                <div className="flex items-center justify-center bg-slate-900 rounded-lg overflow-hidden min-h-[340px]">
                  {currentJob.media_type === 'video' ? (
                    <video
                      controls
                      autoPlay
                      muted
                      loop
                      className="max-h-[480px] w-auto rounded-lg"
                      src={`/api/analysis/${currentJob.id}/annotated-video`}
                    />
                  ) : (
                    <img
                      src={`/${currentJob.annotated_path}`}
                      alt="Annotated Results"
                      className="max-h-[480px] w-auto object-contain rounded-lg"
                    />
                  )}
                </div>
              </div>

              {/* Detected Violations List */}
              {currentJob.violations && currentJob.violations.length > 0 && (
                <div className="bg-slate-900/50 rounded-xl border border-slate-700/60 p-4 space-y-3">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Violations Generated During This Run
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {currentJob.violations.map((v) => (
                      <div
                        key={v.id}
                        onClick={() => setSelectedViolation(v)}
                        className="p-3 bg-slate-800/60 hover:bg-slate-800 border border-slate-700/80 rounded-xl cursor-pointer transition flex items-center justify-between gap-3"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-bold text-white">{v.id}</span>
                            <span className="text-[11px] px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold uppercase">
                              {v.violation_type.replace('_', ' ')}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 mt-1 line-clamp-1">{v.explanation}</p>
                        </div>
                        <button className="px-3 py-1.5 rounded-lg bg-sky-600/20 text-sky-300 text-xs font-medium hover:bg-sky-600/30 transition whitespace-nowrap">
                          Inspect
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Evidence Modal */}
      <EvidenceModal
        violation={selectedViolation}
        onClose={() => setSelectedViolation(null)}
        onStatusUpdated={(updated) => {
          if (currentJob?.violations) {
            setCurrentJob({
              ...currentJob,
              violations: currentJob.violations.map((v) => (v.id === updated.id ? updated : v))
            });
          }
          setSelectedViolation(updated);
        }}
      />
    </div>
  );
};

