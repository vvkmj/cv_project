import React, { useState, useEffect } from 'react';
import {
  Cpu,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Info,
  Layers,
  Terminal,
  Activity
} from 'lucide-react';
import { api } from '../services/api';
import { ModelStatus, SystemStatus } from '../types';

export const ModelStatusPage: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const [mRes, sRes] = await Promise.all([api.getModelStatus(), api.getSystemStatus()]);
        setModelStatus(mRes);
        setSystemStatus(sRes);
      } catch (err) {
        console.error('Error fetching model telemetry:', err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading || !modelStatus) {
    return (
      <div className="py-16 text-center text-slate-500 font-mono text-sm animate-pulse">
        Polling inference engine and hardware telemetry...
      </div>
    );
  }

  const vDet = modelStatus.vehicle_detector;
  const hDet = modelStatus.helmet_detector;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">AI Model Health & Capabilities Matrix</h1>
        <p className="text-sm text-slate-400">
          Hardware acceleration, neural network checkpoint status, and limitation disclosure.
        </p>
      </div>

      {/* Honest Warning Banner for Missing Helmet Model */}
      {!hDet.is_available && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs space-y-2">
          <div className="flex items-center gap-2 font-bold text-sm text-amber-400">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <span>Dedicated Helmet Weights Not Detected — Helmet Analysis Safely Disabled</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            In accordance with the system's strict scientific integrity policy, no-helmet violation alerts are paused rather than fabricating unverified predictions. Vehicle tracking, passenger counting, and wrong-way detection remain fully active.
          </p>
          <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-700/60 font-mono text-[11px] text-slate-300">
            Installation guide: Place trained YOLO helmet weights into <span className="text-sky-400">ml/weights/helmet_yolov8.pt</span> or execute the dataset training pipeline <span className="text-emerald-400">python ml/training/train_helmet.py</span>.
          </div>
        </div>
      )}

      {/* Status Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Vehicle & Person Detector */}
        <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">{vDet.name}</h3>
                <span className="text-xs text-slate-400 font-mono">YOLOv8 Detection Core</span>
              </div>
            </div>
            {vDet.is_loaded ? (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> ONLINE
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30 flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> OFFLINE
              </span>
            )}
          </div>

          <div className="space-y-2 text-xs divide-y divide-slate-800">
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Model Checkpoint:</span>
              <span className="font-mono text-white">{vDet.model_path}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Active Inference Device:</span>
              <span className="font-mono text-sky-400 uppercase font-bold">{vDet.device}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Confidence Threshold:</span>
              <span className="font-mono text-slate-200">{(vDet.confidence_threshold * 100).toFixed(0)}%</span>
            </div>
            <div className="py-1.5">
              <span className="text-slate-400 block mb-1">Target Classes:</span>
              <div className="flex flex-wrap gap-1.5">
                {vDet.target_classes.map((cls) => (
                  <span key={cls} className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[11px] border border-slate-700">
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Dedicated Helmet Detector */}
        <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">{hDet.name}</h3>
                <span className="text-xs text-slate-400 font-mono">Head-Crop Classifier</span>
              </div>
            </div>
            {hDet.is_available ? (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> LOADED
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5" /> WEIGHTS MISSING
              </span>
            )}
          </div>

          <div className="space-y-2 text-xs divide-y divide-slate-800">
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Target Path:</span>
              <span className="font-mono text-white text-[11px]">{hDet.weights_path}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Operational Status:</span>
              <span className="text-slate-300">{hDet.status_message}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Min Classification Confidence:</span>
              <span className="font-mono text-slate-200">{(hDet.confidence_threshold * 100).toFixed(0)}%</span>
            </div>
            <div className="py-1.5">
              <span className="text-slate-400 block mb-1">Supported Classes:</span>
              <div className="flex gap-2">
                <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[11px] border border-slate-700">Helmet</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[11px] border border-slate-700">No Helmet</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[11px] border border-slate-700">Unknown (Low Confidence)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hardware and System Telemetry */}
      {systemStatus && (
        <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-400" /> Host Hardware & System Resource Telemetry
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <span className="text-slate-400 block mb-1">CPU Load</span>
              <div className="text-lg font-bold text-white font-mono">{systemStatus.system.cpu_usage_pct}%</div>
            </div>
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <span className="text-slate-400 block mb-1">RAM Consumption</span>
              <div className="text-lg font-bold text-white font-mono">{systemStatus.system.memory_usage_pct}%</div>
            </div>
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <span className="text-slate-400 block mb-1">Available Memory</span>
              <div className="text-lg font-bold text-sky-400 font-mono">{systemStatus.system.memory_available_mb} MB</div>
            </div>
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
              <span className="text-slate-400 block mb-1">Free Storage</span>
              <div className="text-lg font-bold text-emerald-400 font-mono">{systemStatus.system.disk_free_gb} GB</div>
            </div>
          </div>
        </div>
      )}

      {/* Explicit System Limitations Disclosure */}
      <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Info className="w-4 h-4 text-sky-400" /> Operational Scope & Research Limitations
        </h3>
        <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside">
          <li><strong>Extreme Weather & Lighting:</strong> Severe rain, dense fog, or nighttime low-lux glare may reduce bounding box confidence.</li>
          <li><strong>Severe Occlusion:</strong> Motorcycles occluded by heavy trucks or foliage cannot maintain continuous tracking IDs.</li>
          <li><strong>Head Region Resolution:</strong> For vehicles situated farther than 60 meters from a 1080p camera, head crop resolution drops below 20×20 pixels, triggering the automated 'Unknown' classification.</li>
          <li><strong>Camera Calibration:</strong> Wrong-way detection requires the operator to align the camera heading vector with the real lane orientation.</li>
        </ul>
      </div>
    </div>
  );
};

