import React, { useState } from 'react';
import { X, CheckCircle, XCircle, Clock, AlertTriangle, ExternalLink, Image as ImageIcon } from 'lucide-react';
import { Violation, ReviewStatus } from '../types';
import { api } from '../services/api';

interface EvidenceModalProps {
  violation: Violation | null;
  onClose: () => void;
  onStatusUpdated?: (updated: Violation) => void;
}

export const EvidenceModal: React.FC<EvidenceModalProps> = ({ violation, onClose, onStatusUpdated }) => {
  if (!violation) return null;

  const [activeView, setActiveView] = useState<'annotated' | 'original' | 'crop'>('annotated');
  const [updating, setUpdating] = useState(false);

  const handleUpdateStatus = async (newStatus: ReviewStatus) => {
    try {
      setUpdating(true);
      const updated = await api.updateViolationStatus(violation.id, newStatus);
      if (onStatusUpdated) onStatusUpdated(updated);
    } catch (err) {
      console.error(err);
      alert('Failed to update status.');
    } finally {
      setUpdating(false);
    }
  };

  const getImageSrc = () => {
    switch (activeView) {
      case 'original':
        return `/${violation.evidence_path}`;
      case 'crop':
        return violation.crop_path ? `/${violation.crop_path}` : `/${violation.annotated_evidence_path}`;
      case 'annotated':
      default:
        return `/${violation.annotated_evidence_path}`;
    }
  };

  const getStatusBadge = (status: ReviewStatus) => {
    switch (status) {
      case 'confirmed':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">CONFIRMED</span>;
      case 'rejected':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">REJECTED</span>;
      case 'pending':
      default:
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">PENDING REVIEW</span>;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl w-full max-w-4xl max-h-[92vh] overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white font-mono">{violation.id}</h3>
                {getStatusBadge(violation.review_status)}
              </div>
              <p className="text-xs text-slate-400">
                Camera: <span className="text-sky-300 font-mono">{violation.camera_id}</span> | Timestamp: <span className="text-slate-200">{violation.timestamp.toFixed(2)}s</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* View Mode Toggle Buttons */}
          <div className="flex items-center justify-between">
            <div className="flex bg-slate-800/80 p-1 rounded-xl border border-slate-700/50 text-xs">
              <button
                onClick={() => setActiveView('annotated')}
                className={`px-3 py-1.5 rounded-lg font-medium transition ${
                  activeView === 'annotated' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
              >
                Annotated Evidence
              </button>
              <button
                onClick={() => setActiveView('original')}
                className={`px-3 py-1.5 rounded-lg font-medium transition ${
                  activeView === 'original' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
              >
                Original Frame
              </button>
              <button
                onClick={() => setActiveView('crop')}
                className={`px-3 py-1.5 rounded-lg font-medium transition ${
                  activeView === 'crop' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
              >
                Vehicle Crop
              </button>
            </div>

            <a
              href={getImageSrc()}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1 font-medium"
            >
              Open Full Resolution <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          {/* Visual Evidence Viewer */}
          <div className="relative bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center min-h-[360px]">
            <img
              src={getImageSrc()}
              alt="Violation Evidence"
              className="max-h-[460px] w-auto object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"><rect width="400" height="200" fill="%231e293b"/><text x="50%" y="50%" fill="%2364748b" text-anchor="middle" font-family="sans-serif">Evidence file generated on server</text></svg>';
              }}
            />
          </div>

          {/* Details & Telemetry */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-2">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Violation Classification</h4>
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Violation Category:</span>
                  <span className="font-semibold text-white capitalize">{violation.violation_type.replace('_', ' ')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Tracking ID:</span>
                  <span className="font-mono text-sky-300">{violation.tracking_id ? `#${violation.tracking_id}` : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Violation Confidence:</span>
                  <span className="font-mono text-emerald-400">{(violation.violation_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Detection Confidence:</span>
                  <span className="font-mono text-slate-300">{(violation.detection_confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-2">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">AI Analytical Explanation</h4>
              <p className="text-xs text-slate-300 leading-relaxed bg-slate-900/60 p-3 rounded-lg border border-slate-800 font-mono">
                {violation.explanation}
              </p>
            </div>
          </div>
        </div>

        {/* Footer / Operator Decision Panel */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/80 flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs text-slate-400">
            Human Review Decision Support. Decisions are logged for audit compliance.
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleUpdateStatus('rejected')}
              disabled={updating}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition disabled:opacity-50"
            >
              <XCircle className="w-4 h-4" /> Reject (False Positive)
            </button>
            <button
              onClick={() => handleUpdateStatus('pending')}
              disabled={updating}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition disabled:opacity-50"
            >
              <Clock className="w-4 h-4" /> Mark Pending
            </button>
            <button
              onClick={() => handleUpdateStatus('confirmed')}
              disabled={updating}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50 shadow-md"
            >
              <CheckCircle className="w-4 h-4" /> Confirm Violation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

