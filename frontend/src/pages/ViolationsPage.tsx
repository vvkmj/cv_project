import React, { useState, useEffect } from 'react';
import {
  Search,
  Filter,
  Download,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { api } from '../services/api';
import { Violation, Camera, ReviewStatus } from '../types';
import { EvidenceModal } from '../components/EvidenceModal';

export const ViolationsPage: React.FC = () => {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(15);
  const [loading, setLoading] = useState<boolean>(true);

  // Filters
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [selectedStatus, setSelectedStatus] = useState<string>('');
  const [minConfidence, setMinConfidence] = useState<number>(0.0);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);

  const fetchViolations = async () => {
    try {
      setLoading(true);
      const res = await api.getViolations({
        violation_type: selectedType || undefined,
        camera_id: selectedCamera || undefined,
        review_status: selectedStatus || undefined,
        min_confidence: minConfidence > 0 ? minConfidence : undefined,
        page,
        page_size: pageSize,
      });
      setViolations(res.items);
      setTotal(res.total);
    } catch (err) {
      console.error('Error fetching violations:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    api.getCameras().then(setCameras).catch(console.error);
  }, []);

  useEffect(() => {
    fetchViolations();
  }, [page, selectedType, selectedCamera, selectedStatus, minConfidence]);

  // Client-side search by ID or explanation
  const filteredViolations = violations.filter((v) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      v.id.toLowerCase().includes(q) ||
      v.camera_id.toLowerCase().includes(q) ||
      v.explanation.toLowerCase().includes(q) ||
      v.violation_type.toLowerCase().includes(q)
    );
  });

  const totalPages = Math.ceil(total / pageSize) || 1;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header and Export Tools */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Violation Records & Audit Ledger</h1>
          <p className="text-sm text-slate-400">Search, filter, review evidentiary assets, and export compliance reports.</p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href={api.getExportUrl('csv', { violation_type: selectedType, review_status: selectedStatus })}
            download="traffic_violations_export.csv"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <Download className="w-3.5 h-3.5" /> Export CSV
          </a>
          <a
            href={api.getExportUrl('json', { violation_type: selectedType, review_status: selectedStatus })}
            download="traffic_violations_export.json"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <Download className="w-3.5 h-3.5" /> Export JSON
          </a>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-4 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          {/* Search Box */}
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search ID, explanation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* Violation Type Filter */}
          <select
            value={selectedType}
            onChange={(e) => { setSelectedType(e.target.value); setPage(1); }}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-sky-500"
          >
            <option value="">All Infraction Types</option>
            <option value="no_helmet">No Helmet</option>
            <option value="multiple_riders">Multiple Riders</option>
            <option value="wrong_way">Wrong-Way Driving</option>
          </select>

          {/* Camera Filter */}
          <select
            value={selectedCamera}
            onChange={(e) => { setSelectedCamera(e.target.value); setPage(1); }}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-sky-500"
          >
            <option value="">All Cameras</option>
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>{c.id} ({c.name})</option>
            ))}
          </select>

          {/* Review Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => { setSelectedStatus(e.target.value); setPage(1); }}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-sky-500"
          >
            <option value="">All Review Statuses</option>
            <option value="pending">Pending Review</option>
            <option value="confirmed">Confirmed</option>
            <option value="rejected">Rejected</option>
          </select>

          {/* Refresh Button */}
          <button
            onClick={fetchViolations}
            className="flex items-center justify-center gap-1.5 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Apply Filters
          </button>
        </div>
      </div>

      {/* Violations Table */}
      <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-700/60 text-slate-400 bg-slate-900/60">
                <th className="py-3 px-4">Violation ID</th>
                <th className="py-3 px-4">Thumbnail</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Track ID</th>
                <th className="py-3 px-4">Camera</th>
                <th className="py-3 px-4">Time Offset</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Review Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredViolations.map((v) => (
                <tr key={v.id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3 px-4 font-mono font-medium text-white">{v.id}</td>
                  <td className="py-3 px-4">
                    <img
                      src={`/${v.crop_path || v.annotated_evidence_path}`}
                      alt="Crop"
                      className="w-10 h-10 object-cover rounded-md border border-slate-700 bg-slate-900"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><rect width="40" height="40" fill="%231e293b"/></svg>';
                      }}
                    />
                  </td>
                  <td className="py-3 px-4">
                    <span className="capitalize font-medium text-slate-200">
                      {v.violation_type.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-300">
                    {v.tracking_id ? `#${v.tracking_id}` : '-'}
                  </td>
                  <td className="py-3 px-4 text-sky-400 font-mono">{v.camera_id}</td>
                  <td className="py-3 px-4 text-slate-400">{v.timestamp.toFixed(2)}s</td>
                  <td className="py-3 px-4 font-mono text-emerald-400">
                    {(v.violation_confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-3 px-4">
                    {v.review_status === 'confirmed' && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        Confirmed
                      </span>
                    )}
                    {v.review_status === 'rejected' && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                        Rejected
                      </span>
                    )}
                    {v.review_status === 'pending' && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                        Pending
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      onClick={() => setSelectedViolation(v)}
                      className="px-2.5 py-1 rounded-lg bg-sky-600/20 hover:bg-sky-600/30 text-sky-300 text-xs font-medium transition"
                    >
                      Audit Evidence
                    </button>
                  </td>
                </tr>
              ))}

              {filteredViolations.length === 0 && (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-slate-500">
                    {loading ? 'Loading violation records...' : 'No violation events match the selected criteria.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="px-4 py-3 border-t border-slate-700/60 bg-slate-900/40 flex items-center justify-between text-xs text-slate-400">
          <span>
            Showing {filteredViolations.length} of {total} records
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="font-mono text-slate-200">
              Page {page} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Evidence Modal */}
      <EvidenceModal
        violation={selectedViolation}
        onClose={() => setSelectedViolation(null)}
        onStatusUpdated={(updated) => {
          setViolations((prev) => prev.map((v) => (v.id === updated.id ? updated : v)));
          setSelectedViolation(updated);
        }}
      />
    </div>
  );
};

