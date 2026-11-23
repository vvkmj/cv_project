import React, { useState, useEffect } from 'react';
import {
  Camera as CameraIcon,
  Plus,
  Trash2,
  Edit2,
  Compass,
  CheckCircle2,
  AlertCircle,
  Users,
  HardHat,
  RotateCw
} from 'lucide-react';
import { api } from '../services/api';
import { Camera } from '../types';

export const CamerasPage: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | null>(null);

  // Form State
  const [cameraId, setCameraId] = useState('');
  const [name, setName] = useState('');
  const [locationLabel, setLocationLabel] = useState('');
  const [angleDeg, setAngleDeg] = useState(90);
  const [directionDesc, setDirectionDesc] = useState('Southbound flow (Top to Bottom)');
  const [passengerLimit, setPassengerLimit] = useState(2);
  const [noHelmetEnabled, setNoHelmetEnabled] = useState(true);
  const [multipleRidersEnabled, setMultipleRidersEnabled] = useState(true);
  const [wrongWayEnabled, setWrongWayEnabled] = useState(true);

  const fetchCameras = async () => {
    try {
      setLoading(true);
      const data = await api.getCameras();
      setCameras(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCameras();
  }, []);

  const openCreateModal = () => {
    setEditingCamera(null);
    setCameraId(`CAM-${String(cameras.length + 1).padStart(2, '0')}`);
    setName('');
    setLocationLabel('');
    setAngleDeg(90);
    setDirectionDesc('Southbound flow (Top to Bottom)');
    setPassengerLimit(2);
    setNoHelmetEnabled(true);
    setMultipleRidersEnabled(true);
    setWrongWayEnabled(true);
    setIsModalOpen(true);
  };

  const openEditModal = (cam: Camera) => {
    setEditingCamera(cam);
    setCameraId(cam.id);
    setName(cam.name);
    setLocationLabel(cam.location_label);
    setAngleDeg(cam.expected_direction?.angle_deg ?? 90);
    setDirectionDesc(cam.expected_direction?.description ?? 'Configured flow');
    setPassengerLimit(cam.passenger_limit);
    setNoHelmetEnabled(cam.enabled_violations?.no_helmet ?? true);
    setMultipleRidersEnabled(cam.enabled_violations?.multiple_riders ?? true);
    setWrongWayEnabled(cam.enabled_violations?.wrong_way ?? true);
    setIsModalOpen(true);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    const rad = (angleDeg * Math.PI) / 180;
    const dx = Math.cos(rad);
    const dy = Math.sin(rad);

    const payload = {
      id: cameraId,
      name,
      location_label: locationLabel,
      expected_direction: {
        angle_deg: angleDeg,
        dx: Number(dx.toFixed(3)),
        dy: Number(dy.toFixed(3)),
        description: directionDesc,
      },
      passenger_limit: passengerLimit,
      enabled_violations: {
        no_helmet: noHelmetEnabled,
        multiple_riders: multipleRidersEnabled,
        wrong_way: wrongWayEnabled,
      },
    };

    try {
      if (editingCamera) {
        await api.updateCamera(editingCamera.id, payload);
      } else {
        await api.createCamera(payload);
      }
      setIsModalOpen(false);
      fetchCameras();
    } catch (err: any) {
      alert(err.message || 'Failed to save camera');
    }
  };

  const handleDelete = async (id: string) => {
    if (confirm(`Are you sure you want to remove ${id}?`)) {
      try {
        await api.deleteCamera(id);
        fetchCameras();
      } catch (err: any) {
        alert(err.message || 'Failed to delete camera');
      }
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Camera & Lane Stream Configuration</h1>
          <p className="text-sm text-slate-400">Configure visual direction vectors, speed and passenger policies per feed.</p>
        </div>
        <button
          onClick={openCreateModal}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white transition shadow-sm"
        >
          <Plus className="w-4 h-4" /> Add Camera Feed
        </button>
      </div>

      {/* Camera Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {cameras.map((cam) => {
          const angle = cam.expected_direction?.angle_deg ?? 90;
          return (
            <div
              key={cam.id}
              className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 flex flex-col justify-between space-y-4 hover:border-slate-600 transition"
            >
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-bold text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800/60">
                      {cam.id}
                    </span>
                    <h3 className="text-sm font-bold text-white">{cam.name}</h3>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => openEditModal(cam)}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition"
                      title="Edit Configuration"
                    >
                      <Edit2 className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => handleDelete(cam.id)}
                      className="p-1.5 rounded-lg text-rose-400 hover:text-rose-300 hover:bg-rose-950/40 transition"
                      title="Delete Camera"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                <p className="text-xs text-slate-400 mt-1">{cam.location_label}</p>
              </div>

              {/* Visual Heading Compass Indicator */}
              <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-700/50 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Expected Legal Flow
                  </span>
                  <span className="text-xs font-medium text-slate-200 mt-0.5 block">
                    {cam.expected_direction?.description || `${angle}° Heading`}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">Angle: {angle}°</span>
                </div>

                {/* Compass Dial */}
                <div className="w-12 h-12 rounded-full border border-slate-600 relative flex items-center justify-center bg-slate-800/80">
                  <span className="text-[8px] absolute top-0 text-slate-500 font-mono">N</span>
                  <span className="text-[8px] absolute bottom-0 text-slate-500 font-mono">S</span>
                  <div
                    className="w-1 h-6 bg-gradient-to-t from-transparent to-sky-400 rounded-full transition-transform duration-300 origin-center"
                    style={{ transform: `rotate(${angle}deg)` }}
                  />
                  <div className="w-2 h-2 rounded-full bg-sky-400 absolute" />
                </div>
              </div>

              {/* Enforcement Rules Summary */}
              <div className="space-y-1.5 text-xs text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Passenger Limit:</span>
                  <span className="font-semibold text-white">{cam.passenger_limit} Riders Max</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">No Helmet Rule:</span>
                  <span className={cam.enabled_violations?.no_helmet ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
                    {cam.enabled_violations?.no_helmet ? 'Enforced' : 'Disabled'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Multiple Riders:</span>
                  <span className={cam.enabled_violations?.multiple_riders ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
                    {cam.enabled_violations?.multiple_riders ? 'Enforced' : 'Disabled'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Wrong-Way Rule:</span>
                  <span className={cam.enabled_violations?.wrong_way ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
                    {cam.enabled_violations?.wrong_way ? 'Enforced' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Modal: Add or Edit Camera */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-xl p-6 shadow-2xl space-y-5 animate-fade-in">
            <h2 className="text-lg font-bold text-white">
              {editingCamera ? `Edit Camera: ${editingCamera.id}` : 'Configure New Camera Stream'}
            </h2>

            <form onSubmit={handleSave} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Camera ID</label>
                  <input
                    type="text"
                    required
                    disabled={!!editingCamera}
                    value={cameraId}
                    onChange={(e) => setCameraId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Camera Name</label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. North Gate Main Approach"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Location Label</label>
                <input
                  type="text"
                  required
                  value={locationLabel}
                  onChange={(e) => setLocationLabel(e.target.value)}
                  placeholder="e.g. Highway 101 KM 42 Southbound"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white"
                />
              </div>

              {/* Visual Expected Traffic Direction Configurator */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white flex items-center gap-2">
                    <Compass className="w-4 h-4 text-sky-400" /> Interactive Traffic Direction Compass
                  </span>
                  <span className="font-mono text-sky-400 font-bold">{angleDeg}° Heading</span>
                </div>

                <div className="flex items-center gap-6">
                  {/* Compass Visual Dial */}
                  <div
                    onClick={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const cx = rect.left + rect.width / 2;
                      const cy = rect.top + rect.height / 2;
                      const rad = Math.atan2(e.clientY - cy, e.clientX - cx);
                      let deg = Math.round((rad * 180) / Math.PI + 90);
                      if (deg < 0) deg += 360;
                      setAngleDeg(deg);
                    }}
                    className="w-28 h-28 rounded-full border-2 border-slate-700 relative flex items-center justify-center bg-slate-900 cursor-pointer hover:border-sky-500 transition shadow-inner"
                    title="Click around the compass circle to set the legal traffic direction"
                  >
                    <span className="text-[9px] absolute top-1 text-slate-400 font-mono font-bold">0° N</span>
                    <span className="text-[9px] absolute right-1 text-slate-400 font-mono font-bold">90° E</span>
                    <span className="text-[9px] absolute bottom-1 text-slate-400 font-mono font-bold">180° S</span>
                    <span className="text-[9px] absolute left-1 text-slate-400 font-mono font-bold">270° W</span>
                    <div
                      className="w-1.5 h-12 bg-gradient-to-t from-transparent to-sky-400 rounded-full transition-transform duration-200"
                      style={{ transform: `rotate(${angleDeg}deg)` }}
                    />
                    <div className="w-3 h-3 rounded-full bg-sky-400 absolute" />
                  </div>

                  <div className="flex-1 space-y-2">
                    <p className="text-slate-400 text-[11px]">
                      Drag the slider or click the compass to orient the legal traffic movement vector for this camera angle.
                    </p>
                    <input
                      type="range"
                      min="0"
                      max="359"
                      value={angleDeg}
                      onChange={(e) => setAngleDeg(parseInt(e.target.value))}
                      className="w-full accent-sky-500 cursor-pointer"
                    />
                    <input
                      type="text"
                      value={directionDesc}
                      onChange={(e) => setDirectionDesc(e.target.value)}
                      placeholder="e.g. Southbound flow (Downwards)"
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white text-[11px]"
                    />
                  </div>
                </div>
              </div>

              {/* Policy Options */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Max Passengers Allowed</label>
                  <select
                    value={passengerLimit}
                    onChange={(e) => setPassengerLimit(parseInt(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white"
                  >
                    <option value={1}>1 Rider (Strict Single Seat)</option>
                    <option value={2}>2 Riders (Standard Traffic Rule)</option>
                    <option value={3}>3 Riders</option>
                  </select>
                </div>
                <div className="space-y-1.5 pt-4">
                  <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={noHelmetEnabled}
                      onChange={(e) => setNoHelmetEnabled(e.target.checked)}
                      className="rounded accent-sky-500"
                    />
                    <span>Enforce Helmet Verification</span>
                  </label>
                  <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={wrongWayEnabled}
                      onChange={(e) => setWrongWayEnabled(e.target.checked)}
                      className="rounded accent-sky-500"
                    />
                    <span>Enforce Wrong-Way Direction</span>
                  </label>
                </div>
              </div>

              {/* Buttons */}
              <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold transition shadow-sm"
                >
                  Save Camera Configuration
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

