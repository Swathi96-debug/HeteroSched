import React from 'react';
import { Cpu, Zap, Activity, HardDrive, ShieldCheck } from 'lucide-react';

export default function SystemHeader({ systemInfo, isRunning, isFresh, source, timestamp }) {
  const gpu = systemInfo?.gpu || {
    name: 'NVIDIA GeForce RTX 4050 Laptop GPU',
    compute_capability: '8.9',
    global_memory_gb: 6.0,
    sm_count: 20,
  };

  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : null;

  return (
    <header className="glass-panel" style={{ padding: '1.25rem 1.75rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        {/* Brand & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #10b981 100%)',
            padding: '0.6rem',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(16, 185, 129, 0.35)'
          }}>
            <Zap size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h1 style={{ fontSize: '1.35rem', fontWeight: '700', letterSpacing: '-0.02em', color: '#f8fafc' }}>
                HeteroSched
              </h1>
              <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                v1.0 • C++17/CUDA
              </span>
            </div>
            <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.15rem' }}>
              Adaptive Two-Tier Heterogeneous CPU-GPU Task Scheduling Engine
            </p>
          </div>
        </div>

        {/* Hardware & Telemetry Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* GPU Hardware Badge */}
          <div style={{
            background: 'rgba(15, 23, 42, 0.7)',
            border: '1px solid rgba(16, 185, 129, 0.25)',
            borderRadius: '8px',
            padding: '0.4rem 0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.8rem'
          }}>
            <HardDrive size={15} color="#10b981" />
            <div>
              <span style={{ color: '#10b981', fontWeight: '600' }}>{gpu.name}</span>
              <span style={{ color: '#64748b', marginLeft: '0.35rem' }}>
                ({gpu.global_memory_gb}GB • CC {gpu.compute_capability} • {gpu.sm_count} SMs)
              </span>
            </div>
          </div>

          {/* Host & Worker Threads */}
          <div style={{
            background: 'rgba(15, 23, 42, 0.7)',
            border: '1px solid rgba(59, 130, 246, 0.25)',
            borderRadius: '8px',
            padding: '0.4rem 0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.8rem'
          }}>
            <Cpu size={15} color="#60a5fa" />
            <span style={{ color: '#93c5fd' }}>4 CPU Workers • 4 CUDA Streams</span>
          </div>

          {/* Live Execution Status */}
          <div style={{
            background: isRunning ? 'rgba(245, 158, 11, 0.15)' : (systemInfo?.is_local_gpu ? 'rgba(16, 185, 129, 0.12)' : 'rgba(56, 189, 248, 0.12)'),
            border: `1px solid ${isRunning ? 'rgba(245, 158, 11, 0.4)' : (systemInfo?.is_local_gpu ? 'rgba(16, 185, 129, 0.3)' : 'rgba(56, 189, 248, 0.3)')}`,
            borderRadius: '8px',
            padding: '0.4rem 0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.8rem'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isRunning ? '#f59e0b' : (systemInfo?.is_local_gpu ? '#10b981' : '#38bdf8'),
              display: 'inline-block',
              boxShadow: isRunning ? '0 0 8px #f59e0b' : (systemInfo?.is_local_gpu ? '0 0 8px #10b981' : '0 0 8px #38bdf8')
            }} className={isRunning ? 'animate-pulse' : 'animate-pulse-glow'} />
            <span style={{ color: isRunning ? '#fbbf24' : (systemInfo?.is_local_gpu ? '#34d399' : '#38bdf8'), fontWeight: '600' }}>
              {isRunning
                ? 'Executing Real CUDA Benchmark on RTX 4050...'
                : (systemInfo?.is_local_gpu
                    ? 'Local GPU Engine Active (RTX 4050)'
                    : 'Cloud Demo Mode (Vercel Serverless)')}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
