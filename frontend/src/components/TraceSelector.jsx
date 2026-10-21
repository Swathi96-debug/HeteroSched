import React from 'react';
import { Play, Loader2, Sparkles, Database, AlertCircle } from 'lucide-react';

export default function TraceSelector({
  traces = {},
  selectedTrace,
  onSelectTrace,
  onRunBenchmark,
  isRunning,
  latestRun,
  error,
  isLoadingTraces = false,
  isLoadingLatest = false,
  systemInfo = null,
}) {
  const isFresh = latestRun?.is_fresh;
  const timestamp = latestRun?.timestamp_iso;
  const runId = latestRun?.run_id;
  const elapsedSec = latestRun?.elapsed_seconds;

  const traceEntries = Object.entries(traces || {});

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem' }}>
        {/* Workload Trace Selector */}
        <div style={{ flex: '1 1 500px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.85rem' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: '600', color: '#f8fafc', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Select Approved Workload Trace
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>(Strictly Whitelisted)</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem' }}>
            {isLoadingTraces && traceEntries.length === 0 ? (
              // Skeleton cards while traces are fetching
              [1, 2, 3].map((i) => (
                <div
                  key={`skeleton-trace-${i}`}
                  style={{
                    background: 'rgba(15, 23, 42, 0.5)',
                    border: '1px solid rgba(148, 163, 184, 0.12)',
                    borderRadius: '10px',
                    padding: '1rem',
                    minHeight: '118px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
                    <div style={{ width: '55%', height: '14px', background: 'rgba(148, 163, 184, 0.2)', borderRadius: '4px' }} className="animate-pulse" />
                    <div style={{ width: '25%', height: '14px', background: 'rgba(148, 163, 184, 0.15)', borderRadius: '4px' }} className="animate-pulse" />
                  </div>
                  <div style={{ width: '90%', height: '11px', background: 'rgba(148, 163, 184, 0.1)', borderRadius: '4px', marginBottom: '0.4rem' }} className="animate-pulse" />
                  <div style={{ width: '70%', height: '11px', background: 'rgba(148, 163, 184, 0.1)', borderRadius: '4px' }} className="animate-pulse" />
                </div>
              ))
            ) : (
              traceEntries.map(([key, info]) => {
                const isSelected = selectedTrace === key;
                return (
                  <div
                    key={key}
                    onClick={() => !isRunning && onSelectTrace(key)}
                    style={{
                      background: isSelected ? 'rgba(56, 189, 248, 0.12)' : 'rgba(15, 23, 42, 0.5)',
                      border: `1px solid ${isSelected ? '#38bdf8' : 'rgba(148, 163, 184, 0.15)'}`,
                      borderRadius: '10px',
                      padding: '1rem',
                      cursor: isRunning ? 'not-allowed' : 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: isSelected ? '0 0 15px rgba(56, 189, 248, 0.2)' : 'none',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                      <h3 style={{ fontSize: '0.925rem', fontWeight: '600', color: isSelected ? '#38bdf8' : '#e2e8f0' }}>
                        {info.name}
                      </h3>
                      <span className="badge" style={{
                        background: 'rgba(255, 255, 255, 0.06)',
                        color: '#94a3b8',
                        fontSize: '0.7rem'
                      }}>
                        {info.task_count} tasks
                      </span>
                    </div>
                    <p style={{ fontSize: '0.775rem', color: '#94a3b8', lineHeight: '1.4', marginBottom: '0.65rem' }}>
                      {info.description}
                    </p>
                    <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                      {info.tags?.map((tag) => (
                        <span
                          key={tag}
                          style={{
                            fontSize: '0.65rem',
                            padding: '0.15rem 0.45rem',
                            borderRadius: '4px',
                            background: isSelected ? 'rgba(56, 189, 248, 0.2)' : 'rgba(30, 41, 59, 0.8)',
                            color: isSelected ? '#bae6fd' : '#94a3b8',
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Execution Trigger & Run Status */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', justifyContent: 'center', gap: '1rem', minWidth: '240px' }}>
          <button
            className="btn-primary"
            onClick={onRunBenchmark}
            disabled={isRunning}
            style={{ width: '100%', justifyContent: 'center', padding: '0.85rem 1.5rem', fontSize: '0.95rem' }}
          >
            {isRunning ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                <span>Running CUDA Benchmark on RTX 4050...</span>
              </>
            ) : systemInfo && systemInfo.is_local_gpu === false ? (
              <>
                <Play size={18} fill="#ffffff" />
                <span>Execute Benchmark (Cloud Demo)</span>
              </>
            ) : (
              <>
                <Play size={18} fill="#ffffff" />
                <span>Execute Real Benchmark (RTX 4050)</span>
              </>
            )}
          </button>
          {systemInfo && systemInfo.is_local_gpu === false && (
            <div style={{ fontSize: '0.75rem', color: '#38bdf8', textAlign: 'center', maxWidth: '240px', lineHeight: '1.2' }}>
              Cloud demo mode. For live GPU execution, launch locally via <code style={{ color: '#f8fafc' }}>./scripts/launch_dashboard.sh</code>.
            </div>
          )}

          {/* Provenance Badge: Fresh Run vs Saved Results */}
          <div style={{
            width: '100%',
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid rgba(148, 163, 184, 0.15)',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            fontSize: '0.8rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span style={{ color: '#64748b', fontWeight: '500' }}>Results State:</span>
              {isLoadingLatest && !latestRun ? (
                <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
                  <Loader2 size={12} className="animate-spin" /> Loading Baseline...
                </span>
              ) : isFresh ? (
                <span className="badge badge-fresh">
                  <Sparkles size={12} /> Fresh Real-time Run
                </span>
              ) : latestRun ? (
                <span className="badge badge-saved">
                  <Database size={12} /> Previously Saved Baseline
                </span>
              ) : (
                <span className="badge" style={{ background: 'rgba(148, 163, 184, 0.1)', color: '#94a3b8' }}>
                  No Baseline Loaded
                </span>
              )}
            </div>

            {runId && (
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.2rem' }}>
                <span style={{ color: '#64748b' }}>Run ID:</span>
                <span className="font-mono" style={{ color: '#cbd5e1' }}>{runId}</span>
              </div>
            )}

            {timestamp && (
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.2rem' }}>
                <span style={{ color: '#64748b' }}>Recorded At:</span>
                <span>{new Date(timestamp).toLocaleTimeString()} ({new Date(timestamp).toLocaleDateString()})</span>
              </div>
            )}

            {latestRun?.summary?.total_wall_time_ms !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.2rem' }}>
                <span style={{ color: '#64748b' }} title="Pure C++ task scheduling and execution duration">Scheduler Wall Time:</span>
                <span className="font-mono" style={{ color: '#10b981', fontWeight: '600' }}>
                  {(latestRun.summary.total_wall_time_ms / 1000).toFixed(2)} s ({latestRun.summary.total_wall_time_ms.toFixed(1)} ms)
                </span>
              </div>
            )}

            {elapsedSec !== undefined && elapsedSec > 0 && (
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.2rem' }}>
                <span style={{ color: '#64748b' }} title="Full process lifetime including binary launch, CUDA initialization, trace loading, and buffer allocation">Process Total Duration:</span>
                <span className="font-mono" style={{ color: '#38bdf8' }}>{elapsedSec} s</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem 1rem',
          background: 'rgba(244, 63, 94, 0.12)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: '8px',
          color: '#f87171',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
