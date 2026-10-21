import React, { useState, useEffect, useCallback } from 'react';
import SystemHeader from './components/SystemHeader';
import TraceSelector from './components/TraceSelector';
import MetricCards from './components/MetricCards';
import Visualizations from './components/Visualizations';
import TaskTable from './components/TaskTable';
import { fetchSystemInfo, fetchTraces, fetchLatestBenchmark, runBenchmark } from './api/client';
import { RefreshCw, AlertTriangle } from 'lucide-react';

export default function App() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [traces, setTraces] = useState({});
  const [loadingTraces, setLoadingTraces] = useState(true);

  const [selectedTrace, setSelectedTrace] = useState('mixed');
  const [latestRun, setLatestRun] = useState(null);
  const [loadingLatestRun, setLoadingLatestRun] = useState(true);

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

  // Load system info, traces, and latest benchmark independently
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    const { signal } = controller;

    // 1. Fetch system info independently
    fetchSystemInfo({ signal })
      .then((sys) => {
        if (!isMounted) return;
        setSystemInfo(sys);
        setSystemError(null);
      })
      .catch((err) => {
        if (!isMounted || err.name === 'AbortError') return;
        console.warn('Could not fetch system info:', err);
        setSystemError(err.message || 'Failed to connect to HeteroSched backend.');
      })
      .finally(() => {
        if (isMounted) setLoadingSystem(false);
      });

    // 2. Fetch approved traces independently
    fetchTraces({ signal })
      .then((traceData) => {
        if (!isMounted) return;
        if (traceData?.traces) {
          setTraces(traceData.traces);
        }
      })
      .catch((err) => {
        if (!isMounted || err.name === 'AbortError') return;
        console.warn('Could not fetch traces:', err);
      })
      .finally(() => {
        if (isMounted) setLoadingTraces(false);
      });

    // 3. Fetch latest benchmark results independently
    fetchLatestBenchmark({ signal })
      .then((latest) => {
        if (!isMounted) return;
        if (latest) {
          setLatestRun(latest);
          if (latest.trace_key) {
            setSelectedTrace(latest.trace_key);
          }
        }
      })
      .catch((err) => {
        if (!isMounted || err.name === 'AbortError') return;
        console.warn('Could not fetch latest benchmark:', err);
      })
      .finally(() => {
        if (isMounted) setLoadingLatestRun(false);
      });

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, []);

  const handleRunBenchmark = useCallback(async () => {
    if (isRunning) return;
    try {
      setIsRunning(true);
      setError(null);
      const runResult = await runBenchmark(selectedTrace);
      setLatestRun(runResult);
    } catch (err) {
      setError(err.message || 'Benchmark execution failed.');
    } finally {
      setIsRunning(false);
    }
  }, [isRunning, selectedTrace]);

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '1.5rem 2rem' }}>
      {/* Top Header */}
      <SystemHeader
        systemInfo={systemInfo}
        isRunning={isRunning}
        isFresh={latestRun?.is_fresh}
        source={latestRun?.source}
        timestamp={latestRun?.timestamp_iso}
      />

      {/* Backend Disconnected Warning (only displayed after system probe finishes and confirmed offline) */}
      {!loadingSystem && !systemInfo && systemError && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.12)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '8px',
          padding: '0.85rem 1.25rem',
          color: '#fbbf24',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.875rem',
        }}>
          <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0 }} />
          <div>
            <strong>Backend API Offline: </strong>
            Please start the Flask backend by running{' '}
            <code className="font-mono" style={{ background: 'rgba(0,0,0,0.3)', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>
              python3 -m server.app
            </code>{' '}
            in WSL Ubuntu 24.04 (port 5000).
          </div>
        </div>
      )}

      {/* Control Panel: Workload Trace Selector & Run Button */}
      <TraceSelector
        traces={traces}
        selectedTrace={selectedTrace}
        onSelectTrace={setSelectedTrace}
        onRunBenchmark={handleRunBenchmark}
        isRunning={isRunning}
        latestRun={latestRun}
        error={error}
        isLoadingTraces={loadingTraces}
        isLoadingLatest={loadingLatestRun}
        systemInfo={systemInfo}
      />

      {/* Benchmark Results Area */}
      {loadingLatestRun && !latestRun ? (
        <div className="glass-panel" style={{
          padding: '3rem 2rem',
          textAlign: 'center',
          marginBottom: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <RefreshCw size={28} color="#38bdf8" className="animate-spin" style={{ marginBottom: '0.85rem' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#f8fafc' }}>
            Loading Benchmark Results...
          </h3>
          <p style={{ fontSize: '0.8125rem', color: '#64748b', marginTop: '0.35rem' }}>
            Retrieving baseline telemetry and task execution data
          </p>
        </div>
      ) : latestRun?.summary ? (
        <>
          {/* Aggregated Performance Metric Cards */}
          <MetricCards summary={latestRun.summary} />

          {/* Analytical Visualizations */}
          <Visualizations summary={latestRun.summary} />

          {/* Searchable and Filterable Task Table */}
          {latestRun?.tasks && (
            <TaskTable tasks={latestRun.tasks} />
          )}
        </>
      ) : !isRunning && (
        <div className="glass-panel" style={{
          padding: '2.5rem 2rem',
          textAlign: 'center',
          marginBottom: '1.5rem',
          color: '#94a3b8',
        }}>
          <p style={{ fontSize: '0.9rem' }}>
            No benchmark results available yet. Select a workload trace above and click <strong>Execute Real Benchmark</strong> to begin.
          </p>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        marginTop: '3rem',
        paddingTop: '1.5rem',
        borderTop: '1px solid rgba(148, 163, 184, 0.12)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        fontSize: '0.8rem',
        color: '#64748b',
      }}>
        <div>
          HeteroSched Heterogeneous Computing Scheduler • NVIDIA RTX 4050 • WSL2 Ubuntu 24.04
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <span>C++17 / CUDA 12</span>
          <span>•</span>
          <span>Two-Tier Fast-Path GBDT</span>
          <span>•</span>
          <span>Strict Local Execution</span>
        </div>
      </footer>
    </div>
  );
}
