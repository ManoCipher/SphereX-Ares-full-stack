import { useState, useEffect } from 'react';
import type { Target, FitsRecord, SedPoint, SmartSelectResult } from './types';
import { BlinkCanvas } from './components/BlinkCanvas';
import { TargetSelector } from './components/TargetSelector';
import { TemporalControls } from './components/TemporalControls';
import { SpectralControls } from './components/SpectralControls';
import { ImageGallery } from './components/ImageGallery';
import { CandidateModal } from './components/CandidateModal';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export function App() {
  const [targets, setTargets] = useState<Target[]>([]);
  const [selectedTargetId, setSelectedTargetId] = useState<string>('');
  const [fetchedExposures, setFetchedExposures] = useState<FitsRecord[]>([]);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [isSmartSelecting, setIsSmartSelecting] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [smartExplanation, setSmartExplanation] = useState<string>('');

  // Coordinates: empty by default until user selects a preset or inputs coordinates
  const [ra, setRa] = useState<string>('');
  const [dec, setDec] = useState<string>('');
  const [radius, setRadius] = useState<string>('0.08');
  const [maxResults, setMaxResults] = useState<string>('6');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  // Observation Modes: 'temporal' (default) | 'spectral'
  const [activeDomain, setActiveDomain] = useState<'temporal' | 'spectral'>('temporal');
  const [isBlinkActive, setIsBlinkActive] = useState<boolean>(false);
  const [epochIdx, setEpochIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(2);

  // Calibration Controls (Default: inferno for fast single-band display; RGB reserved for spectral)
  const [stretch, setStretch] = useState<string>('asinh');
  const [colormap, setColormap] = useState<string>('inferno');
  const [minCutPercent, setMinCutPercent] = useState<number>(2);
  const [maxCutPercent, setMaxCutPercent] = useState<number>(98);
  const [selectedDetector, setSelectedDetector] = useState<string>('all');
  const [cleanImage, setCleanImage] = useState<boolean>(false);
  const [isDifferenceMode, setIsDifferenceMode] = useState<boolean>(false);
  const [invert, setInvert] = useState<boolean>(false);

  // Candidate Investigation State
  const [investigationMarker, setInvestigationMarker] = useState<{ x: number; y: number } | null>(null);
  const [candidateReport, setCandidateReport] = useState<any>(null);
  const [isInvestigating, setIsInvestigating] = useState<boolean>(false);

  // Hyperspectral features
  const [rgbMode, setRgbMode] = useState<boolean>(false);
  const [sedData, setSedData] = useState<SedPoint[]>([]);

  // 1. Initial Load: Fetch targets metadata (presets)
  useEffect(() => {
    fetch(`${API_BASE}/api/targets`)
      .then((res) => res.json())
      .then((data: Target[]) => {
        if (Array.isArray(data)) {
          setTargets(data);
          setBackendConnected(true);
        }
      })
      .catch(() => setBackendConnected(false));
  }, []);

  // 2. Fetch Spectral SED data when switching to Spectral mode
  useEffect(() => {
    if (activeDomain === 'spectral' && fetchedExposures.length > 0) {
      fetch(`${API_BASE}/api/sed`)
        .then((res) => res.json())
        .then((data: SedPoint[]) => {
          if (Array.isArray(data)) setSedData(data);
        })
        .catch(() => {});
    }
  }, [activeDomain, fetchedExposures]);

  // 3. User Query Action: Perform search and ingest fresh cutouts
  const handlePerformQuery = () => {
    if (!ra.trim() || !dec.trim()) return;

    setIsQuerying(true);
    setLoadingMessage('Querying NASA/IPAC IRSA SIA2 Archive...');
    setStatusMessage('');
    setSmartExplanation('');
    setInvestigationMarker(null);
    setCandidateReport(null);

    const queryParams = new URLSearchParams({
      ra,
      dec,
      radius,
      max_results: maxResults,
      ...(startDate ? { start_date: startDate } : {}),
      ...(endDate ? { end_date: endDate } : {}),
    });

    fetch(`${API_BASE}/api/search?${queryParams}`)
      .then((res) => res.json())
      .then(() => {
        setLoadingMessage('Downloading bounded FITS cutouts from IRSA IBE...');
        const mode = activeDomain === 'spectral' ? 'spectral' : 'time';
        return fetch(`${API_BASE}/api/fetch_cutouts?ra=${ra}&dec=${dec}&radius=${radius}&max_files=${maxResults}&mode=${mode}`);
      })
      .then((res) => res.json())
      .then((data) => {
        if (data.records && Array.isArray(data.records) && data.records.length > 0) {
          setFetchedExposures(data.records);
          setEpochIdx(0);
          setStatusMessage(`Successfully loaded ${data.records.length} observation cutout(s).`);
        } else {
          setStatusMessage('No SPHEREx observations found for these coordinates. Try a benchmark preset.');
        }
        setIsQuerying(false);
      })
      .catch(() => {
        setStatusMessage('Network request to backend failed. Please verify API server status.');
        setIsQuerying(false);
      });
  };

  // 4. Smart Observation Selection Handler
  const handleSmartSelect = () => {
    if (!ra.trim() || !dec.trim()) return;

    setIsSmartSelecting(true);
    setLoadingMessage('Running astronomical multi-epoch observation selector...');
    setStatusMessage('');
    setSmartExplanation('');
    setInvestigationMarker(null);
    setCandidateReport(null);

    const mode = activeDomain === 'spectral' ? 'spectral' : 'time';
    fetch(`${API_BASE}/api/smart_select?ra=${ra}&dec=${dec}&radius=${radius}&mode=${mode}`)
      .then((res) => res.json())
      .then((data: SmartSelectResult) => {
        if (data?.explanation) {
          setSmartExplanation(data.explanation);
        }
        setLoadingMessage('Downloading prioritized FITS cutouts from NASA IRSA...');
        return fetch(`${API_BASE}/api/fetch_cutouts?ra=${ra}&dec=${dec}&radius=${radius}&max_files=${maxResults}&smart=1&mode=${mode}`);
      })
      .then((res) => res.json())
      .then((data) => {
        if (data.records && Array.isArray(data.records) && data.records.length > 0) {
          setFetchedExposures(data.records);
          setEpochIdx(0);
          setStatusMessage(`Loaded ${data.records.length} optimal observation cutout(s).`);
        } else {
          setStatusMessage('No matching observations found for smart selection.');
        }
        setIsSmartSelecting(false);
      })
      .catch(() => {
        setStatusMessage('Smart selection failed. Please verify server connectivity.');
        setIsSmartSelecting(false);
      });
  };

  // Preset Selection Handlers
  const handleSelectTarget = (t: Target) => {
    setSelectedTargetId(t.id);
    setRa(t.ra_deg.toFixed(4));
    setDec(t.dec_deg.toFixed(4));
    setEpochIdx(0);
    setInvestigationMarker(null);
    setCandidateReport(null);
    setSmartExplanation('');
    setStatusMessage(`Target ${t.name} selected. Click Fetch FITS or Auto Select.`);
  };

  const handleClearTarget = () => {
    setSelectedTargetId('');
    setRa('');
    setDec('');
    setEpochIdx(0);
    setInvestigationMarker(null);
    setCandidateReport(null);
    setSmartExplanation('');
    setStatusMessage('');
  };

  // 5. Candidate Click Investigation Handler
  const handleCanvasClick = (normX: number, normY: number) => {
    if (fetchedExposures.length === 0) return;
    setInvestigationMarker({ x: normX, y: normY });
    setIsInvestigating(true);
    fetch(`${API_BASE}/api/investigate?x=${normX}&y=${normY}&frame=${epochIdx}`)
      .then((res) => res.json())
      .then((report) => {
        setCandidateReport(report);
        setIsInvestigating(false);
      })
      .catch(() => setIsInvestigating(false));
  };

  // Auto-focus on primary detector for temporal blinking when multiple detectors exist
  useEffect(() => {
    if (fetchedExposures.length > 0 && activeDomain === 'temporal') {
      const detectors = [...new Set(fetchedExposures.map((f) => String(f.detector)))];
      if (detectors.length > 1 && selectedDetector === 'all') {
        setSelectedDetector(detectors[0]);
      }
    }
  }, [fetchedExposures, activeDomain]);

  // Filter fetched exposures by detector
  const filteredExposures = selectedDetector === 'all'
    ? fetchedExposures
    : fetchedExposures.filter((f) => String(f.detector) === String(selectedDetector));

  const totalEpochs = filteredExposures.length > 0 ? filteredExposures.length : 1;
  const currentExposure = filteredExposures[epochIdx] || fetchedExposures[0];
  const activeFitsFile = currentExposure?.filename || '';

  // Playback timer loop: ONLY runs when user has explicitly enabled Blink Mode and hit Play
  useEffect(() => {
    if (!isBlinkActive || !isPlaying || totalEpochs <= 1 || activeDomain !== 'temporal') return;
    const interval = setInterval(() => {
      setEpochIdx((prev) => (prev + 1) % totalEpochs);
    }, 1000 / fps);
    return () => clearInterval(interval);
  }, [isBlinkActive, isPlaying, fps, totalEpochs, activeDomain]);

  // Streaming Image URL: Difference mode, single-band (inferno/viridis/gray), or RGB in spectral
  let imageSrc: string = '';
  if (filteredExposures.length > 0) {
    if (isDifferenceMode && activeDomain === 'temporal') {
      const diffLast = totalEpochs > 1 ? (epochIdx > 0 ? epochIdx : totalEpochs - 1) : 1;
      imageSrc = `${API_BASE}/api/difference?first=0&last=${diffLast}&colormap=${colormap === 'rgb' ? 'coolwarm' : colormap}&clean=${cleanImage ? '1' : '0'}&invert=${invert ? '1' : '0'}`;
    } else if (activeDomain === 'spectral' && rgbMode) {
      imageSrc = `${API_BASE}/api/rgb`;
    } else {
      imageSrc = `${API_BASE}/api/render?filename=${encodeURIComponent(activeFitsFile)}&colormap=${colormap}&stretch=${stretch}&min_cut_percent=${minCutPercent}&max_cut_percent=${maxCutPercent}&clean=${cleanImage ? '1' : '0'}&invert=${invert ? '1' : '0'}`;
    }
  }

  const isBusy = isQuerying || isSmartSelecting;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#070b14', color: '#f1f5f9', display: 'flex', flexDirection: 'column' }}>
      {/* Clean Header */}
      <header style={{
        padding: '12px 24px',
        backgroundColor: '#0c1322',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <img
            src="/ares-nebula-logo.jpg"
            alt="Ares Nebula"
            style={{ width: '38px', height: '38px', borderRadius: '50%', objectFit: 'cover', border: '1px solid #38bdf8', boxShadow: '0 0 10px rgba(56, 189, 248, 0.3)' }}
          />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '16px', fontWeight: 'bold', margin: 0, color: '#fff', letterSpacing: '0.04em' }}>
                ARES NEBULA <span style={{ color: '#94a3b8', fontWeight: 'normal' }}>| SPHEREx Portal</span>
              </h1>
              <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', backgroundColor: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.3)', fontFamily: 'monospace' }}>
                NASA Space Apps
              </span>
            </div>
            <p style={{ fontSize: '11px', color: '#64748b', margin: '2px 0 0 0', letterSpacing: '0.08em' }}>
              EXPLORE • ANALYZE • DISCOVER
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontFamily: 'monospace' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: backendConnected ? '#10b981' : '#f59e0b' }} />
          <span style={{ color: backendConnected ? '#10b981' : '#f59e0b' }}>
            {backendConnected ? 'BACKEND: READY' : 'BACKEND: OFFLINE'}
          </span>
        </div>
      </header>

      {/* Main Workspace */}
      <main style={{ maxWidth: '1200px', width: '100%', margin: '0 auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Step 1: Target Definition & Search */}
        <TargetSelector
          targets={targets}
          selectedTargetId={selectedTargetId}
          onSelectTarget={handleSelectTarget}
          onClearTarget={handleClearTarget}
          ra={ra}
          dec={dec}
          radius={radius}
          maxResults={maxResults}
          startDate={startDate}
          endDate={endDate}
          onChangeRa={setRa}
          onChangeDec={setDec}
          onChangeRadius={setRadius}
          onChangeMaxResults={setMaxResults}
          onChangeStartDate={setStartDate}
          onChangeEndDate={setEndDate}
          onQuery={handlePerformQuery}
          isQuerying={isQuerying}
          onSmartSelect={handleSmartSelect}
          isSmartSelecting={isSmartSelecting}
          smartExplanation={smartExplanation}
          statusMessage={statusMessage}
        />

        {/* Queried Image Gallery Strip */}
        {filteredExposures.length > 0 && (
          <ImageGallery
            exposures={filteredExposures}
            activeFilename={activeFitsFile}
            onSelectExposure={(_exp, idx) => {
              setEpochIdx(idx);
              setIsPlaying(false);
            }}
          />
        )}

        {/* Observation Dimension Toggle (Temporal vs Hyperspectral) */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
          paddingBottom: '8px',
          borderBottom: '1px solid #1e293b',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold' }}>Observation Mode:</span>
            <div style={{ display: 'inline-flex', backgroundColor: '#0c1322', border: '1px solid #1e293b', borderRadius: '6px', padding: '2px' }}>
              <button
                onClick={() => { setActiveDomain('temporal'); setRgbMode(false); }}
                title="Single exposure inspection and multi-epoch temporal astrometry"
                style={{
                  padding: '5px 12px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 'bold',
                  fontFamily: 'monospace',
                  backgroundColor: activeDomain === 'temporal' ? '#38bdf8' : 'transparent',
                  color: activeDomain === 'temporal' ? '#000' : '#94a3b8',
                }}
              >
                Temporal Astrometry
              </button>
              <button
                onClick={() => { setActiveDomain('spectral'); setIsPlaying(false); setIsBlinkActive(false); }}
                title="102-channel hyperspectral slicing and spectral energy distribution (SED)"
                style={{
                  padding: '5px 12px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 'bold',
                  fontFamily: 'monospace',
                  backgroundColor: activeDomain === 'spectral' ? '#a855f7' : 'transparent',
                  color: activeDomain === 'spectral' ? '#fff' : '#94a3b8',
                }}
              >
                Hyperspectral Slicing
              </button>
            </div>
          </div>

          <div style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>
            {filteredExposures.length > 0
              ? `${filteredExposures.length} WCS-Aligned Cutout(s) Loaded`
              : 'Awaiting observation query'}
          </div>
        </div>

        {/* Workspace: Unobstructed Canvas & Responsive Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
          {/* Left: Clean FITS Canvas */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <BlinkCanvas
              imageSrc={imageSrc}
              activeFilename={activeFitsFile}
              investigationMarker={investigationMarker}
              onCanvasClick={handleCanvasClick}
              isLoading={isBusy}
              loadingMessage={loadingMessage}
              isEmpty={filteredExposures.length === 0}
            />
          </div>

          {/* Right: Dimension & Calibration Controls */}
          <div>
            {activeDomain === 'temporal' ? (
              <TemporalControls
                isBlinkActive={isBlinkActive}
                onToggleBlink={() => {
                  const next = !isBlinkActive;
                  setIsBlinkActive(next);
                  if (!next) setIsPlaying(false);
                }}
                isPlaying={isPlaying}
                onTogglePlay={() => setIsPlaying(!isPlaying)}
                onPrev={() => { setIsPlaying(false); setEpochIdx((prev) => (prev - 1 + totalEpochs) % totalEpochs); }}
                onNext={() => { setIsPlaying(false); setEpochIdx((prev) => (prev + 1) % totalEpochs); }}
                fps={fps}
                onChangeFps={setFps}
                epochIdx={epochIdx}
                totalEpochs={totalEpochs}
                onScrub={(idx) => { setIsPlaying(false); setEpochIdx(idx); }}
                curFilename={activeFitsFile}
                stretch={stretch}
                onChangeStretch={setStretch}
                minCutPercent={minCutPercent}
                onChangeMinCutPercent={setMinCutPercent}
                maxCutPercent={maxCutPercent}
                onChangeMaxCutPercent={setMaxCutPercent}
                colormap={colormap}
                onChangeColormap={setColormap}
                selectedDetector={selectedDetector}
                onChangeDetector={(d) => { setSelectedDetector(d); setEpochIdx(0); }}
                cleanEnabled={cleanImage}
                onToggleClean={() => setCleanImage((prev) => !prev)}
                isDifferenceMode={isDifferenceMode}
                onToggleDifferenceMode={() => {
                  setIsDifferenceMode((prev) => !prev);
                  if (isBlinkActive) setIsPlaying(false);
                }}
                invert={invert}
                onToggleInvert={() => setInvert((prev) => !prev)}
                epochDate={currentExposure?.date_obs}
                epochDetector={currentExposure?.detector}
              />
            ) : (
              <SpectralControls
                rgbMode={rgbMode}
                onToggleRgb={() => setRgbMode(!rgbMode)}
                sedData={sedData}
              />
            )}
          </div>
        </div>
      </main>

      {/* Candidate Investigation Report Modal */}
      <CandidateModal
        report={candidateReport}
        isLoading={isInvestigating}
        onClose={() => setCandidateReport(null)}
      />
    </div>
  );
}

export default App;