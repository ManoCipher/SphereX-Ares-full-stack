import React from 'react';
import type { SedPoint } from '../types';

interface SpectralControlsProps {
  rgbMode: boolean;
  onToggleRgb: () => void;
  sedData: SedPoint[];
  coaddDetector: string;
  onChangeCoaddDetector: (d: string) => void;
  spectrumSourceCoord?: { x: number; y: number } | null;
}

export const SpectralControls: React.FC<SpectralControlsProps> = ({
  rgbMode,
  onToggleRgb,
  sedData,
  coaddDetector,
  onChangeCoaddDetector,
  spectrumSourceCoord,
}) => {
  const maxFlux = Math.max(...sedData.map((d) => d.flux), 1.0);
  const minFlux = Math.min(...sedData.map((d) => d.flux), 0.0);
  const fluxSpan = maxFlux - minFlux || 1.0;

  // Build SVG path for on-the-fly smooth continuous spectrum
  const svgWidth = 400;
  const svgHeight = 110;
  const padLeft = 35;
  const padRight = 15;
  const padTop = 15;
  const padBottom = 25;
  const plotW = svgWidth - padLeft - padRight;
  const plotH = svgHeight - padTop - padBottom;

  const points = sedData.map((pt, i) => {
    const x = padLeft + (i / Math.max(sedData.length - 1, 1)) * plotW;
    const y = padTop + plotH - ((pt.flux - minFlux) / fluxSpan) * plotH;
    return { x, y, pt };
  });

  const pathD = points.length > 0
    ? points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
    : '';

  return (
    <div style={{
      padding: '14px',
      backgroundColor: '#0c1322',
      borderRadius: '8px',
      border: '1px solid #1e293b',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      fontSize: '12px',
      fontFamily: 'monospace',
    }}>
      {/* Header and Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <span style={{ color: '#c084fc', fontWeight: 'bold', fontSize: '13px' }}>
            102-Channel Hyperspectral Diagnostics
          </span>
          <div style={{ color: '#64748b', fontSize: '10px', marginTop: '2px' }}>
            SPHEREx Linear Variable Filter (0.75µm - 5.0µm)
          </div>
        </div>

        <button
          onClick={onToggleRgb}
          title="Compose multi-band false color image from blue, green, and red SPHEREx detector channels"
          style={{
            padding: '6px 12px',
            backgroundColor: rgbMode ? '#10b981' : '#1e293b',
            color: rgbMode ? '#000' : '#fff',
            fontWeight: 'bold',
            borderRadius: '4px',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          {rgbMode ? 'RGB Active' : 'RGB Composite'}
        </button>
      </div>

      {/* NASA Suggestion: 6-Array Co-addition (Stacking by Detector to Boost SNR) */}
      <div style={{
        padding: '10px 12px',
        backgroundColor: '#070b14',
        borderRadius: '6px',
        border: '1px solid #1e293b',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ color: '#94a3b8', fontSize: '11px', fontWeight: 'bold' }}>
            Detector Array Co-addition (Stack to Boost SNR)
          </span>
          <span style={{ color: '#38bdf8', fontSize: '10px' }}>
            {coaddDetector === 'off' ? 'Single Cutout' : `Array ${coaddDetector} Stack`}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(75px, 1fr))', gap: '6px' }}>
          {[
            { id: 'off', label: 'Raw Slices' },
            { id: '1', label: 'A1 (0.75-1.1µ)' },
            { id: '2', label: 'A2 (1.1-1.6µ)' },
            { id: '3', label: 'A3 (1.6-2.4µ)' },
            { id: '4', label: 'A4 (2.4-3.8µ)' },
            { id: '5', label: 'A5 (3.8-4.4µ)' },
            { id: '6', label: 'A6 (4.4-5.0µ)' },
          ].map((arr) => {
            const active = coaddDetector === arr.id;
            return (
              <button
                key={arr.id}
                onClick={() => onChangeCoaddDetector(arr.id)}
                title={arr.id === 'off' ? 'View individual spectral slices' : `Add together all channel frames in Array ${arr.id} to improve SNR`}
                style={{
                  padding: '5px 4px',
                  borderRadius: '4px',
                  border: active ? '1px solid #a855f7' : '1px solid #1e293b',
                  backgroundColor: active ? 'rgba(168, 85, 247, 0.2)' : '#0c1322',
                  color: active ? '#c084fc' : '#94a3b8',
                  fontSize: '10px',
                  cursor: 'pointer',
                  fontWeight: active ? 'bold' : 'normal',
                }}
              >
                {arr.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* On-The-Fly Extracted Spectrum (SED) */}
      <div style={{
        padding: '12px',
        backgroundColor: '#070b14',
        borderRadius: '6px',
        border: '1px solid #1e293b',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ color: '#d8b4fe', fontWeight: 'bold' }}>On-the-Fly Extracted Spectrum</span>
            <div style={{ color: '#64748b', fontSize: '10px' }}>
              {spectrumSourceCoord
                ? `Sampled at canvas (${Math.round(spectrumSourceCoord.x * 100)}%, ${Math.round(spectrumSourceCoord.y * 100)}%)`
                : 'Center aperture sample (Click any canvas pixel to sample on-the-fly)'}
            </div>
          </div>
          <span style={{ color: '#94a3b8', fontSize: '10px' }}>
            {sedData.length} Spectral Bands
          </span>
        </div>

        {sedData.length > 0 ? (
          <div style={{ width: '100%', overflow: 'hidden' }}>
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ width: '100%', height: 'auto', display: 'block' }}
            >
              {/* Grid Lines */}
              <line x1={padLeft} y1={padTop} x2={padLeft + plotW} y2={padTop} stroke="#1e293b" strokeDasharray="3 3" />
              <line x1={padLeft} y1={padTop + plotH / 2} x2={padLeft + plotW} y2={padTop + plotH / 2} stroke="#1e293b" strokeDasharray="3 3" />
              <line x1={padLeft} y1={padTop + plotH} x2={padLeft + plotW} y2={padTop + plotH} stroke="#334155" />
              <line x1={padLeft} y1={padTop} x2={padLeft} y2={padTop + plotH} stroke="#334155" />

              {/* Y-Axis Labels */}
              <text x={padLeft - 6} y={padTop + 4} fill="#64748b" fontSize="8" textAnchor="end">{maxFlux.toFixed(1)}</text>
              <text x={padLeft - 6} y={padTop + plotH} fill="#64748b" fontSize="8" textAnchor="end">{minFlux.toFixed(1)}</text>

              {/* X-Axis Labels (Wavelengths) */}
              <text x={padLeft} y={svgHeight - 6} fill="#64748b" fontSize="8">0.75µ</text>
              <text x={padLeft + plotW / 2} y={svgHeight - 6} fill="#64748b" fontSize="8" textAnchor="middle">2.8µ</text>
              <text x={padLeft + plotW} y={svgHeight - 6} fill="#64748b" fontSize="8" textAnchor="end">5.0µm</text>

              {/* Area Gradient fill */}
              <defs>
                <linearGradient id="sedGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#a855f7" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#a855f7" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {points.length > 0 && (
                <polygon
                  points={`${padLeft},${padTop + plotH} ${points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} ${padLeft + plotW},${padTop + plotH}`}
                  fill="url(#sedGrad)"
                />
              )}

              {/* Continuous Spectrum Curve */}
              <path
                d={pathD}
                fill="none"
                stroke="#c084fc"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Individual Band Data Points with Tooltips */}
              {points.map((p, i) => (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r={2.5}
                  fill="#f3e8ff"
                  stroke="#7e22ce"
                  strokeWidth="1"
                >
                  <title>{`Array ${p.pt.detector}: ${p.pt.wavelength_um}µm - ${p.pt.flux} DN`}</title>
                </circle>
              ))}
            </svg>
          </div>
        ) : (
          <div style={{ color: '#64748b', fontSize: '11px', textAlign: 'center', padding: '16px' }}>
            Awaiting observation data to compute spectra.
          </div>
        )}
      </div>
    </div>
  );
};
